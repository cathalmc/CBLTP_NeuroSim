#include "functionDefs.hpp"
#include <execution>
#include <chrono>
#include <random>
#include <algorithm>
#include <functional>

PYBIND11_MODULE(cathalTest, m) {
    m.def("testFunc",&testFunc,"run my lil test function",
    "testIntVal"_a,"testNParr"_a);
    m.def("solveNeuralSystem",&solveNeuralSystem,"test accepting dictionaries",
    "myDict"_a,"recordingInfo"_a, "timeStep"_a,"nIterations"_a,"samplingInterval"_a);
}

inline float sigmoid(float x, float xh, float xk,float minOp){
    return minOp + (1.0f/(1.0f + expf((xh-x)*xk))); //normally you divide by K, but I precalculate the inverse to save on computation. 
    //However debugging this has cost me so much time that it probably wasn't worth it.
}

inline float solveGate(float x, float s, float TS, float xh, float xk, float xts0, 
        float xts1, float xth, float xtk,float minOpenness){
    float xInf = sigmoid(s,xh,xk,minOpenness);
    float tau = xts0 + xts1*sigmoid(s,xth,xtk,0.0f);
    //return x + TS*(xInf-x)/tau;
    return (tau*x +TS*xInf)/(tau+TS);
}

inline float calcWeightChange(float p, float Ca, float rate, float gammaP, float gammaD, float thetaP, float thetaD){
    float pStar = 0.5;
    float revP = (1-p);
    float fp = -p*revP*(pStar-p);
    float potentiation = gammaP*revP*(Ca>thetaP);
    float depression = -gammaD*p*(thetaD<Ca && Ca<=thetaP);
    float Noise = 0;
    return rate*fp+potentiation+depression+Noise;
}


py::object solveNeuralSystem(py::dict systemInfo,py::dict recordingInfo,
                            float timeStep,int nIterations,int samplingInterval) {
    //systemInfo is a dict of numpy arrays, we need to extract these arrays and process them in place
    std::random_device rd;  // seed
    std::mt19937 gen(rd()); // Mersenne Twister engine
    std::uniform_real_distribution<float> uniformFloatDist(0.0f, 1.0f);

    constexpr float alpha = 1.0/(2.0*96485); //converts current to calcium diffusion, 
    // constexpr float CaEquilibriumVal = 0.0f; // should be 1e-6, but keeping it 0 for consistency with otsuka
    // constexpr float CaRemovalRate = 2.0f; //from otsuka
    //Single values 
    int nNeurons = systemInfo["nNeurons"].cast<int>();
    int nSpikers = systemInfo["nSpikers"].cast<int>();
    int nChannels = systemInfo["nChannels"].cast<int>();
    int nDiffusiveConnections = systemInfo["nDiffusiveConnections"].cast<int>();
    int maxDelayInt = systemInfo["maxDelayInt"].cast<int>();

    //Neuron data
    auto biasCurrentsBuffer = systemInfo["biasCurrents"].cast<py::array_t<float>>().request();
    float *biasCurrents = (float*)(biasCurrentsBuffer.ptr);
    auto CaEquilibriumValBuffer = systemInfo["CaEquilibriumVal"].cast<py::array_t<float>>().request();
    float *CaEquilibriumVal = (float*)(CaEquilibriumValBuffer.ptr);
    auto CaRemovalRateBuffer = systemInfo["CaRemovalRate"].cast<py::array_t<float>>().request();
    float *CaRemovalRate = (float*)(CaRemovalRateBuffer.ptr);

    //Neuron data
    auto spikerPreNeuronBuffer = systemInfo["spikerPreNeuron"].cast<py::array_t<int>>().request();
    int *spikerPreNeuron = (int*)(spikerPreNeuronBuffer.ptr);
    auto spikerNeuronScaleBuffer = systemInfo["spikerNeuronScale"].cast<py::array_t<float>>().request();
    float *spikerNeuronScale = (float*)(spikerNeuronScaleBuffer.ptr);
    auto spikerProbabilityBuffer = systemInfo["spikerProbability"].cast<py::array_t<float>>().request();
    float *spikerProbability = (float*)(spikerProbabilityBuffer.ptr);

    //bool arrays
    auto spikeBufferBuffer = systemInfo["spikeBuffer"].cast<py::array_t<bool, py::array::c_style>>().request();
    int psuedoMaxDelayInt = spikeBufferBuffer.shape[1];
    size_t spikeBufferNeuronStride = spikeBufferBuffer.strides[0]/sizeof(bool);
    size_t spikeBufferTimeStride = spikeBufferBuffer.strides[1]/sizeof(bool);
    bool* spikeBuffer = (bool *) spikeBufferBuffer.ptr;

    //assert(maxDelayInt==psuedoMaxDelayInt); //is this best practice in c++?
    if (maxDelayInt != psuedoMaxDelayInt)
    throw std::runtime_error("maxDelayInt mismatch detected!");
    
    //integer arrays
    //intKeys = ["preNeuron","postNeuron","diffPostNeuron","diffPreNeuron","writeIndex", "inputDelay"]
    auto channelPreSpikerBuffer = systemInfo["channelPreSpiker"].cast<py::array_t<int>>().request();
    int *channelPreSpiker = static_cast<int*>(channelPreSpikerBuffer.ptr);
    auto postNeuronBuffer = systemInfo["postNeuron"].cast<py::array_t<int>>().request();
    int *postNeuron = static_cast<int*>(postNeuronBuffer.ptr);
    auto diffPreNeuronBuffer = systemInfo["diffPreNeuron"].cast<py::array_t<int>>().request();
    int *diffPreNeuron = static_cast<int*>(diffPreNeuronBuffer.ptr);
    auto diffPostNeuronBuffer = systemInfo["diffPostNeuron"].cast<py::array_t<int>>().request();
    int *diffPostNeuron = static_cast<int*>(diffPostNeuronBuffer.ptr);
    auto writeIndexBuffer = systemInfo["writeIndex"].cast<py::array_t<int>>().request();
    int *writeIndex = static_cast<int*>(writeIndexBuffer.ptr);
    auto inputDelayBuffer = systemInfo["inputDelay"].cast<py::array_t<int>>().request();
    int *inputDelay = static_cast<int*>(inputDelayBuffer.ptr);


    //Ion channel data
    //keys: [''voltages', 'calciumVals', 'spikeVals', "CaCurrent", 
    auto voltageBuffer = systemInfo["voltages"].cast<py::array_t<float>>().request();
    float * voltages = (float *) voltageBuffer.ptr;
    if (voltageBuffer.size < nNeurons) throw std::runtime_error("voltages array too small");
    auto diffRateBuffer = systemInfo["diffRate"].cast<py::array_t<float>>().request();
    float * diffRate = (float *) diffRateBuffer.ptr;
    auto calciumValsBuffer = systemInfo["calciumVals"].cast<py::array_t<float>>().request();
    float * calciumVals = (float *) calciumValsBuffer.ptr;
    auto spikeValsBuffer = systemInfo["spikeVals"].cast<py::array_t<float>>().request();
    float * spikeVals = (float *) spikeValsBuffer.ptr;
    auto isThisCaCurrentBuffer = systemInfo["CaCurrent"].cast<py::array_t<float>>().request();
    float * isThisCaCurrent = (float *) isThisCaCurrentBuffer.ptr;
    auto netCurrentBuffer = systemInfo["netCurrent"].cast<py::array_t<float>>().request();
    float * netCurrent = (float *) netCurrentBuffer.ptr;
    auto netCaCurrentBuffer = systemInfo["netCaCurrent"].cast<py::array_t<float>>().request();
    float * netCaCurrent = (float *) netCaCurrentBuffer.ptr;
    auto channelCurrentBuffer = systemInfo["current"].cast<py::array_t<float>>().request();
    float * channelCurrent = (float *) channelCurrentBuffer.ptr;
    auto CaChannelCurrentBuffer = systemInfo["CaChannelCurrent"].cast<py::array_t<float>>().request();
    float * CaChannelCurrent = (float *) CaChannelCurrentBuffer.ptr;
    auto debugRecordingBuffer = systemInfo["debugRecording"].cast<py::array_t<float>>().request();
    float * debugRecording = (float *) debugRecordingBuffer.ptr;
    
    //gating data 
    //x', 'vHalf', 'vKi', 'tau0', 'tau1', 'th', 'tkInv', 'exponent',"minOpenness" , 
    auto xBuffer = systemInfo["x"].cast<py::array_t<float>>().request(); //.ptr;
    //TODO: consider changing this data structure to be multiple separate arrays
    size_t nGatingVariables = xBuffer.shape[1];
    size_t chanStride = xBuffer.strides[0]/sizeof(float);
    size_t gateStride = xBuffer.strides[1]/sizeof(float);
    float *x = (float*) xBuffer.ptr;

    auto vHalfBuffer = systemInfo["vHalf"].cast<py::array_t<float>>().request();
    float *vHalf = (float *) vHalfBuffer.ptr;
    auto vKiBuffer = systemInfo["vKi"].cast<py::array_t<float>>().request();
    float *vKi = (float *) vKiBuffer.ptr;
    auto tau0Buffer = systemInfo["tau0"].cast<py::array_t<float>>().request();
    float *tau0 = (float *) tau0Buffer.ptr;
    auto tau1Buffer = systemInfo["tau1"].cast<py::array_t<float>>().request();
    float *tau1 = (float *) tau1Buffer.ptr;
    auto thBuffer = systemInfo["th"].cast<py::array_t<float>>().request();
    float *th = (float *) thBuffer.ptr;
    auto tkInvBuffer = systemInfo["tkInv"].cast<py::array_t<float>>().request();
    float *tkInv = (float *) tkInvBuffer.ptr;
    auto exponentBuffer = systemInfo["exponent"].cast<py::array_t<float>>().request();
    float *exponent = (float *) exponentBuffer.ptr;
    auto minOpennessBuffer = systemInfo["minOpenness"].cast<py::array_t<float>>().request();
    float *minOpenness = (float *) minOpennessBuffer.ptr;


    //Synapse data
    // "rate","gammaP","gammaD", "thetaP","thetaD",'conductance',"efficacy","conductanceGain" 'Erev',  ]
    auto rateBuffer = systemInfo["rate"].cast<py::array_t<float>>().request();
    float *rate = (float *) rateBuffer.ptr;
    auto gammaPBuffer = systemInfo["gammaP"].cast<py::array_t<float>>().request();
    float *gammaP = (float *) gammaPBuffer.ptr;
    auto gammaDBuffer = systemInfo["gammaD"].cast<py::array_t<float>>().request();
    float *gammaD = (float *) gammaDBuffer.ptr;
    auto thetaPBuffer = systemInfo["thetaP"].cast<py::array_t<float>>().request();
    float *thetaP = (float *) thetaPBuffer.ptr;
    auto thetaDBuffer = systemInfo["thetaD"].cast<py::array_t<float>>().request();
    float *thetaD = (float *) thetaDBuffer.ptr;
    auto conductanceBuffer = systemInfo["conductance"].cast<py::array_t<float>>().request();
    float *conductance = (float *) conductanceBuffer.ptr;
    auto efficacyBuffer = systemInfo["efficacy"].cast<py::array_t<float>>().request();
    float *efficacy = (float *) efficacyBuffer.ptr;
    auto conductanceGainBuffer = systemInfo["conductanceGain"].cast<py::array_t<float>>().request();
    float *conductanceGain = (float *) conductanceGainBuffer.ptr;
    auto ErevBuffer = systemInfo["Erev"].cast<py::array_t<float>>().request();
    float *Erevs = (float *) ErevBuffer.ptr;

    //Stimulation data 
    // Spike stimulation
    std::vector<float> spikeInputOverwriteVector(nChannels, 0.0f);
    auto spikeStimTimesBuffer = systemInfo["spikeStimTimes"].cast<py::array_t<int>>().request();
    int *spikeStimTimes = (int*)(spikeStimTimesBuffer.ptr);
    auto spikeStimNeuronsBuffer = systemInfo["spikeStimNeurons"].cast<py::array_t<int>>().request();
    int *spikeStimNeurons = (int*)(spikeStimNeuronsBuffer.ptr);
    size_t maxSpikeStimInd = spikeStimTimesBuffer.shape[0];
    size_t spikeStimArrayIter = 0; //used to iterate through the spike stimulation arrays
    
    //Bias current stimulation
    std::vector<float> biasCurrentOverwriteVector(nNeurons, 0.0f);
    //have biasStimTimes, biasStimSegments biasStimValues
    auto biasStimTimesBuffer = systemInfo["biasStimTimes"].cast<py::array_t<int>>().request();
    int *biasStimTimes = (int*)(biasStimTimesBuffer.ptr);
    auto biasStimSegmentsBuffer = systemInfo["biasStimSegments"].cast<py::array_t<int>>().request();
    int *biasStimSegments = (int*)(biasStimSegmentsBuffer.ptr);
    auto biasStimValuesBuffer = systemInfo["biasStimValues"].cast<py::array_t<float>>().request();
    float *biasStimValues = (float*)(biasStimValuesBuffer.ptr);
    size_t maxBiasStimInd = biasStimTimesBuffer.shape[0];
    size_t biasStimArrayIter = 0; //used to iterate through the bias stimulation arrays


    float gatingTotal;
    //assign temporary variables for readability
    float I,V,Ca,xs,xhs,xks,xts0,xts1,xths,xtks,Erev,g,gateVar,perceivedSpike;
    size_t preSynapticNeuron,postSynapticNeuron,readIndex;

    size_t sampleInd = 0;
    
    std::vector<int> indices(nChannels);
    std::iota(indices.begin(), indices.end(), 0); // Fill it with consecutive numbers

    //auto executionPolicy = std::execution::seq;
    // std::for_each(executionPolicy, indices.begin(), indices.end(), [&](int connInd)  {

    // });
    auto executionPolicy = std::execution::par_unseq;

    auto start = std::chrono::high_resolution_clock::now();
    for (size_t timeInd=0;timeInd<nIterations;timeInd++){
        for (size_t idx=0;idx<nNeurons;idx++){
            netCurrent[idx] = 0.0;
            netCaCurrent[idx] = 0.0;
        }
        //Spike overwrite code needs to be implemented here 
        std::fill(spikeInputOverwriteVector.begin(), spikeInputOverwriteVector.end(), 0.0f);
        // Spike overwrite:
        // Create spikeStimTimes,spikeStimNeurons vectors, and use a while loop with those
        while (spikeStimArrayIter<maxSpikeStimInd && timeInd == spikeStimTimes[spikeStimArrayIter])
        {
            int syn = spikeStimNeurons[spikeStimArrayIter];
            spikeInputOverwriteVector[syn] = 1.0f;
            spikeStimArrayIter++;
        }

        //Bias current overwrite 
        std::fill(biasCurrentOverwriteVector.begin(), biasCurrentOverwriteVector.end(), 0.0f);
        while (biasStimArrayIter<maxBiasStimInd && timeInd == biasStimTimes[biasStimArrayIter])
        {
            int neuron = biasStimSegments[biasStimArrayIter];
            float biasVal = biasStimValues[biasStimArrayIter];
            biasCurrentOverwriteVector[neuron] = biasVal;
            biasStimArrayIter++;
        }
        
        //Solve Channels 
        std::iota(indices.begin(), indices.end(), 0); // Fill it with consecutive numbers
        std::for_each(executionPolicy, indices.begin(), indices.end(), [&](int connInd)  {
            //temporary variables
            int idx = connInd;
            postSynapticNeuron = postNeuron[idx];
            V = voltages[postSynapticNeuron];
            Ca = calciumVals[postSynapticNeuron];
            preSynapticNeuron = channelPreSpiker[idx];
            //spikes = spikeVals[postSynapticNeuron];

            g = conductance[idx] + conductanceGain[idx]*efficacy[idx];

            //multiply gating variables to get net gate opening
            gatingTotal = 1.0f;
            for (size_t idy=0;idy<nGatingVariables;idy++){
                float expnt = exponent[idx*chanStride + idy];
                gatingTotal *= pow(x[idx*chanStride + idy],expnt);
            }

            I = g*gatingTotal*(V-Erevs[idx]);// g*gatingTotal*(V-Erev);
            // netCurrent[postSynapticNeuron] += I;
            // netCaCurrent[postSynapticNeuron] += isThisCaCurrent[idx]*I;
            channelCurrent[idx] = I; //we accumulate later
            CaChannelCurrent[idx] = isThisCaCurrent[idx]*I;
            debugRecording[idx] = gatingTotal;

            readIndex = ((writeIndex[preSynapticNeuron] - inputDelay[idx]) % maxDelayInt + maxDelayInt) % maxDelayInt;
            //readIndex = (writeIndex[preSynapticNeuron]-inputDelay[idx])%maxDelayInt;
            perceivedSpike = (float)spikeBuffer[preSynapticNeuron*spikeBufferNeuronStride+readIndex]; //shit its a flattened 2d array, need to deal with the awkward indexing 
            perceivedSpike = std::max(perceivedSpike,spikeInputOverwriteVector[idx]); //stimulateion 

            float dp = calcWeightChange(efficacy[idx],Ca,rate[idx],gammaP[idx],gammaD[idx],thetaP[idx],thetaD[idx]);
            efficacy[idx] +=timeStep*dp;

            for (size_t idy=0;idy<nGatingVariables;idy++) {
                if (idy<2) gateVar = V;
                else if (idy==2) gateVar = Ca; //Ca
                else gateVar = perceivedSpike; //Spikes

                float xp = x[idx*chanStride + idy];
                float vHalfp = vHalf[idx*chanStride + idy];
                float vKip = vKi[idx*chanStride + idy];
                float tau0p = tau0[idx*chanStride + idy];
                float tau1p = tau1[idx*chanStride + idy];
                float thp = th[idx*chanStride + idy];
                float tkInvp = tkInv[idx*chanStride + idy];
                float minOp = minOpenness[idx*chanStride + idy];

                x[idx*chanStride + idy] = solveGate(xp,gateVar,timeStep,vHalfp,vKip,tau0p,tau1p,thp,tkInvp,minOp);


            }
        });

        // for (int idx=0;idx<nChannels;idx++){
        //     //temporary variables
        //     postSynapticNeuron = postNeuron[idx];
        //     V = voltages[postSynapticNeuron];
        //     Ca = calciumVals[postSynapticNeuron];
        //     preSynapticNeuron = channelPreSpiker[idx];
        //     //spikes = spikeVals[postSynapticNeuron];

        //     g = conductance[idx] + conductanceGain[idx]*efficacy[idx];

        //     //multiply gating variables to get net gate opening
        //     gatingTotal = 1.0f;
        //     for (size_t idy=0;idy<nGatingVariables;idy++){
        //         float expnt = exponent[idx*chanStride + idy];
        //         gatingTotal *= pow(x[idx*chanStride + idy],expnt);
        //     }

        //     I = g*gatingTotal*(V-Erevs[idx]);// g*gatingTotal*(V-Erev);
        //     // netCurrent[postSynapticNeuron] += I;
        //     // netCaCurrent[postSynapticNeuron] += isThisCaCurrent[idx]*I;
        //     channelCurrent[idx] = I; //we accumulate later
        //     CaChannelCurrent[idx] = isThisCaCurrent[idx]*I;
        //     debugRecording[idx] = gatingTotal;

        //     readIndex = ((writeIndex[preSynapticNeuron] - inputDelay[idx]) % maxDelayInt + maxDelayInt) % maxDelayInt;
        //     //readIndex = (writeIndex[preSynapticNeuron]-inputDelay[idx])%maxDelayInt;
        //     perceivedSpike = (float)spikeBuffer[preSynapticNeuron*spikeBufferNeuronStride+readIndex]; //shit its a flattened 2d array, need to deal with the awkward indexing 
        //     perceivedSpike = std::max(perceivedSpike,spikeInputOverwriteVector[idx]); //stimulateion 

        //     float dp = calcWeightChange(efficacy[idx],Ca,rate[idx],gammaP[idx],gammaD[idx],thetaP[idx],thetaD[idx]);
        //     efficacy[idx] +=timeStep*dp;

        //     for (size_t idy=0;idy<nGatingVariables;idy++) {
        //         if (idy<2) gateVar = V;
        //         else if (idy==2) gateVar = Ca; //Ca
        //         else gateVar = perceivedSpike; //Spikes

        //         float xp = x[idx*chanStride + idy];
        //         float vHalfp = vHalf[idx*chanStride + idy];
        //         float vKip = vKi[idx*chanStride + idy];
        //         float tau0p = tau0[idx*chanStride + idy];
        //         float tau1p = tau1[idx*chanStride + idy];
        //         float thp = th[idx*chanStride + idy];
        //         float tkInvp = tkInv[idx*chanStride + idy];
        //         float minOp = minOpenness[idx*chanStride + idy];

        //         x[idx*chanStride + idy] = solveGate(xp,gateVar,timeStep,vHalfp,vKip,tau0p,tau1p,thp,tkInvp,minOp);


        //     }

        // }
        
        //accumulate in a separate loop to allow for parallelizing the above loop
        for (int idx=0;idx<nChannels;idx++){
            postSynapticNeuron = postNeuron[idx];
            I = channelCurrent[idx];
            netCurrent[postSynapticNeuron] += I;
            netCaCurrent[postSynapticNeuron] += isThisCaCurrent[idx]*I;
        }

        //Solve diffusive connections. could also be parallelized, but its relatively not as big
        for (size_t idx=0;idx<nDiffusiveConnections;idx++){
            int preSynapticNeuron = diffPreNeuron[idx];
            int postSynapticNeuron = diffPostNeuron[idx];

            float Vpost = voltages[postSynapticNeuron];
            float Capost = calciumVals[postSynapticNeuron];
            float Vpre = voltages[preSynapticNeuron];
            float Capre = calciumVals[preSynapticNeuron];

            float I = diffRate[idx]*(Vpost - Vpre);
            float CaFlow = diffRate[idx]*(Capost - Capre);

            netCurrent[postSynapticNeuron] += I;
            netCaCurrent[postSynapticNeuron] += CaFlow*I;

            netCurrent[preSynapticNeuron] -= I;
            netCaCurrent[preSynapticNeuron] -= CaFlow*I;

        }
            
        //Solve neurons
        for (size_t idx=0;idx<nNeurons;idx++){
            voltages[idx] -= timeStep*(netCurrent[idx]+biasCurrents[idx] +biasCurrentOverwriteVector[idx] ); //+bias overwrite list 
            calciumVals[idx] += timeStep*(-alpha*netCaCurrent[idx] +
            (-calciumVals[idx] + CaEquilibriumVal[idx])*CaRemovalRate[idx]);
        }

        //Solve spikers
        constexpr float vThreshold = 0.0f;
        for (size_t idx=0;idx<nSpikers;idx++){
            //29/7 continue implenting spikers 
            float r = uniformFloatDist(gen);
            bool randomOutput = r<spikerProbability[idx];
            bool spikeOutput = spikerNeuronScale[idx] * (float)((voltages[idx]>vThreshold));
            spikeBuffer[idx*spikeBufferNeuronStride + writeIndex[idx]] = randomOutput || spikeOutput; //think L and R values for later 
            writeIndex[idx] = (writeIndex[idx]+1)%maxDelayInt;
        }

        //maybe add a guard for the case where there is only one sim iter, or deal with that on the python side
        if (timeInd % samplingInterval == 0) {
            //Warning: AI generated
            for (auto& [keyHandle, recObj] : recordingInfo) {
                py::handle key = keyHandle;

                auto simArr = systemInfo[key].cast<py::array_t<float, py::array::c_style | py::array::forcecast>>();
                auto recArr = recordingInfo[key].cast<py::array_t<float, py::array::c_style | py::array::forcecast>>();

                auto simReq = simArr.request();
                auto recReq = recArr.request();

                if (recReq.ndim != simReq.ndim + 1)
                    throw std::runtime_error("Array dimension mismatch");

                for (size_t i = 0; i < simReq.ndim; ++i) {
                    if (recReq.shape[i + 1] != simReq.shape[i])
                        throw std::runtime_error("Shape mismatch between simDict and rDict");
                }

                size_t innerSize = 1;
                for (size_t i = 0; i < simReq.ndim; ++i)
                    innerSize *= static_cast<size_t>(simReq.shape[i]);

                size_t offsetElems = static_cast<size_t>(sampleInd) * innerSize;
                float* recPtr = static_cast<float*>(recReq.ptr);
                float* simPtr = static_cast<float*>(simReq.ptr);

                std::memcpy(recPtr + offsetElems, simPtr, innerSize * sizeof(float));
            }
            sampleInd++;
        }
        
    }
    // auto arr = py::array_t<float>({2, 3});
    // recordingInfo["test"] = arr;
    // auto buf = arr.mutable_unchecked<2>();
    // buf(0, 0) = 42.0f;
    // py::print(recordingInfo["test"]);


    // for (auto item : recordingInfo) {
    //         //Warning: AI generated 
    //         py::handle key = item.first;
    //         py::print(key);
    //         auto recBuf = recordingInfo[key].cast<py::array_t<float>>().request();
    //         auto recArray = (float*)recBuf.ptr;
    //         py::print("recordingInfo[key] is:", recordingInfo[key].get_type());
    //         size_t length = recBuf.size;
    //         py::print("After:", recordingInfo[key]);
    //     }
   
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    py::print("Elapsed time in seconds:", elapsed.count());

    return py::none();
}

py::object testFunc(py::array_t<int> testIntVal,
                    py::array_t<double> testNParr) 
    {
        int testInt = *testIntVal.data();
        auto doubArr = (double *) testNParr.request().ptr;
        doubArr[testInt] = 2.5;

        return py::cast<py::none>(Py_None);
    }