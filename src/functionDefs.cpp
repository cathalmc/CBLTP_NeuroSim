#include "functionDefs.hpp"

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

// def calcWeightChange(p,Ca,rate,gammaP,gammaD,thetaP,thetaD):
//     #from the Graupner and Brunel paper
//     #rate = 1/tau 
//     pStar = 0.5
//     revP = (1-p)
//     fp = -p*revP*(pStar-p)
//     potentiation = gammaP*revP*float(Ca>thetaP)
//     depression = -gammaD*p*float(thetaD<Ca<=thetaP)
//     Noise = 0
//     return rate*fp+potentiation+depression+Noise

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
    
    constexpr float alpha = 1.0/(2.0*96485); //converts current to calcium diffusion, 
    constexpr float CaEquilibriumVal = 0.0f; // should be 1e-6, but keeping it 0 for consistency with otsuka
    constexpr float CaRemovalRate = 2.0f; //from otsuka
    //Single values 
    int nNeurons = systemInfo["nNeurons"].cast<int>();
    int nChannels = systemInfo["nChannels"].cast<int>();
    int nDiffusiveConnections = systemInfo["nDiffusiveConnections"].cast<int>();
    int maxDelayInt = systemInfo["maxDelayInt"].cast<int>();

    //Neuron data
    auto biasCurrentsBuffer = systemInfo["biasCurrents"].cast<py::array_t<float>>().request();
    float *biasCurrents = (float*)(biasCurrentsBuffer.ptr);


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
    auto preNeuronBuffer = systemInfo["preNeuron"].cast<py::array_t<int>>().request();
    int *preNeuron = static_cast<int*>(preNeuronBuffer.ptr);
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


    float gatingTotal;
    //assign temporary variables for readability
    float I,V,Ca,xs,xhs,xks,xts0,xts1,xths,xtks,Erev,g,gateVar,perceivedSpike;
    int preSynapticNeuron,postSynapticNeuron,readIndex;

    size_t sampleInd = 0;
    

    for (size_t timeInd=0;timeInd<nIterations;timeInd++){
        for (size_t idx=0;idx<nNeurons;idx++){
            netCurrent[idx] = 0.0;
            netCaCurrent[idx] = 0.0;
        }
        //Spike overwrite code needs to be implemented here 
        for (int idx=0;idx<nChannels;idx++){
            //temporary variables
            postSynapticNeuron = postNeuron[idx];
            V = voltages[postSynapticNeuron];
            Ca = calciumVals[postSynapticNeuron];
            preSynapticNeuron = preNeuron[idx];
            //spikes = spikeVals[postSynapticNeuron];

            g = conductance[idx] + conductanceGain[idx]*efficacy[idx];

            //multiply gating variables to get net gate opening
            gatingTotal = 1.0f;
            for (size_t idy=0;idy<nGatingVariables;idy++){
                float expnt = exponent[idx*chanStride + idy];
                gatingTotal *= pow(x[idx*chanStride + idy],expnt);
            }

            I = g*gatingTotal*(V-Erevs[idx]);// g*gatingTotal*(V-Erev);
            netCurrent[postSynapticNeuron] += I;
            netCaCurrent[postSynapticNeuron] += isThisCaCurrent[idx]*I;
            channelCurrent[idx] = I; //this is just for recording 
            
            readIndex = ((writeIndex[preSynapticNeuron] - inputDelay[idx]) % maxDelayInt + maxDelayInt) % maxDelayInt;
            //readIndex = (writeIndex[preSynapticNeuron]-inputDelay[idx])%maxDelayInt;
            perceivedSpike = (float)spikeBuffer[preSynapticNeuron*spikeBufferNeuronStride+readIndex]; //shit its a flattened 2d array, need to deal with the awkward indexing 
            //perceivedSpike = max(perceivedSpike,spikeOverwriteList[i]) //stimulateion 

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

        }
        for (size_t idx=0;idx<nDiffusiveConnections;idx++){
            int preSynapticNeuron = diffPreNeuron[idx];
            int postSynapticNeuron = diffPostNeuron[idx];

            float Vpost = voltages[postSynapticNeuron];
            float Capost = calciumVals[postSynapticNeuron];
            float Vpre = voltages[preSynapticNeuron];
            float Capre = calciumVals[preSynapticNeuron];

            float I = diffRate[idx]*(Vpost - Vpre);
            float CaFlow = diffRate[idx]*(Capost - Capre);

            netCurrent[postSynapticNeuron] -= I;
            netCaCurrent[postSynapticNeuron] += CaFlow*I;

            netCurrent[preSynapticNeuron] += I;
            netCaCurrent[preSynapticNeuron] -= CaFlow*I;

        }
            
        
        for (size_t idx=0;idx<nNeurons;idx++){
            constexpr float vThreshold = 0.0f;
            spikeBuffer[idx*spikeBufferNeuronStride + writeIndex[idx]] = (int) (voltages[idx]>vThreshold); //think L and R values for later 
            voltages[idx] -= timeStep*(netCurrent[idx]+biasCurrents[idx]);
            calciumVals[idx] += timeStep*(-alpha*netCaCurrent[idx] + (-calciumVals[idx] + CaEquilibriumVal)*CaRemovalRate);
            writeIndex[idx] = (writeIndex[idx]+1)%maxDelayInt;

        }
        //maybe add a guard for the case where there is only one sim iter, or deal with that on the python side
        if (timeInd%samplingInterval == 0){
            for (auto item : recordingInfo) {
                //Warning: AI generated 
                py::handle key = item.first;
                // py::array simArray = systemInfo[key].cast<py::array>();
                // py::array recArray = item.second.cast<py::array>();

                auto simBuf = systemInfo[key].cast<py::array_t<float>>().request();
                auto recBuf = recordingInfo[key].cast<py::array_t<float>>().request();

                auto simArray = (float*)simBuf.ptr;
                auto recArray = (float*)recBuf.ptr;
                
                if (recBuf.ndim != simBuf.ndim + 1)
                    throw std::runtime_error("Array dimension mismatch");

                // Ensure all inner dimensions match
                for (size_t i = 0; i < simBuf.ndim; ++i) {
                    if (recBuf.shape[i + 1] != simBuf.shape[i])
                        throw std::runtime_error("Shape mismatch between simDict and rDict");
                }
                // if (recBuf.size != simBuf.size){
                //     throw std::runtime_error("recBuf and simBuf are not the same size");
                // }

                size_t length = simBuf.size;
                for (size_t i = 0; i < length; ++i) {
                    recArray[sampleInd*length + i] = simArray[i];
                }

            }
            sampleInd++;
            
        }
    }


    for (auto item : recordingInfo) {
            //Warning: AI generated 
            py::handle key = item.first;
            py::print(key);
            auto recBuf = recordingInfo[key].cast<py::array_t<float>>().request();
            auto recArray = (float*)recBuf.ptr;

            size_t length = recBuf.size;
            for (size_t i = 0; i < length; ++i) {
                py::print(recArray[i]);
            }

        }
   
   
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