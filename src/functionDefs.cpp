#include "functionDefs.hpp"


PYBIND11_MODULE(cathalTest, m) {
    m.def("testFunc",&testFunc,"run my lil test function",
    "testIntVal"_a,"testNParr"_a);
    m.def("solveNeuralSystem",&solveNeuralSystem,"test accepting dictionaries",
    "myDict"_a,"nIterations"_a,"timeStep"_a);
}

inline float sigmoid(float x, float xh, float xk){
    return 1.0f/(1.0f + expf((xh-x)*xk)); //normally you divide by K, but I precalculate the inverse to save on computation. 
    //However debugging this has cost me so much time that it probably wasn't worth it.
}

inline float solveGate(float x, float s, float TS, float xh, float xk, float xts0, float xts1, float xth, float xtk){
    float xInf = sigmoid(s,xh,xk);
    float tau = xts0 + xts1*sigmoid(s,xth,xtk);
    return x + TS*(xInf-x)/tau;
}



py::object solveNeuralSystem(py::dict systemInfo, int nIterations, float timeStep) {
    //systemInfo is a dict of numpy arrays, we need to extract these arrays and process them in place
    // for (auto item : systemInfo) {
    //     py::print("Key:", item.first, "Value:", item.second);
    // }

    //Single values 
    int nNeurons = systemInfo["nNeurons"].cast<int>();
    int nChannels = systemInfo["nChannels"].cast<int>();
    int nDiffusiveConnections = systemInfo["nDiffusiveConnections"].cast<int>();
    int maxDelayInt = systemInfo["maxDelayInt"].cast<int>();

    //bool arrays
    auto spikeBufferBuffer = systemInfo["spikeBuffer"].cast<py::array_t<bool>>().request();
    bool* spikeBuffer = (bool *) spikeBufferBuffer.ptr;


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
            gatingTotal *= x[idx*chanStride + idy];
        }

        I = -g*gatingTotal*(V-Erevs[idx]);// g*gatingTotal*(V-Erev);
        netCurrent[postSynapticNeuron] += I;
        netCaCurrent[postSynapticNeuron] += isThisCaCurrent[idx]*I;
        
        readIndex = (writeIndex[preSynapticNeuron]-inputDelay[idx])%maxDelayInt;
        perceivedSpike = (float)spikeBuffer[preSynapticNeuron][readIndex]; //shit its a flattened 2d array, need to deal with the awkward indexing 
        //perceivedSpike = max(perceivedSpike,spikeOverwriteList[i]) //stimulateion 


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

            x[idx*chanStride + idy] = solveGate(xp,gateVar,timeStep,vHalfp,vKip,tau0p,tau1p,thp,tkInvp);


        }

        
    }
    for (size_t idx=0;idx<nNeurons;idx++){
        voltages[idx] += timeStep*netCurrents[idx];
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