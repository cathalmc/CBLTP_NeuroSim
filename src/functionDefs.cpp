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

    //TODO: grab the single integer values for nChannels, nDiffusiveConnections, maxDelayInt
    int nChannels = -1;

    // get the number of neurons and number of channels from the voltage and neuron ids respectively
    auto voltageBuffer = systemInfo["voltages"].cast<py::array_t<float>>().request();
    float * voltages = (float *) voltageBuffer.ptr;
    int nNeurons = voltageBuffer.shape[0];

    auto postNeuronBuffer = systemInfo["postNeuron"].cast<py::array_t<int>>().request();
    int * postNeurons = (int *) postNeuronBuffer.ptr;
    int *preNeurons = (int *) systemInfo["preNeuron"].cast<py::array_t<int>>().request().ptr;

    int *diffPreNeuron = (int *) systemInfo["diffPreNeuron"].cast<py::array_t<int>>().request().ptr;
    int *diffPostNeuron = (int *) systemInfo["diffPostNeuron"].cast<py::array_t<int>>().request().ptr;
    float *diffRate = (float *) systemInfo["diffRate"].cast<py::array_t<float>>().request().ptr;

    // get the remaining values 
    //keys: ['x', 'vHalf', 'vKi', 'tau0', 'tau1', 'th', 'tkInv', , 'conductance', 'Erev', 'nChannels', 'voltages', 'calciumVals', 'spikeVals']
    auto xBuffer = systemInfo["x"].cast<py::array_t<float>>().request(); //.ptr;

    //TODO: consider changing this data structure to be multiple separate arrays
    size_t nChannels = xBuffer.shape[0];
    size_t nGatingVariables = xBuffer.shape[1];
    size_t chanStride = xBuffer.strides[0]/sizeof(float);
    size_t gateStride = xBuffer.strides[1]/sizeof(float);
    float *x = (float*) xBuffer.ptr;

    auto spikeBufferBuffer = systemInfo["spikeBuffer"].cast<py::array_t<bool>>().request();
    bool* spikeBuffer = (bool *) spikeBufferBuffer.ptr;
    int* writeIndex = (int *) systemInfo["writeIndex"].cast<py::array_t<int>>().request().ptr;
    

    //Ion channel data
    float *vHalf = (float *) systemInfo["vHalf"].cast<py::array_t<float>>().request().ptr;
    float *vKi = (float *) systemInfo["vKi"].cast<py::array_t<float>>().request().ptr;
    float *tau0 = (float *) systemInfo["tau0"].cast<py::array_t<float>>().request().ptr;
    float *tau1 = (float *) systemInfo["tau1"].cast<py::array_t<float>>().request().ptr;
    float *th = (float *) systemInfo["th"].cast<py::array_t<float>>().request().ptr;
    float *tkInv = (float *) systemInfo["tkInv"].cast<py::array_t<float>>().request().ptr;
    float *exponent = (float *) systemInfo["exponent"].cast<py::array_t<float>>().request().ptr;
    float *minOpenness = (float *) systemInfo["minOpenness"].cast<py::array_t<float>>().request().ptr;
    float *isThisACaCurrent = (float *) systemInfo["CaCurrent"].cast<py::array_t<float>>().request().ptr;
    float *inputDelay = (float *) systemInfo["inputDelay"].cast<py::array_t<float>>().request().ptr;

    //Plasticity data
    float *rate = (float *) systemInfo["rate"].cast<py::array_t<float>>().request().ptr;
    float *gammaP = (float *) systemInfo["gammaP"].cast<py::array_t<float>>().request().ptr;
    float *gammaD = (float *) systemInfo["gammaD"].cast<py::array_t<float>>().request().ptr;
    float *thetaP = (float *) systemInfo["thetaP"].cast<py::array_t<float>>().request().ptr;
    float *thetaD = (float *) systemInfo["thetaD"].cast<py::array_t<float>>().request().ptr;
    float *efficacy = (float *) systemInfo["efficacy"].cast<py::array_t<float>>().request().ptr;


    float *g0 = (float *) systemInfo["conductance"].cast<py::array_t<float>>().request().ptr;
    float *p = (float *) systemInfo["efficacy"].cast<py::array_t<float>>().request().ptr;
    float *w = (float *) systemInfo["conductanceGain"].cast<py::array_t<float>>().request().ptr;

    float *Erevs = (float *) systemInfo["Erev"].cast<py::array_t<float>>().request().ptr;
    float *calciumVals = (float *) systemInfo["calciumVals"].cast<py::array_t<float>>().request().ptr;
    float *spikeVals = (float *) systemInfo["spikeVals"].cast<py::array_t<float>>().request().ptr;
    float *netCurrents = (float *) systemInfo["netCurrent"].cast<py::array_t<float>>().request().ptr;

    //To implement:
    //  spike base stimulation
    //  

    //std::vector<float> netCurrents(nNeurons, 0.0);
    // py::print("nChannels",nChannels,"nGatingVariables",nGatingVariables);
    // py::print("chanStride",chanStride,"gateStride",gateStride);
    float gatingTotal;
    //assign temporary variables for readability
    float I,V,Ca,spikes,xs,xhs,xks,xts0,xts1,xths,xtks,Erev,g,gateVar;
    int neurId;

    for (size_t idx=0;idx<nNeurons;idx++){
        netCurrents[idx] = 0.0;
    }

    for (int idx=0;idx<nChannels;idx++){
        //temporary variables
        neurId = postNeurons[idx];
        V = voltages[neurId];
        Ca = calciumVals[neurId];
        spikes = spikeVals[neurId];

        Erev = Erevs[idx];
        g = conductance[idx];

        //multiply gating variables to get net gate opening
        gatingTotal = 1.0f;
        for (size_t idy=0;idy<nGatingVariables;idy++){
            gatingTotal *= x[idx*chanStride + idy];
        }
        I = -g*gatingTotal*(V-Erev);// g*gatingTotal*(V-Erev);
        netCurrents[neurId] += I;
        
        for (size_t idy=0;idy<nGatingVariables;idy++) {
            if (idy<2) gateVar = V;
            else if (idy==2) gateVar = Ca; //Ca
            else gateVar = spikes; //Spikes

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