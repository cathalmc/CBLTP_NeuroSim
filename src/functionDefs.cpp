#include "functionDefs.hpp"


PYBIND11_MODULE(cathalTest, m) {
    m.def("testFunc",&testFunc,"run my lil test function",
    "testIntVal"_a,"testNParr"_a);
    m.def("solveNeuralSystem",&solveNeuralSystem,"test accepting dictionaries",
    "myDict"_a,"nIterations"_a,"timeStep"_a);
}

inline float sigmoid(float x, float xh, float xk){
    return 1.0f/(1.0f + expf((xh-x)/xk));
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

    // get the number of neurons and number of channels from the voltage and neuron ids respectively
    auto voltageBuffer = systemInfo["voltages"].cast<py::array_t<float>>().request();
    float * voltages = (float *) voltageBuffer.ptr;
    int nNeurons = voltageBuffer.shape[0];

    auto neuronIdBuffer = systemInfo["neuronId"].cast<py::array_t<int>>().request();
    int * neuronIds = (int *) neuronIdBuffer.ptr;
    

    // get the remaining values 
    //keys: ['x', 'vHalf', 'vKi', 'tau0', 'tau1', 'th', 'tkInv', , 'conductance', 'Erev', 'nChannels', 'voltages', 'calciumVals', 'spikeVals']
    auto xBuffer = systemInfo["x"].cast<py::array_t<float>>().request(); //.ptr;
    size_t nChannels = xBuffer.shape[0];
    size_t nGatingVariables = xBuffer.shape[1];
    size_t chanStride = xBuffer.strides[0]/sizeof(float);
    size_t gateStride = xBuffer.strides[1]/sizeof(float);
    float *x = (float*) xBuffer.ptr;


    float *vHalf = (float *) systemInfo["vHalf"].cast<py::array_t<float>>().request().ptr;
    float *vKi = (float *) systemInfo["vKi"].cast<py::array_t<float>>().request().ptr;
    float *tau0 = (float *) systemInfo["tau0"].cast<py::array_t<float>>().request().ptr;
    float *tau1 = (float *) systemInfo["tau1"].cast<py::array_t<float>>().request().ptr;
    float *th = (float *) systemInfo["th"].cast<py::array_t<float>>().request().ptr;
    float *tkInv = (float *) systemInfo["tkInv"].cast<py::array_t<float>>().request().ptr;
    float *conductance = (float *) systemInfo["conductance"].cast<py::array_t<float>>().request().ptr;
    float *Erevs = (float *) systemInfo["Erev"].cast<py::array_t<float>>().request().ptr;
    float *calciumVals = (float *) systemInfo["calciumVals"].cast<py::array_t<float>>().request().ptr;
    float *spikeVals = (float *) systemInfo["spikeVals"].cast<py::array_t<float>>().request().ptr;
    float *netCurrents = (float *) systemInfo["netCurrent"].cast<py::array_t<float>>().request().ptr;

    //std::vector<float> netCurrents(nNeurons, 0.0);
    // py::print("nChannels",nChannels,"nGatingVariables",nGatingVariables);
    // py::print("chanStride",chanStride,"gateStride",gateStride);
    float gatingTotal;
    //assign temporary variables for readability
    float I,V,Ca,spikes,xs,xhs,xks,xts0,xts1,xths,xtks,Erev,g;
    int neurId;
    for (int idx=0;idx<nChannels;idx++){
        //temporary variables
        neurId = neuronIds[idx];
        V = voltages[neurId];
        Ca = calciumVals[neurId];
        spikes = spikeVals[neurId];

        // xs = x[idx*chanStride];
        // xhs = vHalf[idx*chanStride];
        // xks = vKi[idx*chanStride];
        // xts0 = tau0[idx*chanStride];
        // xts1 = tau1[idx*chanStride];
        // xths = th[idx*chanStride];
        // xtks = tkInv[idx*chanStride];
        Erev = Erevs[idx];
        g = conductance[idx];

        //multiply gating variables to get net gate opening
        gatingTotal = 1.0f;
        for (size_t idy=0;idy<nGatingVariables;idy++){
            gatingTotal *= x[idx*chanStride + idy];
        }
        I = -g*gatingTotal*(V-Erev);
        netCurrents[neurId] += I;
        
        for (size_t idy=0;idy<nGatingVariables;idy++) {
            x[idx*chanStride + idy] = solveGate(x[idx*chanStride + idy],V,timeStep,
                                                vHalf[idx*chanStride + idy],vKi[idx*chanStride + idy],
                                                tau0[idx*chanStride + idy],tau1[idx*chanStride + idy],
                                                th[idx*chanStride + idy],tkInv[idx*chanStride + idy]); 
        }

        // gatingTotal = 1.0f;
        // for (int idy=0;idy<nGatingVariables;idy++){
        //     gatingTotal *= x[idx*nGatingVariables + idy];
        // }
        
    }
    for (size_t idx=0;idx<nNeurons;idx++){
        voltages[idx] += timeStep*netCurrents[idx];
        netCurrents[idx] = 0.0;
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