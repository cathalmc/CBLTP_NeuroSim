#include "functionDefs.hpp"


PYBIND11_MODULE(cathalTest, m) {
    m.def("testFunc",&testFunc,"run my lil test function",
    "testIntVal"_a,"testNParr"_a);
    m.def("solveNeuralSystem",&solveNeuralSystem,"test accepting dictionaries",
    "myDict"_a,"nIterations"_a,"timeStep"_a);
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
    int nChannels = neuronIdBuffer.shape[0];

    // get the remaining values 
    //keys: ['x', 'vHalf', 'vKi', 'tau0', 'tau1', 'th', 'tkInv', , 'conductance', 'Erev', 'nChannels', 'voltages', 'calciumVals', 'spikeVals']
    auto xBuffer = systemInfo["x"].cast<py::array_t<float>>().request(); //.ptr;
    size_t nGatingVariables = xBuffer.shape[1];
    size_t chanStride = xBuffer.strides[0]/sizeof(float);
    size_t gateStride = xBuffer.strides[0]/sizeof(float);
    float *x = (float*) xBuffer.ptr;



    float *vHalf = (float *) systemInfo["vHalf"].cast<py::array_t<float>>().request().ptr;
    float *vKi = (float *) systemInfo["vKi"].cast<py::array_t<float>>().request().ptr;
    float *tau0 = (float *) systemInfo["tau0"].cast<py::array_t<float>>().request().ptr;
    float *tau1 = (float *) systemInfo["tau1"].cast<py::array_t<float>>().request().ptr;
    float *th = (float *) systemInfo["th"].cast<py::array_t<float>>().request().ptr;
    float *tkInv = (float *) systemInfo["tkInv"].cast<py::array_t<float>>().request().ptr;
    float *conductance = (float *) systemInfo["conductance"].cast<py::array_t<float>>().request().ptr;
    float *Erev = (float *) systemInfo["Erev"].cast<py::array_t<float>>().request().ptr;
    float *calciumVals = (float *) systemInfo["calciumVals"].cast<py::array_t<float>>().request().ptr;
    float *spikeVals = (float *) systemInfo["spikeVals"].cast<py::array_t<float>>().request().ptr;

    //std::vector<float> netCurrents(nNeurons, 0.0);
    float gatingTotal = 1.0f;
    for (int i=0;i<nChannels;i++){
        for (int j=0;j++;j++){
            gatingTotal *= x[i*chanStride + j*gateStride];
            py::print("Chan ",i,"gate",j,"val", gatingTotal);
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