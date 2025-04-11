#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <chrono>

namespace py = pybind11;
using namespace pybind11::literals;

py::object testFunc(py::array_t<int> testIntVal,
                    py::array_t<double> testNParr
                    );

py::object solveNeuralSystem(py::dict systemInfo,py::dict recordingInfo,  float timeStep,int nIterations,int samplingInterval);
