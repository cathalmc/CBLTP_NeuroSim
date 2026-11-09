# Overview
## Preamble
This is very much an unfinished project, but I need something to demonstrate my programming skills. 
At the time of writing (November 2026) an agent could finish this project and implement all my features perfectly in about an hour. However that finished project would not be very informative to someone trying to assess my skills, which is what this is really about. 
I used chatGPT, Gemini, and Claude through their chat interface at various points in this project, but I have not had an agent work on it. 

The aim of this project was to build a generic version of the STN-GPe circuit model I developed in my [activity dependent plasticity](https://www.science.org/doi/full/10.1126/sciadv.adw7421) paper. I was happy with the broad strokes of that model (calcium-based synaptic and structural plasticity, solved in CUDA using a GPU), but over time it became a horrible mess of basal ganglia-specific features and idiosyncrasies relating to the study. I wanted to use the lessons I learned on that project to develop a generic neuron simulator that I could use for my own little experiments. In truth I got more enjoyment from developing features than from doing the experiments, which may partially explain why I never finished it. The key idea that the simulator is based around, is that of a generic 2 state Hodgkin-Huxley type channel that can represent voltage dependent ion channels, calcium sensitive channels, and synapses, all in the one framework. By combining these features in a single framework, we can shift a large amount of the computational work onto a processes that can easily be parallelized, and ultimately run on a GPU. I never got around to writing the CUDA kernel, and recently (because I now have a mac) I have thought to instead write it as a compute shader using the cross-platform WebGPU library. This is a trivial task for an agent, and I will do if I decide that this is worth a full release. With that channel framework acting as the engine, the rest of the software was built around exploiting it. This ended up shifting most of the programming work onto the model description and model compilation parts of the code, and at the same time changed my philosophy towards developing simulators. My approach is now to  build a powerful solver that can make the most use of available compute, build an API that is convenient for describing the models of interest, and then put most of my efforts into the middle stage of compiling from model description to solver data structures.


## How to use
[tuning_neurons2026.ipynb] shows how to create custom channels and neurons.
[tuning_plasticity.ipynb] shows how synapses and plasticity works
[2026_Feb21_fullSTNGPe_plasticityOptimization] shows how to create the full STN-GPe network model. 

There are 4 backends, one is plain python, one uses numpy, one is single-threaded c++ and one is multi-threaded c++.  
I recommend not using plain python, use numpy only if you can't get the c++ to compile, use single threaded c++ for single neurons and small networks, and multithreaded for large networks.
Use the python notebook "compileCrossPlatform.ipynb" to compile the C++ code.



## Old Notes  
Some notes on using pybind11, cmake, and vscode.  
VScode intellisense doesnt seem to automatically know where to look for pyBind11 headers, so you need to direct it. It should be within one of your anaconda or miniconda environments e.g:  
C:\Users\<yourName>\anaconda3\envs\NeuroEnv\include
C:\Users\<yourName>\anaconda3\envs\NeuroEnv\Lib\site-packages\pybind11\include  
You need both of these. 

I use the MSVC compiler here so the compiler argument is cl.exe  

How it works (pyBind):  
pybind11 lets you do in-place modification of numpy files.  
so if you want to run some large simulation in C++, you use numpy to initialize your data and parameters, then you call the c++ function you wrote on these numpy arrays.  
the arrays get modified in place, so when the function returns, your data is updated.  
The best development approach to this is to first write all your code in python, do all the validation and error checking, and then translate it piece by piece to c++.  
You do NOT want to be error checking on the c++ side.  
Validate the c++ code against your python code.  Their results be identical (should ...)  


Tips:  
If you get a linker (LINK) error when compiling, it's often because the module is open in your python script. Restart the kernel and try again.

## Installing on Mac. 
Make sure homebrew and conda are installed. 
Install cmake, llvm, and libomp using homebrew: brew install cmake llvm libomp. 
Apple's version of llvm doesnt support openmp so you have to download the stock llvm.
In .zshrc, add the the following:
export PATH="/opt/homebrew/opt/llvm/bin:$PATH"
export LDFLAGS="-L/opt/homebrew/opt/llvm/lib"
export CPPFLAGS="-I/opt/homebrew/opt/llvm/include"





## Random notes on the simulator  
Synapses are implemented in a weird way, using a HH-type gate where the input is a spike. Spikes are detected in neurons by checking when the voltage is greater than the spike thereshold (equivalent to how L-type calcium channels work at the pre-synaptic part of the axon). The threshold voltage (vThreshold) is 0mV. I was thinking of setting it to be a user-adjustable parameter but I dont see the need. Spikes are written to a buffer so that we can have spike-delays. Spike buffers are unique to the neuron/ spiker which assumes each axon on each individual neuron has the same delay. Neurons can all individually have different spike delay lengths. When a spike reaches the end of the buffer it gets transmitted to all the post synaptic channels. A spike detecting gate in a synapse is modelled as using HH-dynapics as a sigmoid function that checks if the spike value is > 0, and if so will increase the spike gating variable towards 1 within a couple of time steps. Some problems can arise when the presynaptic voltage is above 0 for more than 1 iteration. This sends multiple spikes in a row, which when recieved by the post synaptic gate can give slightly different output values. Or in other words, post synaptic current can depend on the shape of the presynaptic action potential, which is kind of realistic. E.g. strong pre-AP (more Vm>0) gives stronger output. I'm not sure if this can be considered an issue but its worth being aware of. If I add vThreshold as a parameter it might allow the user to have some ability to tune this behaviour. 
