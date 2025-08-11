Some notes on using pybind11, cmake, and vscode.  
VScode intellisense doesnt seem to automatically know where to look for pyBind11 headers, so you need to direct it. It should be within one of your anaconda or miniconda environments e.g:  
C:\Users\CathalMcLoughlin\anaconda3\envs\morpho311\include
C:\Users\CathalMcLoughlin\anaconda3\envs\morpho311\Lib\site-packages\pybind11\include  
You need both of these. 

I use the MSVC compiler here so the compiler argument is cl.exe  

How it works:  
pybind11 lets you do in-place modification of numpy files.  
so if you want to run some large simulation in C++, you use numpy to initialize your data and parameters, then you call the c++ function you wrote on these numpy arrays.  
the arrays get modified in place, so when the function returns, your data is updated.  
The best development approach to this is to first write all your code in python, do all the validation and error checking, and then translate it piece by piece to c++.  
You do NOT want to be error checking on the c++ side.  
Validate the c++ code against your python code.  Their results be identical (should ...)  


Tips:  
If you get a linker (LINK) error when compiling, it's often because the module is open in your python script. Restart the kernel and try again.

## Random notes on the simulator  
Synapses are implemented in a weird way, using a HH-type gate where the input is a spike. Spikes are detected in neurons by checking when the voltage is greater than the spike thereshold (equivalent to how L-type calcium channels work at the pre-synaptic part of the axon). The threshold voltage (vThreshold) is 0mV. I was thinking of setting it to be a user-adjustable parameter but I dont see the need. Spikes are written to a buffer so that we can have spike-delays. Spike buffers are unique to the neuron/ spiker which assumes each axon on each individual neuron has the same delay. Neurons can all individually have different spike delay lengths. When a spike reaches the end of the buffer it gets transmitted to all the post synaptic channels. A spike detecting gate in a synapse is modelled as using HH-dynapics as a sigmoid function that checks if the spike value is > 0, and if so will increase the spike gating variable towards 1 within a couple of time steps. Some problems can arise when the presynaptic voltage is above 0 for more than 1 iteration. This sends multiple spikes in a row, which when recieved by the post synaptic gate can give slightly different output values. Or in other words, post synaptic current can depend on the shape of the presynaptic action potential, which is kind of realistic. E.g. strong pre-AP (more Vm>0) gives stronger output. I'm not sure if this can be considered an issue but its worth being aware of. If I add vThreshold as a parameter it might allow the user to have some ability to tune this behaviour. 
