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
