Some notes on using pybind11, cmake, and vscode.  
VScode intellisense doesnt seem to automatically know where to look for pyBind11 headers, so you need to direct it. It should be within one of your anaconda or miniconda environments e.g:  
C:\Users\CathalMcLoughlin\anaconda3\envs\morpho311\include
C:\Users\CathalMcLoughlin\anaconda3\envs\morpho311\Lib\site-packages\pybind11\include  
You need both of these. 


I use the MSVC compiler here so the compiler argument is cl.exe
