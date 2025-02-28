# ExaDEM

![](doc/logo/exaDEMlogo2.png)

ExaDEM is a software solution in the field of computational simulations. It's a Discrete Element Method (DEM) code developed within the exaNBody framework. This framework provides the basis for DEM functionalities and performance optimizations. A notable aspect of ExaDEM is its hybrid parallelization approach, which combines the use of MPI (Message Passing Interface) and Threads (OpenMP). This combination aims to enhance computation times for simulations, making them more efficient and manageable.

Additionally, ExaDEM offers compatibility with MPI+GPUs, using the CUDA programming model (Onika layer). This feature provides the option to leverage GPU processing power for potential performance gains in simulations. Written in C++17, ExaDEM is built on a contemporary codebase. It aims to provide researchers and engineers with a tool for adressing DEM simulations.

## Documentation

Documentation is available here: 

- Website: [ExaDEM Website](https://collab4exanbody.github.io/exaDEM/)
- Github: [ExaDEM Documentation](https://github.com/Collab4exaNBody/doc_exaDEM.git)

## Authors and acknowledgment

`ExaDEM` is part of the `PLEIADES` platform which has been developped in collaboration with the French nuclear industry - mainly `CEA`, `EDF`, and `Framatome` - for simulation of fuel1 elements.

## License

See `LICENSE.txt` 
