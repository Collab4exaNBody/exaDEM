[![CMake](https://github.com/Collab4exaNBody/exaDEM/actions/workflows/cmake.yml/badge.svg)](https://github.com/Collab4exaNBody/exaDEM/actions/workflows/cmake.yml)
[![Spack](https://github.com/Collab4exaNBody/exaDEM/actions/workflows/spack.yml/badge.svg)](https://github.com/Collab4exaNBody/exaDEM/actions/workflows/spack.yml)
[![DOI](https://joss.theoj.org/papers/10.21105/joss.07484/status.svg)](https://doi.org/10.21105/joss.07484)

# ExaDEM

![](doc/logo/exaDEMlogo2.png)

ExaDEM is a software solution in the field of computational simulations. It's a Discrete Element Method (DEM) code developed within the exaNBody framework. This framework provides the basis for DEM functionalities and performance optimizations. A notable aspect of ExaDEM is its hybrid parallelization approach, which combines the use of MPI (Message Passing Interface) and Threads (OpenMP). This combination aims to enhance computation times for simulations, making them more efficient and manageable.

Additionally, ExaDEM offers compatibility with MPI+GPUs, using the CUDA programming model (Onika layer). This feature provides the option to leverage GPU processing power for potential performance gains in simulations. Written in C++17, ExaDEM is built on a contemporary codebase. It aims to provide researchers and engineers with a tool for adressing DEM simulations.

## Documentation


<img src="https://github.com/user-attachments/assets/a1891669-f579-4dd3-a109-82f2fe3cd587" width="200">

Documentation is available here: 

- Website: [ExaDEM Website](https://collab4exanbody.github.io/doc_exaDEM/)
- Github: [ExaDEM Documentation](https://github.com/Collab4exaNBody/doc_exaDEM.git)

- Main Sections:
  - [Overview](https://collab4exanbody.github.io/doc_exaDEM/project_exaDEM/Overview.html#overview-of-exadem)
  - [Installation](https://collab4exanbody.github.io/doc_exaDEM/project_exaDEM/Installation.html)
  - [ExaNBody](https://collab4exanbody.github.io/doc_exaDEM/project_exaNBody/index.html)
  - [Spheres](https://collab4exanbody.github.io/doc_exaDEM/project_exaDEM/user_guide/spheres.html) / [Polyhedra](https://collab4exanbody.github.io/doc_exaDEM/project_exaDEM/user_guide/polyhedra.html)
  - [Force field](https://collab4exanbody.github.io/doc_exaDEM/project_exaDEM/user_guide/force_field.html)
  - [I/O](https://collab4exanbody.github.io/doc_exaDEM/project_exaDEM/user_guide/IO.html)
  - [Examples](https://collab4exanbody.github.io/doc_exaDEM/project_exaDEM/Test_cases.html)
  - [Tutorials](https://collab4exanbody.github.io/doc_exaDEM/project_exaDEM/Tutorial.html)

## Community Guidelines

For more details, see `CONTRIBUTING.md`. Main guidelines are:

- For any bug, please create an issue and add the label “bug”. We welcome all feedback to make exaDEM as robust as possible.
- If you would like to participate and add functionality to `exaDEM`, you can find instructions for coding style, tests and pull request process in `CONTRIBUTING.md`.
- If you have any support-related / collaboration questions, please contact the team at `raphael.prat@cea.fr`. If you are a `CEA` member, please request access to the group : "exaNBody & Co. (exaStamp, exaDEM, exaSPH)", an external access can also be provided. 

## Authors and acknowledgment

### Main developers

- Raphaël Prat (CEA/DES) (raphael.prat@cea.fr)
- Thierry Carrard (CEA/DAM)
- Carlo Elia Doncecchi (CEA/DES)

### Other Developers

- Paul Lafourcade (CEA/DAM)
- Lhassan Amarsid (CEA/DES)
- Vincent Richefeu (CNRS)

### Acknowledgment

`ExaDEM` is part of the `PLEIADES` platform which has been developped in collaboration with the French nuclear industry - mainly `CEA`, `EDF`, and `Framatome` - for simulation of fuel1 elements.

## License

See `LICENSE.txt`
