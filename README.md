# ExaDEM


![](doc/logo/exaDEMlogo2.png)

ExaDEM is a software solution in the field of computational simulations. It's a Discrete Element Method (DEM) code developed within the exaNBody framework. This framework provides the basis for DEM functionalities and performance optimizations. A notable aspect of ExaDEM is its hybrid parallelization approach, which combines the use of MPI (Message Passing Interface) and Threads (OpenMP). This combination aims to enhance computation times for simulations, making them more efficient and manageable.

Additionally, ExaDEM offers compatibility with MPI+GPUs, using the CUDA programming model (Onika layer). This feature provides the option to leverage GPU processing power for potential performance gains in simulations. Written in C++17, ExaDEM is built on a contemporary codebase. It aims to provide researchers and engineers with a tool for adressing DEM simulations.

## Documentation

Documentation is available here: [ExaDEM Documentation](https://github.com/Collab4exaNBody/doc_exaDEM.git)

Build the documentation:

```
git clone https://github.com/Collab4exaNBody/doc_exaDEM.git
cd doc_exaDEM/
git submodule init
git submodule update
make html
firefox build/html/index.html 
```

## Installation

### Minimal Requirements

To proceed with the installation, your system must meet the minimum prerequisites. The first step involves the installation of exaNBody:

```
git clone https://github.com/Collab4exaNBody/exaNBody.git
mkdir build-exaNBody/ && cd build-exaNBody/
cmake ../exaNBody/ -DCMAKE_INSTALL_PREFIX=path_to_install
make install
export exaNBody_DIR=path_to_install
```

The next step involves the installation of yaml-cpp, which can be achieved using either the spack package manager or cmake:

```
spack install yaml-cpp@0.6.3
spack load yaml-cpp
```

Variant: 

```
apt install libyaml-cpp-dev
```

### Optional Dependencies

Before proceeding further, you have the option to consider the following dependencies:

- Cuda
- MPI

### Buidling ExaDEM With CMAKE

To install `ExaDEM`, follow these steps:

Set the `exaNBody_DIR` environment variable to the installation path. Clone the ExaDEM repository using the command:

```
git clone https://github.com/Collab4exaNBody/exaDEM.git
```


Create a directory named build-exaDEM and navigate into it:

```	
mkdir build-exaDEM && cd build-exaDEM
```

Run CMake to configure the ExaDEM build, specifying that CUDA support should be turned off:

```		
cmake ../exaDEM -DXSTAMP_BUILD_CUDA=OFF
```

Build ExaDEM using the make command with a specified number of parallel jobs (e.g., -j 4 for 4 parallel jobs):

```	
make -j 4
```

Build Plugins

```	
make UpdatePluginDataBase
```

This command will display all plugins and related operators. Example: 

```	
 + exadem_force_fieldPlugin
   operator    cylinder_wall
   operator    gravity_force
   operator    hooke_force
   operator    rigid_surface
 + exadem_ioPlugin
   operator    print_simulation_state
   operator    read_xyz
   operator    read_dump_particles
```

### Building ExaDEM With SPACK

Installation with spack is preferable for people who don't want to develop in `ExaDEM`. Only stable versions are added when you install `ExaDEM` with `Spack`. Note: `ExaDEM` main will never be directly accessible via this installation method.

#### Installing Spack

```
git clone https://github.com/spack/spack.git
export SPACK_ROOT=$PWD/spack
source ${SPACK_ROOT}/share/spack/setup-env.sh
```

#### Installing ExaDEM

First get the spack repository in `ExaDEM` directory and it to spack. It contains two packages: `exanbody` and `exadem`.

```
git clone https://github.com/Collab4exaNBody/exaDEM.git
cd exaDEM
spack repo add spack_repo
```

Current variante(s):
  
- +cuda: Add GPU support (disabled by default)

Second install `ExaDEM` (this command will install cmake, yaml-cpp and exanbody).

```
spack install exadem
```

Finally the `ExaDEM` executable has been created in the spack directory. You can run your simulation with your input file (`your_input_file.msp`) such as:

```
spack load exadem
exaDEM your_input_file.msp
```

### Running your simulation

Now that you have installed the exaDEM and exaNBody packages, you can create your simulation file in YAML format (refer to the 'example' folder or the documentation for each operator). Once this file is constructed, you can initiate your simulation using the following instructions.

```
export N_OMP=1
export N_MPI=1
export OMP_NUM_THREADS=$N_OMP
mpirun -n $N_MPI ./exaDEM test-case.msp
```

Note: for your first example, copy the `exaDEMexample/rotating-drum` directory and run `path-to-build/exaDEM rotating-drum.msp`.

## Authors and acknowledgment
Show your appreciation to those who have contributed to the project.

## License
For open source projects, say how it is licensed.
