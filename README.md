# ExaDEM

ExaDEM is a software solution in the field of computational simulations. It's a Discrete Element Method (DEM) code developed within the exaNBody framework. This framework provides the basis for DEM functionalities and performance optimizations. A notable aspect of ExaDEM is its hybrid parallelization approach, which combines the use of MPI (Message Passing Interface) and Threads (OpenMP). This combination aims to enhance computation times for simulations, making them more efficient and manageable.

Additionally, ExaDEM offers compatibility with MPI+GPUs, using the CUDA programming model (Onika layer). This feature provides the option to leverage GPU processing power for potential performance gains in simulations. Written in C++17, ExaDEM is built on a contemporary codebase. It aims to provide researchers and engineers with a tool for adressing DEM simulations.


## Add your files

```
cd existing_repo
git remote add origin https://www-git-cad.intra.cea.fr/DEC/collaboratif/rp269144/exadem.git
git branch -M main
git push -uf origin main
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
spack install yaml-cpp
spack load yaml-cpp
```

### Optional Dependencies

Before proceeding further, you have the option to consider the following dependencies:

- Cuda
- MPI

### ExaDEM Installation

To install ExaDEM, follow these steps:

Set the exaNBody_DIR environment variable to the installation path. Clone the ExaDEM repository using the command:

```
git clone https://www-git-cad.intra.cea.fr/DEC/collaboratif/rp269144/exadem.git
```

Create a directory named build-exaDEM and navigate into it:

```
mkdir build-exaDEM && cd build-exaDEM
```

Run CMake to configure the ExaDEM build, specifying that CUDA support should be turned off:

```
cmake ../exadem -DXSTAMP_BUILD_CUDA=OFF
```

Build ExaDEM using the make command with a specified number of parallel jobs (e.g., -j 4 for 4 parallel jobs):

```
make -j 4
```

## Test cases

You can explore various basic test cases located in the `example` directory. These test cases serve as illustrative examples of ExaDEM's functionality and can assist you in understanding its behavior and capabilities.

### Example 1: Rotating drum

A DEM simulation of a rotating drum involves modeling the movement of spherical particles within a drum container as it rotates. Through this simulation, we can observe how particles interact, collide, and move in response to the drum's motion. This provides insights into phenomena like particle segregation, convection currents, and mixing patterns, contributing to improved understanding of granular material behavior in rotational scenarios.

| Test case | Start | End |
|-----------|-------|-----|
| rotating-drum | ![](doc/example/rotating_drum_start.png) | ![](doc/example/rotating_drum_end.png) |


### Example 2: Axial stress

A DEM simulation under axial stress involves subjecting a collection of particles to pressure from a rigid surface along a specific axis. By simulating this scenario using the Discrete Element Method (DEM), we can study how particles respond to the applied stress. The simulation reveals how particles interact, deform, and reposition under the influence of the external force, providing insights into the behavior of granular materials under axial loading conditions. 

| Test case | Start | End |
|-----------|-------|-----|
| axial-stress | ![](doc/example/axial_stress_start.png) | ![](doc/example/axial_stress_end.png) |


### Example 3: Rigid stress

In a DEM simulation under radial stress, particles are exposed to pressure from a central point, causing an outward force in all directions. Using the Discrete Element Method (DEM) to simulate this scenario allows us to analyze how particles within a system react to the applied radial stress. The simulation offers insights into particle rearrangements, contact forces, and structural changes, giving us a deeper understanding of granular material behavior under radial loading conditions.

| Test case | Start | End |
|-----------|-------|-----|
| radial-stress | ![](doc/example/radial_stress_start.png) | ![](doc/example/radial_stress_end.png) |

### Example 4: Rigid surface 

A DEM simulation involving spherical particles falling onto a rigid surface offers a virtual exploration of particle dynamics in a gravity-driven scenario. This simulation captures the behavior of individual spherical particles as they descend and interact with a solid surface below.

| Test case | Start | End |
|-----------|-------|-----|
| rigid-surface | ![](doc/example/rigid_surface_start.png) | ![](doc/example/rigid_surface_end.png) |


### Example 5: 

In this DEM simulation, a scenario is simulated where a group of particles with imposed velocity occupies a defined area. As other particles fall into this region, they interact with the moving particles, impacting their trajectories. The simulation provides insights into how moving particles influence the behavior of surrounding particles.

| Test case | Start | End |
|-----------|-------|-----|
| impose-velocity | ![](doc/example/impose_velocity_start.png) | ![](doc/example/impose_velocity_end.png) |


### Example 6 : Movable wall

In this DEM simulation, a cluster of spherical particles is constrained against a rigid surface. A piston is introduced to apply a steadily increasing stress that linearly evolves over time. This simulation captures the dynamics as the piston's force gradually grows. As the piston imparts its stress, the particle block undergoes deformation and stress propagation. 

| Test case | Start | End |
|-----------|-------|-----|
| movable-wall | ![](doc/example/movable_wall_start.png) | ![](doc/example/movable_wall_end.png) |

## List of DEM operators

### Global operators

Some operators from default `exaNBody` operator list to  

- Operator `domain` : see @exaNBody in plugin @exanbIOPlugin
  - `cell_size` : The cell size will be approximately the value given by the possible subdivision of the grid covering the simulation volume. Cell size must be greater than twice the radius of the largest sphere. Note that cell size can greatly influence performance. 
  - `periodic` : Define whether or not the boundary conditions of the system under study are periodic along the (Ox), (Oy) and (Oz) axes.

YAML example:

```
domain:
  cell_size: 2 m
  periodic: [false,true,false]
```

- Operator `global` :  see @exaNBody in plugin @exanbCorePlugin
  - `simulation_dump_frequency` : Writes an mpiio file each time the number of iterations modulo this number is true.
  - `simulation_paraview_frequency` : Writes an paraview file each time the number of iterations modulo this number is true.
  - `simulation_log_frequency` : Prints logs each time the number of iterations modulo this number is true.
  - `simulation_end_iteration` : Total number of iterations
  - `dt` : Is the integration time value
  - `rcut_inc` : Corresponds to the Verlet radius used by the Verlet list method. Note that the smaller the radius, the more often verlet lists are reconstructed. On the other hand, the larger the radius, the more expensive it is to build the verlet lists.
  - `friction_rcut` : This radius is used to construct lists containing the value of friction between sphere pairs. This radius must be `>= rcut + rcut_inc`.

YAML example:
```
global:
  simulation_dump_frequency: -1
  simulation_end_iteration: 100000
  simulation_log_frequency: 1000
  simulation_paraview_frequency: 5000
  dt: 0.00005 s
  rcut_inc: 0.01 m
  friction_rcut: 1.1 m
```

### Force law

| Operator name | hooke_force |
|--|--|
| Description | This operator computes forces between spheric particles using the Hooke law. |
| config | flow : IN <br> type : exaDEM::HookeParams <br> desc : Data structure that contains hooke force parameters (rcut, dncut, kn, kt, kr, fc, mu, damp_rate) |

Comment: hooke force includes a cohesion force from `rcut` to `rcut+dncut` with the cohesion force parameter `fc`.

YAML example:

```
- hooke_force:
  config: { rcut: 1.1 m , dncut: 1.1 m, kn: 100000, kt: 100000, kr: 0.1, fc: 0.05, mu: 0.9, damp_rate: 0.9}
```

| Operator name | gravity_force |
|--|--|
| Description | This operator computes forces related to the gravity. |
| gravity | flow : IN <br> type : Vec3d <br> desc : Define the gravity constant in function of the gravity axis, default value are x axis = 0, y axis = 0 and z axis = -9.807 |

YAML example:

```
- gravity_force:
  - gravity: [0,0,-0.009807]
```


### Drivers

| Operator name  | rigid_surface |
|--|--|
| Description | This operator computes forces between particles and a rigid surface (named wall in other operators) using the Hooke law. |    			
| damprate | flow : IN <br> type double <br> desc : Parameter of the force law used to model contact rigid surface/sphere |
| kn | flow : IN <br>  type : double <br> desc : Parameter of the force law used to model contact rigid surface/sphere |
| kr | flow : IN <br> type : double <br> desc : Parameter of the force law used to model contact rigid surface/sphere |
| kt | flow : IN <br> type : double <br> desc : Parameter of the force law used to model contact rigid surface/sphere |
| mu | flow : IN <br> type double <br> desc : Parameter of the force law used to model contact rigid surface/sphere |
| normal | flow : IN <br> type : Vec3d <br> desc : Normal vector of the rigid surface |
| offset | flow : IN <br> type : double <br> desc : Offset from the origin (0,0,0) of the rigid surface |

Yaml example, see `example/rigid_surface.msp`:

```
- rigid_surface:
   normal: [0,0,1]
   offset: -1
   kt: 80000
   kn: 100000
   kr : 0
   mu: 0.9
   damprate: 0.9
```

### Reader/Writer

| Operator | read_xyz |
|--|--|
| Description | This operator reads a file written according to the xyz format |
| bounds_mode | flow : IN <br> type : ReadBoundsSelectionMode |
| enlarge_bounds | flow : IN <br> type : double <br> desc : Define a layer around the volume size in the xyz file | 
| file | flow : IN <br> type : string <br> desc : Filename | 
| pbc_adjust_xform | flow : IN <br> type : bool <br> desc : Ajust the form |

YAML example: 

```
- read_xyz:
  file: input_file_rigid_surface.xyz
  bounds_mode: FILE
  enlarge_bounds: 1.0 m
```


## Authors and acknowledgment
Show your appreciation to those who have contributed to the project.

## License
For open source projects, say how it is licensed.
