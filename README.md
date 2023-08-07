# exaDEM

- Exadem is "Discrete Element Method" DEM code developed using the exaNBody framwork. 
- ExaDEM uses hybrid MPI+Threads(OpenMP) parallelization to accelerate computation times. 
- ExaDEM can also be used on MPI+GPUs (cuda). 
- ExaDEM is written in C++17.


## Add your files

```
cd existing_repo
git remote add origin https://www-git-cad.intra.cea.fr/DEC/collaboratif/rp269144/exadem.git
git branch -M main
git push -uf origin main
```

## Installation

First, you need to isntall exaNBody : 

```
git clone https://github.com/Collab4exaNBody/exaNBody.git
mkdir build-exaNBody/ && cd build-exaNBody/
cmake ../exaNBody/ -DCMAKE_INSTALL_PREFIX=path_to_install
make install
export exaNBody_DIR=path_to_install
```

Install ExaDEM

```
export exaNBody_DIR=path_to_install
git clone https://www-git-cad.intra.cea.fr/DEC/collaboratif/rp269144/exadem.git
mkdir build-exaDEM && cd build-exaDEM
cmake ../exaDEM
make -j 4
```

## Test cases

The following test-cases are available in the directory : example

| Test case | Start | End |
|-----------|-------|-----|
| rotating-drum | ![](doc/example/rotating_drum_start.png) | ![](doc/example/rotating_drum_end.png) |
| axial-stress | ![](doc/example/axial_stress_start.png) | ![](doc/example/axial_stress_end.png) |
| radial-stress | ![](doc/example/radial_stress_start.png) | ![](doc/example/radial_stress_end.png) |
| rigid-surface | ![](doc/example/rigid_surface_start.png) | ![](doc/example/rigid_surface_end.png) |
| impose-velocity | ![](doc/example/impose_velocity_start.png) | ![](doc/example/impose_velocity_end.png) |
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
