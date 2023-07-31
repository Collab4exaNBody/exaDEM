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
git clone pathto/exaDEM.git
mkdir build-exaDEM && cd build exaDEM
cmake ../exaDEM
make -j 4
```

## Test cases

The following test-cases are available in the directory : example

| Test case | Start | End |
|-----------|-------|-----|
| rotating-drum | ![](doc/examples/rotating_drum_start.png) | ![](doc/examples/rotating_drum_end.png) |
| axial-stress | ![](doc/examples/axial_stress_start.png) | ![](doc/examples/axial_stress_end.png) |
| radial-stress | ![](idoc/examples/radial_stress_start.png) | ![](doc/examples/radial_stress_end.png) |
| rigid-surface | ![](doc/examples/rigid_surface_start.png) | ![](doc/examples/rigid_surface_end.png) |
| impose-velocity | ![](doc/examples/impose_velocity_start.png) | ![](doc/examples/impose_velocity_end.png) |
| movable-wall | ![](doc/examples/movable_wall_start.png) | ![](doc/examples/movable_wall_end.png) |

## List of DEM operators

### Drivers

#### Rigid Surface

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

## Authors and acknowledgment
Show your appreciation to those who have contributed to the project.

## License
For open source projects, say how it is licensed.
