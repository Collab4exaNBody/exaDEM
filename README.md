# ExaDEM


![](doc/logo/exaDEMlogo2.png)

ExaDEM is a software solution in the field of computational simulations. It's a Discrete Element Method (DEM) code developed within the exaNBody framework. This framework provides the basis for DEM functionalities and performance optimizations. A notable aspect of ExaDEM is its hybrid parallelization approach, which combines the use of MPI (Message Passing Interface) and Threads (OpenMP). This combination aims to enhance computation times for simulations, making them more efficient and manageable.

Additionally, ExaDEM offers compatibility with MPI+GPUs, using the CUDA programming model (Onika layer). This feature provides the option to leverage GPU processing power for potential performance gains in simulations. Written in C++17, ExaDEM is built on a contemporary codebase. It aims to provide researchers and engineers with a tool for adressing DEM simulations.

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

### ExaDEM Installation

To install ExaDEM, follow these steps:

Set the exaNBody_DIR environment variable to the installation path. Clone the ExaDEM repository using the command:

```
git clone https://github.com/Collab4exaNBody/exaDEM.git
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

### Running your simulation

Now that you have installed the exaDEM and exaNBody packages, you can create your simulation file in YAML format (refer to the 'example' folder or the documentation for each operator). Once this file is constructed, you can initiate your simulation using the following instructions.

```
export N_OMP=1
export N_MPI=1
export OMP_NUM_THREADS=$N_OMP
mpirun -n $N_MPI ./exaDEM test-case.msp
```

Note: for your first example, copy the `exaDEMexample/rotating-drum` directory and run `path-to-build/exaDEM rotating-drum.msp`.

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

### Example 6 : Using a STL Mesh

In this DEM simulation, a cluster of spherical particles is  a cluster of particles falls onto an stl mesh and into a box. This case study highlights the use of meshes containing numerous facets.

| Test case | Start | End |
|-----------|-------|-----|
| mesg-stl | ![](doc/example/mesh_stl_start.png) | ![](doc/example/mesh_stl_end.png) |

## Tutorial: How to add a new operator

### Add a new plugin

You can create new plugins if you wish to develop a new set of operators that can be compiled independently of others or require specific plugin compilation. Here's how to define and add a new plugin.

First of all, add your new plugin directory in `src/CMakeLists.txt`:

```
mkdir my_new_plugin
add_subdirectory(my_new_plugin)
cd my_new_plugin
vi CMakeLists.txt
```

Create a CMakeLists.txt and use the macros defined to plug your operators to exaNBody support such as:

```
set(list_of_plugins_required exanbIO exanbDefBox exanbParticleNeighbors exadem_numerical_scheme exadem_friction exadem_force_field)
set(exadem_my_new_plugin_LINK_LIBRARIES ${list_of_plugins_required})
xstamp_add_plugin(exadem_my_new_plugin ${CMAKE_CURRENT_SOURCE_DIR})
```

Warning: Do not forget to do a `make UpdatePluginDataBase`.

### Add a new operator

Initially, it's crucial to establish a precise definition of the intended kernel, the targeted data, and the method for executing this kernel.

#### Define your kernel

We recommend writing the computation kernel (in a functor) in a `.h` file within an `include/exaDEM` folder of your plugin directory. For instance, if one intends to apply gravitational force to a particle, it's necessary to have knowledge of the external force and access to mutators for modifying the force fields. The kernel for a particle is then written in a file `include/exaDEM/gravity_force.h`: 

```
namespace exaDEM
{
  struct GravityForceFunctor
  {
    exanb::Vec3d g = { 0.0 , 0.0 , -9.807};
    ONIKA_HOST_DEVICE_FUNC inline void operator () (double mass, double& fx, double& fy, double& fz ) const
    {
      fx += g.x * mass;
      fy += g.y * mass;
      fz += g.z * mass;
    }
  };
}
```

It is possible to provide specific kernel `Traits` that will be used by the data traversal functions (ex: `ComputeCellParticles`). For example, if we want to provide the GPU capabilities for this kernel:

```
namespace exanb
{
  template<> struct ComputeCellParticlesTraits<exaDEM::GravityForceFunctor>
  {
    static inline constexpr bool CudaCompatible = true;
  };
}
```

#### Define an operator:

To begin with, all operators inherit from the OperatorNode class, where you establish the necessary physical fields for applying your operator. Following that, it's essential to define the slots that will establish the execution graph among operators. Ultimately, you are required to furnish an Execute() function, which the graph will execute, along with a Documentation() function.

This is an example of an empty operator:

```
#include <exaDEM/your_kernel.h>
namespace exaDEM
{
  using namespace exanb;
  template<typename GridT
    , class = AssertGridHasFields< GridT, list_of_fields>
    >
  class YourOperator : public OperatorNode
  {
    using ComputeFields = FieldSet< list_of_fields>;
    static constexpr ComputeFields compute_field_set {};
    ADD_SLOT( GridT  , grid     , INPUT_OUTPUT );
  public:

    inline std::string documentation() const override final
    {
      return R"EOF(Your operator.)EOF";
    }

    inline void execute () override final
    {
      YourFunctor func;
      compute_cell_particles( *grid , false , func , compute_field_set , gpu_execution_context() , gpu_time_account_func() );
    }
  };

  template<class GridT> using YourOperatorTmpl = YourOperator<GridT>;

  // === register factories ===  
  CONSTRUCTOR_FUNCTION
  {
    OperatorNodeFactory::instance()->register_factory( "your_operator_name", make_grid_variant_operator< YourOperatorTmpl > );
  }
}
```

Comments:
- Please refers to `src/forcefields/gravity_force.cpp` to illustrate a simple example. 
- AssertGridHasFields allows to eliminate wrong grids
- If `ComputeFields` does not correspond to the input parameters of your functor, the operator can't compile.
- ADD_SLOT macro works as `ADD_SLOT(Type, Name, TypeOfSlot, DefaultValue, Documentation)` with:
  - Type: `double`, `int` ...
  - TypeOfSlot: `INPUT`, `OUTPUT` or `INPUT_OUTPUT`
  - Documentation: is a DocString type: `DocString{slot documentation}`
- To access to a slot value, please add `*`.
- Constant value are hidden the operator, for example the gravity functor is initialized as: ` GravityForceFunctor func { *gravity};`, gravity is a slot. 
- Note that you need to specify you operator for a given Grid and define the `operator_name` in your dictionnary. 

Comment: You can do your own traversal function by explicitely iterating over cells. Example:

```
ADD_SLOT( GridT  , grid            , INPUT_OUTPUT );
inline void execute () override final
{
  MPI_Comm comm = *mpi;
  auto cells = grid->cells();
  IJK dims = grid->dimension();
  size_t ghost_layers = grid->ghost_layers();
  IJK dims_no_ghost = dims - (2*ghost_layers);
# pragma omp parallel
  {
    GRID_OMP_FOR_BEGIN(dims_no_ghost,_,loc_no_ghosts)
    {
      IJK loc = loc_no_ghosts + ghost_layers;
      size_t cell_i = grid_ijk_to_index(dims,loc);
      auto& cell_ptr = cells[cell_i];
      const size_t n = cells[cell_i].size();
      auto* __restrict__ _my_field = cell_ptr[field::my_field];
#     pragma omp simd
      for(size_t j=0;j<n;j++)
        _my_field[j] += 27;
    }
  }
}
```



## List of DEM operators

In this section, we describe the main ExaDEM operators and the operators used from ExaNBody. Each operator description is accompanied by a general description, a slot description and an example of use in yaml that you can include in your simulations.

### Global operators

Some operators from the default `exaNBody` operator list. Here's a list of the main operators that will be useful for your simulations.

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

### exadem_force_fieldPlugin

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


### exadem_driverPlugin

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

| Operator name  | rigid_surface |
|--|--|
| Description | External forces that model air or fluid, f = - mu * cx * v * vector(v). |
| cx | flow : IN <br>  type : double <br> desc : aerodynamic coefficient, air = 0.38 |
| mu | flow : IN <br> type : double <br> desc : drag coefficient. air = 0.000015 |


YAML example; see example `quadratic-force-test/QuadraticForceInput.msp`

```
- quadratic_force:
   cx: 0.38
   mu: 0.0000015
```

### exadem_frictionPlugin

TODO

### exadem_numerical_schemePlugin

TODO

### exadem_ioPlugin 

#### Reader/Writer

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


| Operator | read_dump_particles |
|--|--|
| Description | This operator readz a dump file with all particles information required to restart the simulation. See operator : @write_dump_particles |
| enable_friction | flow : IN <br> type : bool <br> desc : Enable to write friction information. By default this option is activated |
| filename | flow : IN <br> type : string <br> desc : dump file name | 

```
YAML example:
input_data:
  - read_dump_particles:
      filename: last.dump
```

| Operator | write_dump_particles |
|--|--|
| Description | This operator writes a dump file with all particles information required to restart the simulation. See operator : @read_dump_particles |
| compression_level | flow : IN <br> type : long <br> desc : Zlib compression level |
| filename | flow : IN <br> type : string <br> desc : dump file name | 

This operator is defined in the default `ExaDEM` operator named `dump_data_particles`. Note that the default name is defined by : `- timestep_file: "exaDEM_%09d.dump` and piloted by `simulation_dump_frequency: 1` in the operator `global`.

TODO : @Lhassan, il faut définir les outputs de cet opérateur.

| Operator | print_simulation_state |
|--|--|
| Description | This operator prints logs during the simulation.|

This operator is defined as part of default operators. To trigger this log, you need to specify : `simulation_log_frequency: 1000` in the operator `global`.


### exadem_field_mutatorPlugin

This repertory plugin only provides operators for modifying fields, especially at initialization. The following operators are based on the functor `set` and initialize one ore more fields: 
- set_densities_multiple_materials : 
  - [std::vector<double>] `densities`
  - Comment: mass is deduced from the density and radius
- set_density
  - [double] `density`
- set_homothety
  - [double] `homothety`
- set_material_properties
  - [uint8_t] `type`
  - [double] `rad`
  - [double] `density`
  - [Quaternion] `quat`
- set_quaternion
  - [Quaternion] `quat` 
- set_radius
  - [double] `rad`
- set_radius_multiple_materials
	- [std::vector<double>]` radius` (list of radius according to types)
- set_rand_vrot_arot:
  - [double] `var_vrot` (variance), default = 0
  - [double] `var_arot` (variance), default = 0
  - [Vec3d] `mean_arot`(mean), default = {0,0,0}
  - [Vec3d] `mean_vrot`(mean), default = {0,0,0}
  - Comment : This operator set the angular acceleration and velocity using a normal distribution
- set_rand_velocity
  - [double] `var` (variance), default = 0
  - [Vec3d] `mean`, default = {0,0,0}

YAML example:

```
  - set_radius:
     rad: 0.5
  - set_quaternion
  - set_rand_velocity:
     var: 0.1
     mean: [0.0,0.0,0.0]
  - set_density:
     density: 0.02
  - set_rand_vrot_arot
```

This is a minimal example to add your own mutator_field operator:
- [1] Set class name: `SetYourFields`
- [2] Set fields: `field::_YOUR_FIELD_1, field::_YOUR_FIELD_2, ..., field::_YOUR_FIELD_N`
- [3] Set types: `YOUR_TYPE_1, YOUR_TYPE_Z, ... , YOUR_TYPE_N`
- [4] Set field slots: `your_field_1, your_field_2, ..., your_field_N` 
- [5] Set operator name: `set_your_fields`
- [6] Specify template: `SetYourFields`

```
#include <exaDEM/set_fields.h>
namespace exaDEM
{
	using namespace exanb;
	template<typename GridT
		, class = AssertGridHasFields< GridT, field::_YOUR_FIELD_1, field::_YOUR_FIELD_2, ..., field::_YOUR_FIELD_N>
		>
		class SetYourFields : public OperatorNode
		{
			static constexpr YOUR_TYPE_1 default_field_value_1 = YOUR_TYPE_1();
    			static constexpr YOUR_TYPE_2 default_field_value_2 = YOUR_TYPE_2();
			...
			static constexpr YOUR_TYPE_N default_field_value_N = YOUR_TYPE_N();
			using ComputeFields = FieldSet< field::_YOUR_FIELD_1, field::_YOUR_FIELD_2, ..., field::_YOUR_FIELD_N>;
			static constexpr ComputeFields compute_field_set {};

			ADD_SLOT( GridT  		, grid  , INPUT_OUTPUT );
			ADD_SLOT( YOUR_TYPE_1  		, your_field_1  	, INPUT , default_radius	, DocString{"default radius value for all particles"} );
			ADD_SLOT( YOUR_TYPE_2  		, your_field_2  	, INPUT , default_radius	, DocString{"default radius value for all particles"} );
			...
			ADD_SLOT( YOUR_TYPE_N  		, your_field_N  	, INPUT , default_radius	, DocString{"default radius value for all particles"} );

			public:

			inline std::string documentation() const override final
			{
				return R"EOF(
      	  				This operator sets the ... value(s) for every particles.
     	   			)EOF";
			}

			inline void execute () override final
			{
				SetFunctor<YOUR_TYPE_1,YOUR_TYPE_2, ... , YOUR_TYPE_N> func = { {*your_field_1}, {*your_field_2},... , {*your_field_N} };
				compute_cell_particles( *grid , false , func , compute_field_set , gpu_execution_context() , gpu_time_account_func() );
			}
		};

	template<class GridT> using SetYourFieldsTmpl = SetYourFields<GridT>;

	// === register factories ===  
	CONSTRUCTOR_FUNCTION
	{
		OperatorNodeFactory::instance()->register_factory( "set_your_fields", make_grid_variant_operator< SetYourFieldsTmpl > );
	}
}
```

## Authors and acknowledgment
Show your appreciation to those who have contributed to the project.

## License
For open source projects, say how it is licensed.
