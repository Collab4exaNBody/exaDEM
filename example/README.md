# Examples

Here you'll find a very short list of examples, with more information in the documentation [here](https://collab4exanbody.github.io/doc_exaDEM/project_exaDEM/Test_cases.html)

## Spheres

### Rotating drum

One test case with periodic conditions (rotating-drum/rotating-drum-full.msp) and another with surfaces instead of periodic conditions (rotating-drum/rotating-drum-no-periodic.msp). This examples use a perfect cylinder to model the drum. 


### Cylinder (STL) && Complex Meshes

- It corresponds to the show-case in the documentation. Instead of using a perfect cylinder, we'll read an stl file with a cylinder containing small barriers. See (cylinder_stl/cylinder_stl.msp)
- This example shows how to add an stl mesh to your simulation. See: mesh-stl/mesh_stl_full.msp
- Last example is not really physic but try to repoduce a geyce, input files are availables here: exaDEM/example/spheres/jet/

### Rigid surface / Wall

- An example of falling spheres on a rigid surface / wall is available here: rigid_surface/rigid_surface_full.msp
- Another example shows how to move rigid surfaces: movable-wall/movable_wall.msp

### Use region operator to impose specific behaviors 

Two test cases, the first one impose the same veloctity to all particle to a given region (impose_velocity/impose_velocity_full.msp) and the second add a hole into the lattice generation (impose_velocity/impose_velocity_hole.msp

## Polyhedra


