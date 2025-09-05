# Spheres Test Cases

This document provides a brief list of examples to showcase different test cases for sphere simulations. For more detailed information, please refer to the [documentation](https://collab4exanbody.github.io/doc_exaDEM/project_exaDEM/Test_cases.html).

## Rotating Drum

Three test cases demonstrate the use of a rotating drum:

1. A test case with periodic conditions: `rotating-drum/rotating-drum-periodic.msp`.
2. A test case with surfaces instead of periodic conditions: `rotating-drum/rotating-drum-no-periodic.msp`.
3. A test case with periodic condition and a cohesive laws: `rotating-drum/rotating-drum-periodic-cohesion.mspp`.

These examples use a perfect cylinder to model the drum.

## Ball

Three test cases to higligth different scenarion with different motions.

- Stationary: This test case represents a scenario where there is no movement, showcasing a static condition.
- Linear Motion: This case involves motion in a straight line, providing insights into linear stress behaviors.
- Radial Stress: This scenario examines the effects of radial stress.

## Cylinder (STL) & Complex Meshes

- **Cylinder with barriers:** Instead of a perfect cylinder, this example uses an STL file representing a cylinder with small barriers: `cylinder_stl/cylinder_stl.msp`.
- **Adding an STL mesh:** Demonstrates how to include an STL mesh in your simulation: `mesh-stl/mesh_stl_full.msp`.
- **Geyser simulation:** A non-physical example replicating a geyser; input files are available at: `exaDEM/example/spheres/jet/`.

## Rigid Surface / Wall

- **Falling spheres on a rigid surface/wall:** Example available at: `rigid_surface/rigid_surface_full.msp`.
- **Movable walls:** Demonstrates how to move rigid surfaces: `movable-wall/movable_wall.msp`.

## Use Region Operator to Impose Specific Behaviors

Two test cases showcase the use of the region operator:

1. Applying the same velocity to all particles within a specific region: `impose_velocity/impose_velocity_full.msp`.
2. Adding a hole to the lattice generation: `impose_velocity/impose_velocity_hole.msp`.

## Compression with a Piston

- **init_piston** This simulation sets up the particle sample.
- **piston_linear_force_motion** This simulation sets the piston in motion above the sample, applying a constant force.
- **piston_linear_compression_motion** This simulation sets the piston in motion above the sample, applying a pressure controlled by the variable sigma.

## Shaker

Two test cases demonstrate the use of the shaker motion:

- **shaker_surface:** Simulation of a sphere deposit on a vibrating surface.
- **shaker_surface_start_stop:** Simulation of a sphere deposit on a vibrating surface, where the surface vibration is triggered and then stopped.

## Mixer Simulation with Falling Spherical Particles

This example simulates a mixer into which spherical particles are dropped. A propeller rotates at **0.25 rad/s**, and the grain size decreases progressively to observe different dynamic behaviors.

The simulation is based on an example from the `chronoDEM::gpu` code and is described in the following paper:  
**_Chrono::GPU: An open-source simulation package for granular dynamics using the discrete element method_**

To accelerate the simulation, the time step was set to **1×10⁻³**, and the contact law parameters were adapted accordingly.

### Input Files

Available in the directory:  
`exaDEM/example/spheres/mixer`

- `mixer_57k_sph.msp`  
- `mixer_3M_sph.msp`  
- `mixer_29M_sph.msp`
