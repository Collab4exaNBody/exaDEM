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

## Driver Extractor

This example demonstrates the use of a driver extractor in a rotating drum simulation with side surfaces (driver-extractor.msp).
The operator used to add trackers is: `register_driver_extractor`.
To manage the frequency use: `simulation_driver_extractor_frequency: 10`.

## Force-controlled Rigid Frame

This example shows how to drive a group of particles as a rigid frame with the `compute_frame_velocity` operator: `impose_frame/movable_frame_force.msp`.

Spheres are placed in a box made of rigid surfaces. The particles located in the `FRAME` region (top layer) are frozen and move together as a rigid body. Each velocity component of the frame is either imposed (`velocity`) or computed from a virtual rigid body driven by a target force (`force`):

M dv/dt = F_target(t) + F_specimen - c v

where F_specimen is the sum of the forces of the frame particles. Here, x and y are velocity-controlled (zero velocity) and z is force-controlled: a target force of -72 (0.5 x 144, i.e. a pressure applied on the top of the sample) compresses the sample from above.

The operator only computes `frame_velocity`; it is applied with `freeze_particles` and `set_velocity` (value rebound to `frame_velocity`). Optional parameters: `force_ramp_start` / `force_ramp_steps` (linear force ramp), `frame_mass`, `frame_damping`, `frame_max_velocity` (velocity cap) and `frame_log_frequency` (prints the target force, the specimen force and the frame velocity).

## Hooke Contact Law Verification

Single-impact tests adapted from Chung & Ooi, *"Benchmark tests for verifying discrete element modelling codes at particle impact level"*, Granular Matter 13, 643-656 (2011), to the linear (Hooke) contact law: elastic impact of two identical spheres, elastic impact of a sphere on a rigid plane, damped impacts with different restitution coefficients. The results (velocity, normal force vs overlap and vs time) are checked against the exact analytical solution as non-regression tests. See [`verif_hooke/README.md`](verif_hooke/README.md).

## Hertz Contact Law Verification

Normal impact tests 1, 2 and 3 of Chung & Ooi (2011) with the Hertz contact law (spheres only, `contact_sphere_hertz_[none|cohesive|dmt]`) and the materials of the paper: elastic impact of two identical spheres (glass, limestone), elastic impact of a sphere on a rigid plane, impacts with different restitution coefficients. The maximal overlap, force and contact time match Table 3 of the paper, and the results are checked against the solution of the Hertz impact as non-regression tests. See [`verif_hertz/README.md`](verif_hertz/README.md).

