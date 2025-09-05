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

### Documenation / Pictures

See : https://collab4exanbody.github.io/doc_exaDEM/project_exaDEM/Test_cases.html#mixer-simulation
