# Polyhedra Test Cases

This document provides a brief list of examples to showcase different test cases for polyhedra simulations. For more detailed information, please refer to the documentation [here](https://collab4exanbody.github.io/doc_exaDEM/project_exaDEM/Test_cases.html)

## Generator

This example demonstrates how to add a lattice generator for particles. For details, see the file generator/generator.msp.

## Use Drivers

### Rotating Drum

Two test cases illustrate the use of a rotating drum:

- A small test case with 125 octahedra falling into a drum: example/polyhedra/rotating_drum/rotating-drum.msp.
- A mixed case with hexapods and octahedra falling into a rotating drum: rotating_drum/rotating-drum-mixte.msp.

### Big Sphere/Ball

An example demonstrating the "ball" driver with hexapods is available in: balls/balls_full.msp.
STL Meshes

Three main examples showcase simulations involving large shapes imported via STL files:

- Hexapods: stl_mesh/stl_mesh_box_hexapod.msp
- Octahedra: stl_mesh/stl_mesh_box_octahedron.msp
- Hexapods + Octahedra: stl_mesh/stl_mesh_box_mixte.msp

An additional example of a particle falling into a funnel can be found at: funnel/funnel.msp.
