# Simulation "Depot"


In this simulation, spherical particles are confined within a cylindrical container with a base. These particles are arranged into two distinct groups, forming two cylindrical layers. The first layer consists of large type 0 spheres, extending from center to layer 1. The second layer comprises small type 1 spheres, covering the region from layer 1 to layer 2. An empty space is intentionally left between the outer walls of the container and the type 2 spheres.

## Details

- Type 0 spheres: radius = 10
- Type 1 spheres: radius = 0.5
- Layer 1: radius = 30
- Layer 2: radius = 40
- Sleeve: radius 44


1D : | CENTER | type 1 - type 1 - ... - type 1 | Layer 1 | type 2 - ... - type 2 | Layer 2 | empty | Sleeve

## Generate the data layout:

Use the generator in tools/genDepotSpheres.cpp
