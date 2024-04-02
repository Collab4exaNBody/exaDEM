---
title: 'ExaDEM: an HPC application based on ExaNBody targetting DEM simulations with spheropolyhedra'
tags:
  - DEM
  - HPC
  - N-Body
  - MPI
  - OpenMP
- GPU
authors:
  - name: Raphaël Prat
    orcid: 0009-0002-3808-5401
    affiliation: 1
  - name: Thiery Carrad
    orcid: 0000-0000-0000-0000
    affiliation: 2
  - name: Lhassan Amarsid
    orcid: 0000-0000-0000-0000
    affiliation: 1
  - name: Vincent Richefeu
    orcid: 0000-0000-0000-0000
    affiliation: 3
  - name: Guillaume Latu
    orcid: 0009-0001-7274-1305
    affiliation: 1
  - name: Carlo-eliat Donchecchi
    orcid: 0000-0000-0000-0000
    affiliation:
  - name: Jean-Mathieu Vanson
    orcid: 0000-0000-0000-0000
    affiliation: 1
affiliations:
 - name: DES/IRESNE/DEC/SESC, CEA, France
   index: 1
 - name: DAM/DIF/DPTA, CEA, France
   index: 2
 - name: CNRS ...
   index: 3
date: 16 August 2023
bibliography: paper.bib
---

# Summary 

`ExaDEM` is a Discrete Element Method (`DEM`) code developed using the exaNBody framework [CITE]. This framework provides `DEM` functionalities to model spheres and spheropolyhedra while proposing performance optimizations on current HPC plateforms. A notable aspect of `ExaDEM` is its hybrid parallelization approach, which combines the use of MPI and Threads (OpenMP). Additionally, `ExaDEM` offers compatibility with `MPI`+`GPU`s, using the `CUDA` programming model (`Onika` layer) for DEM simulations with spherical particles. Written in `C++17`, `ExaDEM` aims to provide HPC capabilities for scientific community.

# DEM Background

The `DEM` method, used to study granular media, falls within the scope of so-called N-body methods. It consists in numerically reproducing the evolution of a set of particles over time. Time is discretized into time steps and at each time step n, we solve Newton's equation f=ma to deduce the acceleration for each particle and then calculate its velocity, which will give us the new positions at time n+1. "f" is computed from interaction between particules, i.e. contact interactions or external forces such as gravity. A usual numerical scheme used is the Velocity Verlet and to model contact interaction, Hooke is widely used. The DEM method allows to simulate rigid bodies with differents shape: from spherical particle to polyhedral particles. 

A crucial point for `DEM` simulation code is dicted by the need to figure out quickly the nearest neighbor particles to compute contact interactions. The common way to do it is to use the fuse between the linked cells method and Verlet list that limits the refresh rate of Neighbor list while optimizing the neighbor search using a cartesian grid of cells (complexity of N).   

# Statement of needs

The behavior of granular media is still an open issue for the scientific community and DEM simulations improve our knowledges by studying phenomena that are unreachables, or expensive, to examine with experimentations. However, to repoduce such phenomena, we need to simulate a representative number of particles that can be thousands particles to billion of particles. To simulate thousand particles, a current single processor can achieved such simulations while simulating millions of particles required HPC ressources. This simulations we are either limited by the memory footprint, either by runtime. The `DEM` method has the advantage to be naturarly parallel and several works exist for this subjects, spatial domain decomposition, thread parallelization over cells and so on. In this paper we highlight our code ExaDEM designed to achieve large scale DEM simulation on HPC platform. This code relies on many features of the exaNBody framework, including data structure management, MPI+X parallelization and add-on modules (IO, Paraview).

# Implementation and features

`ExaDEM` takes avantage of exaNBody data structures (grid, cells, fields) and main parallel algorithms (domain depcomposition, particle migration, numerical schemes) while integrating DEM specificities.  
Parallelization: `ExaDEM` acheives an `MPI` parallelization by decomposing the simulation domain into subdomains with spatial domain decomposition and the Recursive Coordinate Bissection to evenly distrubte the workload among `MPI` processes. A subdomain is a grid of cells and particle informations are stored in cells. The use of cells allows to apply the linked cells method to speedup the neighbor search (complexity O(N), with N the number of particles) while the Verlet lists method allow to maintain bigger neighbor lists on several timesteps as long as a particle has not displaced more than 1/2 of the Vertle radius. For spherical particles, the OpenMP parallelization is done by iterating over cells while a block of GPU threads corresponds to a cell. In the case of spheropolyhedra, another level is used for thread parallelization, the interaction. Indeed, in the opposite of spherical particles, two spheropolyhedra can have multiple contacts of different types (vertex-vertex, vertex-edge, vertex-face, edge-edge), therefore it is perferable to consider interaction than particle pairs or cells. The GPU parallelizarion of spheropolyhedra is upcomming.

Data layout: 3 levels, the grid (AOSOA) associatied to a subdomain, the cell (SOA) and the field (Array). Note that the DEM grid contains the fields: type, position, velocities, accelerations, radius, angular velocties, orientation. The AOSAO data structure facilitate data movement between `MPI` processes while maintaining a good data locality, i.e. particles in a same cell or in a neighbor cell can interact. The SOA storage (cell layout) improve the use of SIMD instructions. 

Finally, `ExaDEM` provides the following features:

- Handle different particle types: spherical and spheropolyhedron particles
- Hybrid parallelisation MPI + X
	- X = OpenMP or Cuda for spherical particles
	- X = OpenMP for spheropolyhedron particles
	- The Recursive Coordinate Bissection method is used for the load balancing
- I/O support for check and restart file
- Paraview output files
- Drivers: Wall, Rotating drum, STL mesh for complex geometries ...
- Numericall Scheme: Verlet Vitess
- Contact detection: Linked-cell method and Verlet Lists
- Force fields: contact force (Hooke law), cohesive force, gravity, and quadratic force

# Performance Results

# Conclusion
