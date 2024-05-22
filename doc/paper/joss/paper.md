---
title: 'ExaDEM: a HPC application based on ExaNBody targetting DEM simulations with polyhedron particles'
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

`ExaDEM` is a Discrete Element Method (`DEM`) code developed using the exaNBody framework [@carrard2023exanbody] at the french atomic commission (CEA). This software provides `DEM` functionalities to model spheres and polyhedra while proposing performance optimizations on current HPC platforms. A notable aspect of `ExaDEM` is its hybrid parallelization approach, which combines the use of MPI and Threads (OpenMP). Additionally, `ExaDEM` offers compatibility with `MPI`+`GPU`s, using the `CUDA` programming model (named `Onika` layer) for DEM simulations with spherical particles. Written in `C++17`, `ExaDEM` aims to provide HPC capabilities for scientific community. `ExaDEM` intends to embed the physics of interest developed in the `Rockable` code developed at CNRS.  


# Statement of needs

The behavior of granular media is still an open issue for the scientific community, and DEM simulations improve our knowledges by studying phenomena that are unreachables, or expensive, to examine with experimentations. However, to reproduce such phenomena, we need to simulate a representative number of particles that can reach thousands of particles to billion of particles. To simulate thousands of particles, a current single processor can achieve such simulations while simulating millions of particles required HPC resources. These simulations are, either limited by the memory footprint, either by the runtime. The `DEM` method has the advantage to be naturally parallel and several works exist for these subjects, spatial domain decomposition [@plimpton1995fast], thread parallelization over cells and so on. In this paper we highlight our code `ExaDEM` designed to achieve large scale DEM simulation on HPC platform. This code relies on many features of the exaNBody framework, including data structure management, MPI+X parallelization and add-on modules (IO, Paraview).

# DEM Background

The `DEM` method, used to study granular media, falls within the scope of so-called N-body methods. It consists in numerically reproducing the evolution of a set of rigid particles over time. Time is discretized into time steps and at each time step n, we solve Newton's equation f=ma to deduce the acceleration for each particle and then calculate its velocity, which will give us the new positions at time n+1. "f" is computed from interaction between particles, i.e. contact interactions or external forces such as gravity. A usual numerical scheme used is the Velocity Verlet and to model contact interaction, Hooke law is widely used. The DEM method allows to simulate rigid bodies with differents shape: from spherical particle to polyhedral particles. 

A crucial point for `DEM` simulation code is driven by the need to figure out quickly the nearest neighbor particles to compute contact interactions. The common way to do it is to use the fuse between the linked cells method [@ciccotti1987simulation] and Verlet lists [@verlet1967computer] that limits the refresh rate of neighbor lists while optimizing the neighbor search using a cartesian grid of cells (complexity of N).   

Several DEM software have been developed over the last year and propose HPC features such as LIGGGTHS [@kloss2012models] based on LAMMPS [@thompson2022lammps] data structures (Molecular Dynamics code) with spherical particles (MPI) or Blaze-DEM [@govender2018study] with spheres and polyhedra on GPU using CUDA. `ExaDEM`'s objective is to position itself in the literature as a software product that combines MPI parallelization with OpenMP thread parallelization and CUDA GPU parallelization for polyhedral and spherical particles. As LIGGGTHS with LAMMPS, `ExaDEM` takes advantage of several HPC developments done in `ExaSTAMP` (Molecular Dynamics code) [@cieren2014exastamp] that have been mutualized in the `ExaNBody` framework such as `AMR` data structures [@prat2020amr] or In-situ analysis [@dirand2018tins].

# Implementation

`ExaDEM` takes advantage of `exaNBody` data structures (grid, cells, fields) and main parallel algorithms (domain decomposition, particles migration, numerical schemes) while integrating DEM specificities. `ExaDEM` achieves a `MPI` parallelization by decomposing the simulation domain into subdomains with spatial domain decomposition and the Recursive Coordinate Bisection (RCB) partitioning method to evenly distribute the workload among `MPI` processes. A subdomain corresponds to a grid of cells while particle informations are stored into cells. The use of cells aims to apply the state-of-the-art linked cells method to speedup the neighbor search with a complexity of O(N), with N the number of particles, while the Verlet lists method aims to maintain bigger neighbor lists on several timesteps as long as a particle has not displaced more than 1/2 of the Verlet radius. Concerning the data layout, it is decomposed on two levels. The first level is associated to the grid of cells (AOSOA) that corresponds to a subdomain. The second level is the cell (SOA) composed of fields (Array) containing particle data. Note that the DEM grid contains the fields: type, position, velocities, accelerations, radius, angular velocities, orientation. The AOSAO data structure facilitates data movement between `MPI` processes while maintaining a good data locality, i.e. particles in a same cell or in a neighbor cell can interact. In addition, the use of SOA storage (cell layout) improves the use of SIMD instructions. 

About the intra-`MPI` parallelization, we distingue two main differences corresponding to the type of particle, i.e. sphere or polyhedron: 

- For spherical particles, the OpenMP parallelization is done by iterating over cells, about the GPU parallelization, a block of GPU threads is attributed to a cell and each GPU threads works on a particle. 
- For polyhedron particles, another parallel level is chosen for thread parallelization, the interaction. Indeed, in the opposite of spherical particles, two polyhedra can have multiple contacts with different types (vertex-vertex, vertex-edge, vertex-face, edge-edge), therefore it is preferable to consider interaction than cells to achieved thread-parallelization, whereas this strategy adds costly synchronizations (usagge of mutexes). The GPU parallelizarion of polyhedra is an upcoming development.

Finally, it is important to note that the design of `ExaDEM` lead by the framework `ExaNBody` allows to add or remove one operator/feature without impacting the other functionalities as long as operators are independents. For example, the gravity force operator can be removed from the `ExaDEM` repository while the contact_neighbor operator (building neighbor lists for every particle) is required to runs the Hooke force operator. Efforts have be done to limit interactions between operators in order to add or remove easily new modules/operators coded by a new developer. 

# Main features

![Simulation of near 700 thousands octahedra in a rotating drum running on 128 mpi processes with 8 OpenMP threads per mpi process processor: (AMD EPYC Milan 7763). \label{fig:rotating-drum}](./rotating-drum.png "test"){width=95%}

![Simulation of near 20 million spherical particles falling in a funnel. This simulation runs on 512 mpi processes with 8 OpenMP threads per mpi process (processor: (AMD EPYC Milan 7763).  \label{fig:funnel}](./funnel.png "test"){width=80%}

`ExaDEM` attends to meet scientific expectations, especially for fuel nuclear simulations consisting in rotating drum (see figure \ref{fig:rotating-drum}) or compression simulations. To do it, `ExaDEM` provides the following features:

- Handle different particle types: spherical and polyhedral particles,
- Hybrid parallelisation MPI + X,
	- X = OpenMP or CUDA for spherical particles,
	- X = OpenMP for polyhedron particles,
	- The Recursive Coordinate Bissection method is used for the load balancing,
- I/O support for check and restart files (MPIIO files),
- Paraview output files containing fieds,
- Drivers: Wall, Rotating drum or mesh of polyhedron surface for complex geometries such as funnel (see figure \ref{fig:funnel}),
- Numericall Scheme: Verlet Vitess,
- Contact detection: Linked-cell method and Verlet Lists,
- Force fields: contact force (Hooke law), cohesive force, gravity, and quadratic force,

All these functionalities are subject to evolution in line with new development needs, such as the addition of particle fragmentation. Note that most of these functionalities have been tested over 500 million spheres or 10 million polyhedra over ten thousand cores in `MPI` + `OpenMP` on AMD EPYC Milan 7763 processors.

# Future of `ExaDEM`

In the upcoming years, we plan to add several `DEM` features such as complex geometries (propeller), particle fragmentation, or more diagnostics. On the other hand, we will develop other parallel strategies, especially on GPU, to run on future supercomputers. A specific focus will be done for simulation polyhedron particles. `ExaDEM` is available under an APACHE 2 license at https://github.com/Collab4exaNBody/exaDEM.git and the documentation is available at https://github.com/Collab4exaNBody/doc_exaDEM.git.


# Acknowledgement

This work was performed using HPC resources from CCRT funded by the CEA/DEs simulation programme.
