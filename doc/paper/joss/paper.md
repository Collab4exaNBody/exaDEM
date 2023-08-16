---
title: 'MFEM-MGIS-MFRONT, a HPC mini-application targeting nonlinear thermo-mechanical simulations of nuclear fuels at mesoscale'
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
  - name: Guillaume Latu
    orcid: 0009-0001-7274-1305
    affiliation: 1
affiliations:
 - name: DES/IRESNE/DEC/SESC, CEA, France
   index: 1
 - name: DAM/DIF/DPTA, CEA, France
   index: 2
date: 16 August 2023
bibliography: paper.bib
---

# Summary 

`ExaDEM` is a software solution in the field of computational simulations. It's a Discrete Element Method (`DEM`) code developed within the exaNBody framework. This framework provides the basis for `DEM` functionalities and performance optimizations. A notable aspect of `ExaDEM` is its hybrid parallelization approach, which combines the use of MPI (Message Passing Interface) and Threads (OpenMP). This combination aims to enhance computation times for simulations, making them more efficient and manageable.
Additionally, ExaDEM offers compatibility with `MPI`+`GPU`s, using the `CUDA` programming model (`Onika` layer). This feature provides the option to leverage `GPU` processing power for potential performance gains in simulations. Written in `C++17`, `ExaDEM` is built on a contemporary codebase. It aims to provide researchers and engineers with a tool for adressing `DEM` simulations.

# DEM Background

The `DEM` method falls within the scope of so-called N-body methods used to study granular media. It consists in numerically reproducing the evolution of a set of particles over time. Time is discretized into time steps and at each time step n, we solve Newton's equation f=ma to deduce the acceleration for each particle and then calculate its velocity, which will give us the new positions at time n+1. "f" is computed from interaction between particules, i.e. contact interactions, short range interactions or external forces such as gravity. A usual numerical scheme used is the Velocity Verlet and to model contact interaction, Hooke or Hertz laws are widely used.

DEM allows to simulate rigid bodies with differents shape : from spherical particle to plyhedral particles. Concerning exaDEM, we aims to model spherical particles and extend our code to polyhedral particles in a future work.

A crucial point for `DEM` simulation code is dicted by the need to figure out quickly the nearest neighbor particles to compute contact interactions. The common way to do it is to use the fuse between the linked cells method and Verlet list that limits the refresh rate of Neighbor list while optimizing the neighbor search using a cartesian grid of cells (complexity of N).   

# Statement of needs

The behavior of granular media is still an open issue for the scientific community and DEM simulations improve our knowledges by studying phenomena impossible, or expensive, to examine with experimentations. However, to observe such phenomena, we need to simulate a representative number of particles that can be thousands particles to billion of particles. To simulate thousand particles, a current single processor can achieved such simulation but to simulate more than 1 million particles, we are limited either by the memory limit, either by runtime. To tackle this issue, a lot of scientific code use HPC tools these codes run on HPC platforms. The `DEM` method has the advantage to be naturarly parallel and several works exist for this subjects, spatial domain decomposition, thread parallelization over cells and so on. In this paper we highlight our code ExaDEM designed to achieve large scale DEM simulation on HPC platform. This code relies on many features of the exaNBody framework, including data structure management, MPI+X parallelization and add-on modules (IO, Paraview).    

# Software stack

# Performance Results

# Conclusion
