# Data configuration

We generate a set of around 3250 spheres with radii ranging from 0.1 to 0.5 [dimension in space] which we drop onto a piston in a cylindrical cell. In a second step, we set a piston in motion above the spheres to compress the sample.

# Commands

## Create the sample

```
mpirun -n 1 ../../exaDEM init_piston.msp --omp_num_threads 12
```

## Linear Force Motion

```
mpirun -n 1 ../../exaDEM piston_linear_force_motion.msp --omp_num_threads 12
```

## Linear Compression Motion

```
mpirun -n 1 ../../exaDEM piston_linear_compression_motion.msp --omp_num_threads 12
```
