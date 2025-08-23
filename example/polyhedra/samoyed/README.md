# Example: Falling Complex Shape

This simulation demonstrates the fall of a relatively expensive shape  
(in this case, a Samoyed) into a cup.  
The main purpose of this example is to provide a second test case involving  
computationally expensive geometries.

To increase the sample size, you can enlarge the generation area (`BOX`)  
and the cup size by adjusting the `resize` value when registering  
the driver `cup.stl`.

> **Note**  
> Remember to increase the simulation time accordingly when scaling up  
> the geometry.

```
cp -R exaDEM/example/polyhedra/samoyed
cd samoyed
./exaDEM samoyed.msp --omp_num_threads 12
```

Duration on a laptop of 12 cores: 202 seconds with 95% of the total time spent in `nbh_polyhedra`
