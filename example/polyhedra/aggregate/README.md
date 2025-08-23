# Example: Falling Aggregates into an Empty Cup

This simulation demonstrates the fall of aggregates into an empty cup.  
The main purpose of this example is to provide another test case involving  
computationally expensive geometries.

To increase the sample size, you can enlarge the generation area (`BOX`)  
and the cup size by adjusting the `resize` value when registering  
the driver `cup.stl`.

> **Note**  
> Remember to increase the simulation time accordingly when scaling up  
> the geometry.


```
cp -R exaDEM/example/polyhedra/aggregate
cd aggregate
./exaDEM aggregate.msp --omp_num_threads 12
```

Duration on a latop with 12 cores: 144 s while `nbh_polyhedra` takes 137 s (95%)
