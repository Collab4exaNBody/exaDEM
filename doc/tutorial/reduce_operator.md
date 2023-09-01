# Reduction Operator

1 reduction operation refers to a common parallel computing technique used to aggregate or combine data from multiple processes or threads into a single result. Reduction operations are frequently employed in parallel and distributed computing environments to perform operations like summation, finding the maximum or minimum value, etc.

In this section we consider the summation of particle volumes. 

Example :

```
V = sum_i (Vi)
```

## Requirements

You need to specify:
- Data structure (double, int or your own data structure composed of double, int, ...): The volume is double. 
- Fieds required: radius cutoff.
- Local kernel: `local_volume += 4/3 * pi * rcut * rcut * rcut;`
- Kind of reduction: summation
- The initial value: 0

## Apply your reduction operator

The reduce routine is given by `ONIKA` and the routine `reduce_cell_particles`. As `compute_cells_particles`, you have to define which fields are required such as: 

```
static constexpr FieldSet<field::_rcut> reduce_field_set {};
```

The reduction type is defined in your reduction functor `TotalVolume` and the reduction is runned as follow:

```
TotalVolume func {};
double vol = 0; // particle volume
double total_vol = reduce_cell_particles( *grid , false , func , vol, reduce_field_set , gpu_execution_context() , gpu_time_account_func() )
```

## How to write your own structure to make a reduction working on CPU and GPU

A reduction operator needs three `operator()` to carry on the reduction on CPU and GPU. Given our example, this is the corresponding structure:

```
namespace exaDEM
{
  using namespace exanb;
  struct TotalVolume
  {
    constexpr double pi = 3,14159265358979323846264338327950288419716939937510582;
    ONIKA_HOST_DEVICE_FUNC inline void operator () (double& local_variable, double rcut, reduce_thread_local_t={} ) const
    {
      local_variable += 4/3 * pi * rcut * rcut * rcut;
    }

    ONIKA_HOST_DEVICE_FUNC inline void operator () ( double& global, double local, reduce_thread_block_t ) const
    {
      ONIKA_CU_ATOMIC_ADD( global , local );
    }

    ONIKA_HOST_DEVICE_FUNC inline void operator () (double& global , double local, reduce_global_t ) const
    {
      ONIKA_CU_ATOMIC_ADD( global , local );
    }
  };

};
```

To run this simulation on GPU, you need to add CudaCompatible Trait such as:
```
namespace exanb
{
  template<> struct ReduceCellParticlesTraits<exaDEM::TotalVolume>
  {
    static inline constexpr bool RequiresBlockSynchronousCall = false;
    static inline constexpr bool RequiresCellParticleIndex = false;
    static inline constexpr bool CudaCompatible = true;
  };
};
```


