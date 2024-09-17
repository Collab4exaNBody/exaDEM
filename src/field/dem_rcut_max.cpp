/*
Licensed to the Apache Software Foundation (ASF) under one
or more contributor license agreements.  See the NOTICE file
distributed with this work for additional information
regarding copyright ownership.  The ASF licenses this file
to you under the Apache License, Version 2.0 (the
"License"); you may not use this file except in compliance
with the License.  You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.  See the License for the
specific language governing permissions and limitations
under the License.
*/
//#pragma xstamp_cuda_enable // DO NOT REMOVE THIS LINE
#include <exanb/core/operator.h>
#include <exanb/core/operator_slot.h>
#include <exanb/core/operator_factory.h>
#include <exanb/core/make_grid_variant_operator.h>
#include <exanb/core/parallel_grid_algorithm.h>
#include <exanb/core/grid.h>
#include <exanb/compute/reduce_cell_particles.h>
#include <mpi.h>



namespace exaDEM
{
  using namespace exanb;

  struct DEMRcutMaxFunctor
  {
    ONIKA_HOST_DEVICE_FUNC inline void operator () (double& local_variable, const double rad, reduce_thread_local_t={} ) const
    {
      std::cout << rad << std::endl;
      local_variable = std::max(local_variable, 2*rad);
    }

    ONIKA_HOST_DEVICE_FUNC inline void operator () ( double& global, double local, reduce_thread_block_t ) const
    {
      ONIKA_CU_ATOMIC_MAX( global, local );
    }

    ONIKA_HOST_DEVICE_FUNC inline void operator () (double& global , double local, reduce_global_t ) const
    {
      ONIKA_CU_ATOMIC_MAX( global, local );
    }
  };

};
namespace exanb
{
  template<> struct ReduceCellParticlesTraits<exaDEM::DEMRcutMaxFunctor>
  {
    static inline constexpr bool RequiresBlockSynchronousCall = false;
    static inline constexpr bool RequiresCellParticleIndex = false;
    static inline constexpr bool CudaCompatible = true;
  };
};

namespace exaDEM
{
	using namespace exanb;

	template<typename GridT
		, class = AssertGridHasFields< GridT, field::_radius>
		>
		class DEMRcutMax : public OperatorNode
		{
      using ReduceFields = FieldSet<field::_radius>;
      static constexpr ReduceFields reduce_field_set {};

      ADD_SLOT( MPI_Comm , mpi      , INPUT , MPI_COMM_WORLD);
			ADD_SLOT( GridT    , grid     , INPUT , REQUIRED );
			ADD_SLOT( double   , rcut_max , INPUT_OUTPUT , 0.0 );

			// -----------------------------------------------
			// ----------- Operator documentation ------------
			inline std::string documentation() const override final
			{
				return R"EOF(Fill rcut_max with the maximum of the radii. )EOF";
			}

			public:
			inline void execute () override final
			{
        auto& g = *grid;
        const auto cells = g.cells();
        const size_t n_cells = g.number_of_cells(); // nbh.size();
				std::cout << "grid size = " << n_cells << std::endl;
				double res = 0.0;
        MPI_Comm comm = *mpi;
        DEMRcutMaxFunctor func;
        reduce_cell_particles( *grid , false , func , res, reduce_field_set , parallel_execution_context());
        MPI_Allreduce(MPI_IN_PLACE, &res, 1, MPI_DOUBLE, MPI_MAX, comm);
        double& rmax = *rcut_max; 
        rmax = std::max(rmax, res);
        lout << "rcut max is equal to: " << rmax << std::endl;
			}

		};

	template<class GridT> using DEMRcutMaxTmpl = DEMRcutMax<GridT>;

	// === register factories ===  
	CONSTRUCTOR_FUNCTION
	{
		OperatorNodeFactory::instance()->register_factory( "dem_rcut_max", make_grid_variant_operator< DEMRcutMaxTmpl > );
	}
}

