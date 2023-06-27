//#pragma xstamp_cuda_enable //! DO NOT REMOVE THIS LINE

#include <exanb/core/operator.h>
#include <exanb/core/operator_slot.h>
#include <exanb/core/operator_factory.h>
#include <exanb/core/make_grid_variant_operator.h>
#include <exanb/core/parallel_grid_algorithm.h>
#include <exanb/core/grid.h>
#include <exanb/compute/compute_cell_particles.h>
#include <mpi.h>
#include <memory>
#include <exaDEM/dem_simulation_state.h>

namespace exaDEM
{
  using namespace exanb;

	struct DrivePebbleRadiusFunctor
	{
		double m_dr;
		double m_rmax;
		long int m_modulo;

		ONIKA_HOST_DEVICE_FUNC inline void operator () (uint64_t a_id, double& a_radius) const
		{
			if(a_id % m_modulo == 0)
			{
				if(a_id != 0) // skip first particle, last particle is skipped by construction
				{
					const double new_radius = a_radius + m_dr;
					if(new_radius <m_rmax) {
						a_radius = new_radius;
					}
				}
			}
		}
	};
}

namespace exanb
{
	template<> struct ComputeCellParticlesTraits<exaDEM::DrivePebbleRadiusFunctor>
	{
		static inline constexpr bool RequiresBlockSynchronousCall = false;
		static inline constexpr bool CudaCompatible = true;
	};
}

namespace exaDEM
{
  using namespace exanb;

	template<typename GridT
		, class = AssertGridHasFields< GridT, field::_id, field::_radius>
		>
		class DrivePebbleRadius : public OperatorNode
		{
			static constexpr double default_dr = 0.0;
			static constexpr double default_rmax = 0.0;
			// attributes processed during computation
			using ComputeFields = FieldSet< field::_id, field::_radius>;
			static constexpr ComputeFields compute_field_set {};

			ADD_SLOT( GridT           , grid        	, INPUT_OUTPUT );
			ADD_SLOT( MPI_Comm        , mpi         	, INPUT , MPI_COMM_WORLD);
			ADD_SLOT( double          , dr          	, INPUT , default_dr 	, DocString{"Distance added to the radius at each timestep"});
			ADD_SLOT( double          , rmax        	, INPUT , default_rmax	, DocString{"Maximum radius size of the sphere"});
			ADD_SLOT( int             , number_of_pebbles 	, INPUT, 0		, DocString{"Define the number of pebbles wanted"});
			ADD_SLOT( long int        , number_of_particles , INPUT_OUTPUT, -1	, DocString{"Put the number of particles or do not set this slot, this slot will be automaticly filled at the first iteration"});

			public:

			inline std::string documentation() const override final
			{
				return R"EOF(
        This operator drives the pebble radius for a set of "number_of_particles" particles.
        )EOF";
			}

			inline void execute () override final
			{
				if(*number_of_pebbles == 0) return;
				if(*number_of_particles == -1)
				{
					MPI_Comm comm = *mpi;
					long int total_particles = 0;
					auto cells = grid->cells();
					IJK dims = grid->dimension();
					size_t ghost_layers = grid->ghost_layers();
					IJK dims_no_ghost = dims - (2*ghost_layers);
#       pragma omp parallel
					{
						GRID_OMP_FOR_BEGIN(dims_no_ghost,_,loc_no_ghosts, reduction(+:total_particles) )
						{
							IJK loc = loc_no_ghosts + ghost_layers;
							size_t cell_i = grid_ijk_to_index(dims,loc);
							long int n = cells[cell_i].size();
							total_particles += n;
						}
						GRID_OMP_FOR_END;

						MPI_Allreduce(MPI_IN_PLACE,&total_particles,1,MPI_LONG,MPI_SUM,comm);
						*number_of_particles = total_particles;
					}
				}
				const auto modulo = *number_of_particles / (*number_of_pebbles + 1);
				DrivePebbleRadiusFunctor func { *dr , *rmax, modulo};
				compute_cell_particles( *grid , false , func , compute_field_set , gpu_execution_context() , gpu_time_account_func() );
			}
		};

	template<class GridT> using DrivePebbleRadiusTmpl = DrivePebbleRadius<GridT>;

	// === register factories ===  
	CONSTRUCTOR_FUNCTION
	{
		OperatorNodeFactory::instance()->register_factory( "drive_pebble_radius", make_grid_variant_operator< DrivePebbleRadiusTmpl > );
	}
}

