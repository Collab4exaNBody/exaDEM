#pragma xstamp_grid_variant

#include <exanb/core/operator.h>
#include <exanb/core/operator_slot.h>
#include <exanb/core/operator_factory.h>
#include <exanb/core/log.h>
#include <exanb/core/basic_types_stream.h>
#include <exanb/core/grid.h>
#include <exanb/core/domain.h>
#include <exanb/core/make_grid_variant_operator.h>
#include <exanb/grid_cell_particles/grid_cell_values.h>

#include <exanb/mpi/migrate_cell_particles.h>
#include <exaDEM/neighbor_friction.h>

#include <mpi.h>
#include <vector>
#include <cstring>


namespace exaDEM
{
	using namespace exanb;

	// operator interface
	template<class GridT, class CellValueMergeOperatorT = exanb::UpdateValueAdd >
		class MigrateCellParticlesFriction : public OperatorNode
	{
		using CellParticles = typename GridT::CellParticles;
		using ParticleTuple = typename CellParticles::TupleValueType; // decltype( GridT().cells()[0][0] );
		using ParticleBuffer = std::vector<ParticleTuple>;
		//using BufListIt = typename std::list<ParticleBuffer>::iterator;
		//using GridCellValueType = typename GridCellValues::GridCellValueType;
		using MergeOp = CellValueMergeOperatorT;

		// -----------------------------------------------
		// Operator slots
		// -----------------------------------------------
		ADD_SLOT( MPI_Comm  , mpi        , INPUT , MPI_COMM_WORLD );
		ADD_SLOT( long      , mpi_tag    , INPUT , 0 );
		ADD_SLOT( MergeOp   , merge_func , INPUT, MergeOp{} );

		ADD_SLOT( GridBlock , lb_block   , INPUT , REQUIRED );
		ADD_SLOT( double    , ghost_dist , INPUT , REQUIRED );
		ADD_SLOT( Domain    , domain     , INPUT , REQUIRED );
		ADD_SLOT( GridT     , grid       , INPUT_OUTPUT);

		// optional particles "outside the box", not in the grid. (possibly due to move_particles operator)
		// it is in/out because elements of this array may be swapped (order changed), though it will contain the same element after operation
		ADD_SLOT( ParticleBuffer , otb_particles, INPUT_OUTPUT , DocString{"Particles outside of local processor's grid"} );

		// MPI buffer size for exchanges
		ADD_SLOT( long , buffer_size, INPUT , 100000 , DocString{"Performance tuning parameter. Size of send/receive buffers in number of particles"} );

		// threshold, in number of particles, above which a task is generated to copy cell content to send buffer
		ADD_SLOT( long , copy_task_threshold, INPUT , 256 , DocString{"Performance tuning parameter. Number of particles in a cell above which an asynchronous OpenMP task is created to pack particles to send buffer"} );

		// number of receive buffers per peer process. if negative, interpreted as -N * mpi_processes
		ADD_SLOT( long , extra_receive_buffers, INPUT , -1 , DocString{"Performance tuning parameter. Number of extraneous receive buffers allocated allowing for asynchronous (OpenMP task) particle unpacking. A negative value n is interpereted as -n*NbMpiProcs"} );

		// optional : set to true to force complete destruction and creation of the grid even when lb_block hasn't changed
		ADD_SLOT( bool , force_lb_change, INPUT ,false , DocString{"Force particle packing/unpacking to and from send buffers even if a load balancing has not been triggered"} );

		// optional per cell set of scalar values
		ADD_SLOT( GridCellValues , grid_cell_values  , INPUT_OUTPUT , OPTIONAL );    

		ADD_SLOT( GridCellParticleNeigborFriction , nbh_friction , INPUT_OUTPUT , GridCellParticleNeigborFriction{} , DocString{"Neighbor particle friction term"} );
		ADD_SLOT( ParticleNeighborFrictionCellMoveBuffer , nbh_friction_otb , INPUT_OUTPUT, ParticleNeighborFrictionCellMoveBuffer{} , DocString{"friction data of particles moving outside the box"} );

		public:
		// -----------------------------------------------
		// -----------------------------------------------
		inline std::string documentation() const override final
		{
			return R"EOF(
migrate_cell_particles does 2 things :
1. it repartitions the data accross mpi processes, as described by lb_block.
2. it reserves space for ghost particles, but do not populate ghost cells with particles.
The ghost layer thickness (in number of cells) depends on ghost_dist.
inputs from different mpi process may have overlapping cells (but no duplicate particles).
the result grids (of every mpi processes) never have overlapping cells.
the ghost cells are always empty after this operator.
)EOF";
		}

		// -----------------------------------------------
		// -----------------------------------------------
		inline void execute () override final
		{
			const long cptask_threshold = *copy_task_threshold;
			const int comm_tag = *mpi_tag;
			MPI_Comm comm = *mpi;
			const MergeOp & merge_func = *(this->merge_func);
			GridBlock out_block = *lb_block;
			size_t comm_buffer_size = *buffer_size;
			const long extra_receive_buffers = *(this->extra_receive_buffers);
			bool force_lb_change = *(this->force_lb_change);
			ParticleBuffer& otb_particles = *(this->otb_particles);
			const Domain& domain = *(this->domain);
			GridT& grid = *(this->grid);
			const double max_nbh_dist = *ghost_dist;

			GridCellValues * grid_cell_values_ptr = nullptr;
			if( grid_cell_values.has_value() ) grid_cell_values_ptr = & (*grid_cell_values) ;

			NeighborFrictionParticleMigrationHelper friction_migration_helper = { nbh_friction->m_cell_friction , *nbh_friction_otb };

			exanb::MigrateCellParticlesImpl<GridT,CellValueMergeOperatorT>::migrate_cell_particles(
					ldbg
					, cptask_threshold
					, comm_tag
					, comm
					, merge_func
					, out_block
					, comm_buffer_size
					, extra_receive_buffers
					, force_lb_change
					, otb_particles
					, domain
					, grid
					, max_nbh_dist
					, grid_cell_values_ptr
					, friction_migration_helper
					);
		}

	};


	// helper template to avoid bug in older g++ compilers
	template<class GridT> using MigrateCellParticlesFrictionTmpl = MigrateCellParticlesFriction<GridT>;

	// === register factory ===
	CONSTRUCTOR_FUNCTION
	{
		OperatorNodeFactory::instance()->register_factory( "migrate_cell_particles_friction", make_grid_variant_operator<MigrateCellParticlesFrictionTmpl> );
	}

}

