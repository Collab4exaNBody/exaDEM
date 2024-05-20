#include <chrono>
#include <ctime>
#include <mpi.h>
#include <string>
#include <numeric>

#include <exanb/core/basic_types_yaml.h>
#include <exanb/core/operator.h>
#include <exanb/core/operator_slot.h>
#include <exanb/core/operator_factory.h>
#include <exanb/core/make_grid_variant_operator.h>
#include <exanb/core/grid.h>
#include <exanb/core/basic_types_stream.h>
#include <exanb/core/log.h>
//#include "ustamp/vector_utils.h"
#include <exanb/core/file_utils.h>
#include <exanb/core/domain.h>
#include <exanb/core/check_particles_inside_cell.h>
#include <exanb/core/simple_block_rcb.h>


// rsa_mpi stuff
#include <rsa_data_storage.hxx>
#include <rsa_random.hxx>
#include <rsa_domain.hxx>
#include <rsa_decoration.hxx>
#include <operator_algorithm.hxx>
#include <radius_generator.hxx>



namespace exaDEM
{
using namespace exanb;

  struct ParticleType
  {
		static inline constexpr size_t MAX_STR_LEN = 16;

		double m_mass = 1.0;
		double m_radius = 1.0;
		char m_name[MAX_STR_LEN] = {'\0'};

		inline void set_name(const std::string& s)
		{
			if( s.length() >= MAX_STR_LEN ) { std::cerr<<"Particle name too long : length="<<s.length()<<", max="<<(MAX_STR_LEN-1)<<"\n"; std::abort(); }
			std::strncpy(m_name,s.c_str(),MAX_STR_LEN); m_name[MAX_STR_LEN-1]='\0';
		}
		inline std::string name() const { return m_name; }    
	};

	using ParticleTypes = onika::memory::CudaMMVector<ParticleType>;

	template<typename GridT>
		class InitRSA : public OperatorNode
	{
		ADD_SLOT( MPI_Comm        , mpi          , INPUT , MPI_COMM_WORLD  );
		ADD_SLOT( ReadBoundsSelectionMode, bounds_mode   , INPUT , ReadBoundsSelectionMode::FILE_BOUNDS );
		ADD_SLOT( Domain          , domain       , INPUT_OUTPUT );
		ADD_SLOT( GridT           , grid         , INPUT_OUTPUT );
		ADD_SLOT( double          , enlarge_bounds, INPUT , 0.0 );
		ADD_SLOT( std::vector<bool> , periodicity     , INPUT ,OPTIONAL , DocString{"if set, overrides domain's periodicity stored in file with this value"}  );
		ADD_SLOT( bool        , expandable      , INPUT ,OPTIONAL , DocString{"if set, override domain expandability stored in file"} );
		ADD_SLOT( AABB        , bounds          , INPUT ,REQUIRED , DocString{"if set, override domain's bounds, filtering out particle outside of overriden bounds"} );
		ADD_SLOT( int         , type , INPUT , 0 );
    ADD_SLOT( bool        , pbc_adjust_xform , INPUT , true );

		ADD_SLOT( double   , radius , INPUT , REQUIRED ); // optional. if no species given, type ids are allocated automatically

		public:
		inline void execute () override final
		{
			//-------------------------------------------------------------------------------------------
			using ParticleTupleIO = onika::soatl::FieldTuple<field::_rx, field::_ry, field::_rz, field::_id, field::_type>;
			using ParticleTuple = decltype( grid->cells()[0][0] );

			assert( grid->number_of_particles() == 0 );

			// MPI Initialization
			int rank=0, np=1;
			MPI_Comm_rank(*mpi, &rank);
			MPI_Comm_size(*mpi, &np);
			[[maybe_unused]] uint64_t n_particles = 0;

			AABB b = *bounds;
      constexpr int DIM = 3;
			constexpr int method = 1;
      constexpr int ghost_layer = 1;
      std::array<double, DIM> domain_inf = {b.bmin.x, b.bmin.y, b.bmin.z};
      std::array<double, DIM> domain_sup = {b.bmax.x, b.bmax.y, b.bmax.z};
      rsa_domain<DIM> rsa_domain(domain_inf, domain_sup, ghost_layer, *radius);

      //rsa_domain.domain_log();
      size_t seed = 0;
      algorithm::uniform_generate<DIM, method>(rsa_domain, *radius, 6000, 10, seed);
      auto& cells = rsa_domain.get_grid();
      remove_ghost(cells);

			if(rank==0)
			{
				compute_domain_bounds(*domain,*bounds_mode,*enlarge_bounds, b, b, *pbc_adjust_xform );
			}

			//send bounds and size_box values to all cores
			MPI_Bcast( & (*domain), sizeof(Domain), MPI_CHARACTER, 0, *mpi );
			assert( check_domain(*domain) );
			grid->set_offset( IJK{0,0,0} );
			grid->set_origin( domain->bounds().bmin );
			grid->set_cell_size( domain->cell_size() );
			grid->set_dimension( domain->grid_dimension() );

			// add particles
			std::vector<ParticleTupleIO> particle_data;
			ParticleTupleIO pt;
			for (int c = 0 ; c < cells.get_size() ; c++ )
			{
				auto& cell = cells.get_data(c);
				for (size_t s = 0 ; s < cell.get_size() ; s++)
				{
					Vec3d pos;
					pos.x = cell.get_center(s, 0);
					pos.y = cell.get_center(s, 1);
					pos.z = cell.get_center(s, 2);
					auto id = cell.get_id(s);
					std::cout << pos << std::endl;
					pt = ParticleTupleIO( pos.x + b.bmin.x, pos.y + b.bmin.y, pos.z + b.bmin.z, id, *type );
					particle_data.push_back(pt);
				}
			}


			// Fill grid, particles will migrate accross mpi processed via the operator migrate_cell_particles
			for( auto p : particle_data )
			{
				Vec3d r{ p[field::rx] , p[field::ry] , p[field::rz] };
				IJK loc = domain_periodic_location( *domain , r ); //grid.locate_cell(r);
				assert( grid->contains(loc) );
				assert( min_distance2_between( r, grid->cell_bounds(loc) ) < grid->epsilon_cell_size2() );
				p[field::rx] = r.x;
				p[field::ry] = r.y;
				p[field::rz] = r.z;
				ParticleTuple t = p;
				grid->cell( loc ).push_back( t , grid->cell_allocator() );
			}

			// Display information
			lout << "=================================" << std::endl;
			lout << "Particles        = "<<particle_data.size()<<std::endl;
			lout << "Domain XForm     = "<<domain->xform()<<std::endl;
			lout << "Domain bounds    = "<<domain->bounds()<<std::endl;
			lout << "Domain size      = "<<bounds_size(domain->bounds()) <<std::endl;
			lout << "Real size        = "<<bounds_size(domain->bounds()) * Vec3d{domain->xform().m11,domain->xform().m22,domain->xform().m33} <<std::endl;
			lout << "Cell size        = "<<domain->cell_size()<<std::endl;
			lout << "Grid dimensions  = "<<domain->grid_dimension()<<" ("<<grid_cell_count(domain->grid_dimension())<<" cells)"<< std::endl;
			lout << "=================================" << std::endl;

			grid->rebuild_particle_offsets();
		}

	};

	// === register factories ===
	__attribute__((constructor)) static void register_factories()
	{
		OperatorNodeFactory::instance()->register_factory("init_rsa", make_grid_variant_operator< InitRSA >);
	}

}
