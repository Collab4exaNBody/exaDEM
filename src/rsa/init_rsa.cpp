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
		ADD_SLOT( AABB        , bounds          , INPUT ,OPTIONAL , DocString{"if set, override domain's bounds, filtering out particle outside of overriden bounds"} );
		ADD_SLOT( ParticleTypes   , particle_types , INPUT ); // optional. if no species given, type ids are allocated automatically

		public:
		inline void execute () override final
		{
			//-------------------------------------------------------------------------------------------
			using ParticleTupleIO = onika::soatl::FieldTuple<field::_rx, field::_ry, field::_rz, field::_id, field::_shape>;
			using ParticleTuple = decltype( grid->cells()[0][0] );

			assert( grid->number_of_particles() == 0 );

			// MPI Initialization
			int rank=0, np=1;
			MPI_Comm_rank(*mpi, &rank);
			MPI_Comm_size(*mpi, &np);
			uint64_t n_particles = 0;

			std::map<std::string,unsigned int> typeMap;
			unsigned int nextTypeId = 0;
			if( particle_types.has_value() )
			{
				for(size_t i=0;i<particle_types->size();i++)
				{
					typeMap[particle_types->at(i).m_name]=i;
				}
				nextTypeId = particle_types->size();
			}



      if( *init_domain )
      {    
        const double box_size_x = repeats->i * size.x;
        const double box_size_y = repeats->j * size.y;
        const double box_size_z = repeats->k * size.z;
        domain_size = Vec3d{ box_size_x, box_size_y, box_size_z };

        if(rank==0)
        {
	        AABB lattice_bounds  = { { 0., 0., 0. } , {box_size_x,box_size_y,box_size_z} };
	        ldbg << "Lattice bounds      = "<<lattice_bounds<<std::endl;

	        AABB computed_bounds = lattice_bounds;
	        ldbg << "Computed_bounds  = " << computed_bounds << std::endl;

          compute_domain_bounds(*domain,*bounds_mode,*enlarge_bounds,lattice_bounds,computed_bounds, *pbc_adjust_xform );
        }
        
        //send bounds and size_box values to all cores
        MPI_Bcast( & (*domain), sizeof(Domain), MPI_CHARACTER, 0, *mpi );
        assert( check_domain(*domain) );

        // compute local processor's grid size and location so that cells are evenly distributed
        GridBlock in_block = { IJK{0,0,0} , domain->grid_dimension() };
        GridBlock out_block = simple_block_rcb( in_block, np, rank );
        ldbg<<"Domain = "<< *domain << std::endl;
        ldbg<<"In  block = "<< in_block << std::endl;
        ldbg<<"Out block = "<< out_block << std::endl;

        // initializes local processor's grid
        grid->set_offset( out_block.start );
        grid->set_origin( domain->bounds().bmin );
        grid->set_cell_size( domain->cell_size() );
        local_grid_dim = out_block.end - out_block.start;
        grid->set_dimension( local_grid_dim );      
      }




			std::vector<ParticleTupleIO> particle_data;


			// file particle data
			particle_data.push_back( ParticleTupleIO(x,y,z,n_particles++,typeMap[type]) );

			ldbg << "min position xyz file : " << min_x << " " << min_y << " " << min_z << std::endl;
			ldbg << "max position xyz file : " << max_x << " " << max_y << " " << max_z << std::endl;

			//DOMAIN
			AABB computed_bounds = { {min_x, min_y, min_z} , {max_x, max_y, max_z} };
			ldbg << "computed_bounds  = " << computed_bounds << std::endl;

			//domain->m_bounds = bounds;
			compute_domain_bounds(*domain,*bounds_mode,*enlarge_bounds,file_bounds,computed_bounds, false );

			lout << "Particles        = "<<particle_data.size()<<std::endl;
			lout << "Domain XForm     = "<<domain->xform()<<std::endl;
			lout << "Domain bounds    = "<<domain->bounds()<<std::endl;
			lout << "Domain size      = "<<bounds_size(domain->bounds()) <<std::endl;
			lout << "Real size        = "<<bounds_size(domain->bounds()) * Vec3d{domain->xform().m11,domain->xform().m22,domain->xform().m33} <<std::endl;
			lout << "Cell size        = "<<domain->cell_size()<<std::endl;
			lout << "Grid dimensions  = "<<domain->grid_dimension()<<" ("<<grid_cell_count(domain->grid_dimension())<<" cells)"<< std::endl;

			//send bounds and size_box values to all cores
			MPI_Bcast( & (*domain), sizeof(Domain), MPI_CHARACTER, 0, *mpi);
			assert( check_domain(*domain) );

			grid->set_offset( IJK{0,0,0} );
			grid->set_origin( domain->bounds().bmin );
			grid->set_cell_size( domain->cell_size() );
			grid->set_dimension( domain->grid_dimension() );

			if(rank==0)
			{
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
			}

			lout << "============================" << std::endl;

			grid->rebuild_particle_offsets();

#     ifndef NDEBUG
			bool particles_inside_cell = check_particles_inside_cell(*grid);
			assert( particles_inside_cell );
#     endif
		}

	};

	// === register factories ===
	__attribute__((constructor)) static void register_factories()
	{
		OperatorNodeFactory::instance()->register_factory("init_rsa", make_grid_variant_operator< InitRSA >);
	}

}
