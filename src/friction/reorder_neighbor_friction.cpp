#include <exanb/core/operator.h>
#include <exanb/core/operator_slot.h>
#include <exanb/core/operator_factory.h>
#include <exanb/core/make_grid_variant_operator.h>
#include <exanb/core/parallel_grid_algorithm.h>
#include <exanb/core/grid.h>

#include <exanb/particle_neighbors/chunk_neighbors.h>
#include <exanb/particle_neighbors/chunk_neighbors_apply.h>

#include <memory>

#include <exaDEM/neighbor_friction.h>

namespace exaDEM
{
  using namespace exanb;

  template<typename GridT
    , class = AssertGridHasFields< GridT, field::_radius >
    >
  class ReorderNeighborFriction : public OperatorNode
  {
    // attributes processed during computation
    using ComputeFields = FieldSet< field::_vrot, field::_arot >;
    static constexpr ComputeFields compute_field_set {};

    ADD_SLOT( GridT  , grid     , INPUT_OUTPUT , REQUIRED );
    ADD_SLOT( exanb::GridChunkNeighbors , chunk_neighbors   , INPUT , OPTIONAL , DocString{"neighbor list"} );
    ADD_SLOT( GridCellParticleNeigborFriction , nbh_friction , INPUT_OUTPUT , GridCellParticleNeigborFriction{} , DocString{"Neighbor particle friction term"} );

  public:

		inline std::string documentation() const override final
		{
			return R"EOF(
        				This operator reorders data friction by copying old data friction value in the right place.
				        )EOF";
		}

    inline void execute () override final
    {
      const auto cells = grid->cells();
      const size_t n_cells = grid->number_of_cells(); // nbh.size();
            
      const IJK dims = grid->dimension();
      const int gl = grid->ghost_layers();
    
      auto & friction = *nbh_friction;
      
      ldbg << "ReorderNeighborFriction: "<< n_cells<<" cells, chunk_neighbors.has_value()="<<chunk_neighbors.has_value() <<", grid="<<dims<<", gl="<<gl <<std::endl;

      // if grid structure (dimensions) changed, we invalidate the whole friction data
      if( friction.m_cell_friction.size() != n_cells )
      {
        ldbg << "number of cells changed, reset friction data" << std::endl;
        friction.m_cell_friction.clear();
        friction.m_cell_friction.resize( n_cells );
      }
      assert( friction.m_cell_friction.size() == n_cells );

      if( ! chunk_neighbors.has_value() ) return;

      size_t n_missing_particles = 0;
      size_t n_missing_nbh = 0;

#     pragma omp parallel
      {
        std::vector<size_t> sorted_friction_id;
        std::vector<size_t> sorted_friction_nbh_id;
            
        GRID_OMP_FOR_BEGIN(dims-2*gl,_,block_loc, schedule(dynamic) reduction(+:n_missing_particles,n_missing_nbh) )
        {
          IJK loc_a = block_loc + gl;
          size_t cell_a = grid_ijk_to_index( dims , loc_a );
          const int n_particles = cells[cell_a].size();
                    
          CellParticleNeighborFriction reordered_friction;
          reordered_friction.initialize( n_particles );

          auto & friction = nbh_friction->m_cell_friction[cell_a];
          size_t friction_n_particles = friction.number_of_particles(); // may be 0
          sorted_friction_id.clear(); // discard any value to avoid old value copy
          sorted_friction_id.resize(friction_n_particles);
          for(unsigned int i=0;i<friction_n_particles;i++) sorted_friction_id[i]=i;
          std::sort( sorted_friction_id.begin() , sorted_friction_id.end() ,
                     [&friction](const int& a, const int& b)->bool { return friction.particle_id(a) < friction.particle_id(b); }
                   );

          int last_p_a = -1;
          int cur_friction_idx = -1;
          
          apply_cell_particle_neighbors(*grid, *chunk_neighbors, cell_a, loc_a, std::false_type() /* not symetric */,
            [cells,cell_a,&last_p_a,&friction,&reordered_friction,&sorted_friction_id,&sorted_friction_nbh_id,&cur_friction_idx,&n_missing_particles,&n_missing_nbh]
            ( int p_a, size_t cell_b, unsigned int p_b , size_t p_nbh_index )
            {
              if( p_a != last_p_a )
              {
                assert( p_a>=0 && last_p_a<p_a );
                while( last_p_a < p_a ) // ensures we have a begin/end for each particle, even if it hasn't any neighbor
                {
                  if( last_p_a != -1 ) reordered_friction.end_nbh_write( last_p_a , cells[cell_a][field::id][last_p_a] );
                  ++ last_p_a;
                  reordered_friction.begin_nbh_write( last_p_a , cells[cell_a][field::id][last_p_a] );
                }
                assert( last_p_a == p_a );
                //assert( size_t(p_a) < friction.number_of_particles() );                
                auto sorted_idx_it = std::lower_bound( sorted_friction_id.begin() , sorted_friction_id.end() , cells[cell_a][field::id][p_a] ,
                                                       [&friction](const int& idx, const uint64_t& id)->bool { return friction.particle_id(idx) < id; }
                                                     );
                cur_friction_idx = -1;
                if( sorted_idx_it != sorted_friction_id.end() )
                {
                  if( friction.particle_id(*sorted_idx_it) == cells[cell_a][field::id][p_a] )
                  {
                    cur_friction_idx = *sorted_idx_it;
                    // sort neighbor ids to find them quickly
                    const int n_pairs = friction.particle_number_of_pairs(cur_friction_idx);
                    sorted_friction_nbh_id.clear();
                    sorted_friction_nbh_id.resize(n_pairs);
                    for(int i=0;i<n_pairs;i++) sorted_friction_nbh_id[i]=i;
                    std::sort( sorted_friction_nbh_id.begin() , sorted_friction_nbh_id.end() ,
                              [cur_friction_idx,&friction](const int& a, const int& b)->bool{ return friction.pair_friction(cur_friction_idx,a).m_particle_id < friction.pair_friction(cur_friction_idx,b).m_particle_id ; }
                              );
                  }
                }
                if( cur_friction_idx == -1 ) ++ n_missing_particles;
              }
            
              auto id_b = cells[cell_b][field::id][p_b];
              ParticlePairFriction pair_friction = { id_b , { 0. , 0. , 0. } };
              if( cur_friction_idx != -1 )
              {
                assert( cur_friction_idx>=0 && size_t(cur_friction_idx)<friction.number_of_particles() );
                auto sorted_nbh_it = std::lower_bound( sorted_friction_nbh_id.begin() , sorted_friction_nbh_id.end() , id_b ,
                                                        [cur_friction_idx,&friction](const int& idx, const uint64_t& id)->bool { return friction.pair_friction(cur_friction_idx,idx).m_particle_id < id; }
                                                     );
                bool nbh_found = false;
                if( sorted_nbh_it != sorted_friction_nbh_id.end() )
                {
                  const auto& pf = friction.pair_friction(cur_friction_idx,*sorted_nbh_it);
                  if( pf.m_particle_id == id_b )
                  {
                    nbh_found = true;
                    pair_friction = pf;
                  }
                }
                if( ! nbh_found ) ++ n_missing_nbh;
              }
              reordered_friction.push_back( pair_friction );
            });

          // ensures correct encoding for every particles, even those without any neighbors
          while( last_p_a < n_particles ) // ensures we have a begin/end for each particle, even if it hasn't any neighbor
          {
            if( last_p_a != -1 ) reordered_friction.end_nbh_write( last_p_a , cells[cell_a][field::id][last_p_a] );
            ++ last_p_a;
            if( last_p_a < n_particles ) reordered_friction.begin_nbh_write( last_p_a , cells[cell_a][field::id][last_p_a] );
          }

          assert( reordered_friction.check_consistency() );
          friction = reordered_friction; // not a move, a copy, we keep the temporary reordered_friction allocated for next step
        }
        GRID_OMP_FOR_END
      }
      ldbg << "ReorderNeighborFriction: end : n_missing_particles="<<n_missing_particles<<" , n_missing_nbh="<<n_missing_nbh <<std::endl;
    }
  };
  
  template<class GridT> using ReorderNeighborFrictionTmpl = ReorderNeighborFriction<GridT>;
  
 // === register factories ===  
  CONSTRUCTOR_FUNCTION
  {
   OperatorNodeFactory::instance()->register_factory( "reorder_neighbor_friction", make_grid_variant_operator< ReorderNeighborFrictionTmpl > );
  }

}

