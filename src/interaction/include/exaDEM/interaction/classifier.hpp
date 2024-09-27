#pragma once

#include <exaDEM/interaction/interaction.hpp>
#include <exaDEM/interaction/interactionAOS.hpp>
#include <exaDEM/interaction/interactionSOA.hpp>
#include <exaDEM/interaction/grid_cell_interaction.hpp>
#include <exaDEM/itools/buffer.hpp>

namespace exaDEM
{

  template<typename GridT>
    inline bool filter_duplicates(const GridT& G, const exaDEM::Interaction& I)
    {
      if(I.type < 4) // polyhedron - polyhedron or sphere - sphere
      {
        if(G.is_ghost_cell(I.cell_j) && I.id_i > I.id_j)
        {
          return false;
        }
      }
      return true;
    }

  //template< typename T >
  struct InteractionWrapper
  {
  
    bool aos;
  	
    exaDEM::Interaction* interactions;
    
    double* ft_x;
    double* ft_y;
    double* ft_z;
    
    double* mom_x;
    double* mom_y;
    double* mom_z;
    
    uint64_t* id_i;
    uint64_t* id_j;
    
    uint32_t* cell_i;
    uint32_t* cell_j;
    
    uint16_t* p_i;
    uint16_t* p_j;
    
    uint16_t* sub_i;
    uint16_t* sub_j;
    
    uint16_t m_type;
 
    void initialize( InteractionAOS& data )
    {
    	aos = true;
    	
    	interactions = onika::cuda::vector_data(data.interactions); 
    }
    
    void initialize( InteractionSOA& data )	
    {
    	aos = false;
    	
    	ft_x = onika::cuda::vector_data(data.ft_x);
    	ft_y = onika::cuda::vector_data(data.ft_y);
    	ft_z = onika::cuda::vector_data(data.ft_z);
    	
    	mom_x = onika::cuda::vector_data(data.mom_x);
    	mom_y = onika::cuda::vector_data(data.mom_y);
    	mom_z = onika::cuda::vector_data(data.mom_z);
    	
    	id_i = onika::cuda::vector_data(data.id_i);
    	id_j = onika::cuda::vector_data(data.id_j);
    	
    	cell_i = onika::cuda::vector_data(data.cell_i);
    	cell_j = onika::cuda::vector_data(data.cell_j);
    	
    	p_i = onika::cuda::vector_data(data.p_i);
    	p_j = onika::cuda::vector_data(data.p_j);
    	
    	sub_i = onika::cuda::vector_data(data.sub_i);
    	sub_j = onika::cuda::vector_data(data.sub_j);
    }
    
    ONIKA_HOST_DEVICE_FUNC inline exaDEM::Interaction operator()(const uint64_t idx) const
    {
    	exaDEM::Interaction res;
    	
    	if(aos)
    	{
    		res = interactions[idx];
    	}
    	else
    	{
    		res = { {ft_x[idx], ft_y[idx], ft_z[idx]}, {mom_x[idx], mom_y[idx], mom_z[idx]}, id_i[idx], id_j[idx], cell_i[idx], cell_j[idx], p_i[idx], p_j[idx], sub_i[idx], sub_j[idx], m_type };
    	}
    	
    	return res;
    }
    
    ONIKA_HOST_DEVICE_FUNC void update( const uint64_t idx, exaDEM::Interaction item ) const
    {
    	if(aos)
    	{
    		auto& item2 = interactions[idx];
    		item2.update_friction_and_moment(item);
    	}
    	else
    	{
    		ft_x[idx] = item.friction.x;
    		ft_y[idx] = item.friction.y;
    		ft_z[idx] = item.friction.z;
    		
    		mom_x[idx] = item.moment.x;
    		mom_y[idx] = item.moment.y;
    		mom_z[idx] = item.moment.z;
    	}
    }
    
  };


  /**
   * @brief Classifier for managing interactions categorized into different types.
   *
   * The Classifier struct manages interactions categorized into different types (up to 13 types).
   * It provides functionalities to store interactions in CUDA memory-managed vectors (`VectorT`).
   */
  template< typename T >
  struct Classifier
  {
    static constexpr int types = 13;
    std::vector< T > waves; ///< Storage for interactions categorized by type.
    std::vector< itools::interaction_buffers > buffers; ///< Storage for analysis. Empty if there is no analysis

    /**
     * @brief Default constructor.
     *
     * Initializes the waves vector to hold interactions for each type.
     */
    Classifier() { waves.resize(types); buffers.resize(types) ; }

    /**
     * @brief Initializes the waves vector to hold interactions for each type.
     */
    void initialize() { waves.resize(types); buffers.resize(types) ; }

    /**
     * @brief Clears all stored interactions in the waves vector.
     */
    void reset_waves()
    {
      for(auto& wave : waves)
      {
        wave.clear();
      }
    }

    /**
     * @brief Retrieves the CUDA memory-managed vector of interactions for a specific type.
     *
     * @param id Type identifier for the interaction wave.
     * @return Reference to the CUDA memory-managed vector storing interactions of the specified type.
     */
    T& get_wave(size_t id) {return waves[id];}
    const T get_wave(size_t id) const {return waves[id];}

    /**
     * @brief Retrieves the pointer and size of the data stored in the CUDA memory-managed vector for a specific type.
     *
     * @param id Type identifier for the interaction wave.
     * @return Pair containing the pointer to the interaction data and the size of the data.
     */

    std::pair<T&, size_t> get_info(size_t id)
    {
    	const unsigned int data_size = waves[id].size();
    	T& data_ptr = waves[id];
    	return {data_ptr, data_size};
    }
    
    const std::pair<const T, const size_t> get_info(size_t id) const
    {
    	const unsigned int data_size = waves[id].size();
    	const T data_ptr = waves[id];
    	return {data_ptr, data_size};
    }

		std::tuple<double*, Vec3d*,Vec3d*,Vec3d*> buffer_p(int id)
		{
			auto& analysis = buffers[id]; 
			// fit size if needed
			const size_t size = waves[id].size();
			analysis.resize(size);
			double* const dnp = onika::cuda::vector_data( analysis.dn ); 
			Vec3d*  const cpp = onika::cuda::vector_data( analysis.cp ); 
			Vec3d*  const fnp = onika::cuda::vector_data( analysis.fn ); 
			Vec3d*  const ftp = onika::cuda::vector_data( analysis.ft );
			return {dnp, cpp, fnp, ftp}; 
		}

		/**
		 * @brief Returns the number of interaction types managed by the classifier.
		 *
		 * @return Number of interaction types.
		 */
		size_t number_of_waves() { assert(types == waves.size()) ; return types;}
		size_t number_of_waves() const { assert(types == waves.size()) ; return types;}

		/**
		 * @brief Classifies interactions into categorized waves based on their types.
		 *
		 * This function categorizes interactions into different waves based on their types,
		 * utilizing the `waves` vector in the `Classifier` struct. It resets existing waves,
		 * calculates the number of interactions per wave, and then stores interactions
		 * accordingly.
		 *
		 * @param ges Reference to the GridCellParticleInteraction object containing interactions to classify.
		 */
		void classify(GridCellParticleInteraction& ges, size_t* idxs, size_t size)
		{
			reset_waves(); // Clear existing waves
			auto& ces = ges.m_data; // Reference to cells containing interactions

#pragma omp parallel
			{
				std::array<std::vector<exaDEM::Interaction>,types> tmp; ///< Storage for interactions categorized by type.
#pragma omp for schedule(dynamic) nowait
				for(size_t c = 0 ; c < size ; c++)
				{
					auto& interactions = ces[idxs[c]];
					const unsigned int  n_interactions_in_cell = interactions.m_data.size();
					exaDEM::Interaction* const __restrict__ data_ptr = onika::cuda::vector_data( interactions.m_data );
					// Place interactions into their respective waves  
					for( size_t it = 0; it < n_interactions_in_cell ; it++ )
					{
						Interaction& item = data_ptr[it];
						const int t = item.type;
						tmp[t].push_back(item);
					}
				}

				for(int w = 0 ; w < types ; w++ )
				{
#pragma omp critical
					{
						waves[w].insert(tmp[w]);
					}
				}
			}
		}

		/**
		 * @brief Restores friction and moment data for interactions from categorized waves to cell interactions.
		 *
		 * This function restores friction and moment data from categorized waves back to their corresponding
		 * interactions in cell data (`ges.m_data`). It iterates through each wave, retrieves interactions
		 * with non-zero friction and moment from the wave, and updates the corresponding interaction in the
		 * cell data.
		 *
		 * @param ges Reference to the GridCellParticleInteraction object containing cell interactions.
		 */
		void unclassify(GridCellParticleInteraction& ges)
		{
			Vec3d null = {0,0,0};
			auto& ces = ges.m_data; // Reference to cells containing interactions
															// Iterate through each wave
			for(int w = 0 ; w < types ; w++)
			{
				auto& wave = waves[w];
				const unsigned int n1 = wave.size();
				// Parallel loop to process interactions within a wave
#pragma omp parallel for
				for(size_t it = 0 ; it < n1 ; it++) 
				{
					exaDEM::Interaction item1 = wave[it];
					// Check if interaction in wave has non-zero friction and moment
					if( item1.friction != null || item1.moment != null)
					{ 
						auto& cell = ces[item1.cell_i];
						const unsigned int  n2 = onika::cuda::vector_size( cell.m_data );
						exaDEM::Interaction* data_ptr = onika::cuda::vector_data( cell.m_data );
						// Iterate through interactions in cell to find matching interaction
						for(size_t it2 = 0; it2 < n2 ; it2++)
						{
							exaDEM::Interaction& item2 = data_ptr[it2];
							if(item1 == item2)
							{
								item2.update_friction_and_moment(item1);
								break;
							}
						}
					}
				}
			}
			reset_waves();
		}
	};
}
