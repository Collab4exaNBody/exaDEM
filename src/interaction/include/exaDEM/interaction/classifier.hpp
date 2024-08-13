
#pragma once

#include <exaDEM/interaction/interaction.hpp>
#include <exaDEM/interaction/grid_cell_interaction.hpp>

namespace exaDEM
{

  struct InteractionWrapper
  {
    const int m_type;
    exaDEM::Interaction* const m_data;
    ONIKA_HOST_DEVICE_FUNC inline exaDEM::Interaction& operator()(const uint64_t idx) const
    {
      return m_data[idx];
    }
    ONIKA_HOST_DEVICE_FUNC inline uint8_t type() { return m_type; }
    ONIKA_HOST_DEVICE_FUNC inline size_t pi      (const uint64_t idx) { return m_data[idx].p_i; }
    ONIKA_HOST_DEVICE_FUNC inline size_t pj      (const uint64_t idx) { return m_data[idx].p_j; }
    ONIKA_HOST_DEVICE_FUNC inline size_t celli   (const uint64_t idx) { return m_data[idx].cell_i; }
    ONIKA_HOST_DEVICE_FUNC inline size_t cellj   (const uint64_t idx) { return m_data[idx].cell_j; }
    ONIKA_HOST_DEVICE_FUNC inline Vec3d& moment  (const uint64_t idx) { return m_data[idx].moment; }
    ONIKA_HOST_DEVICE_FUNC inline Vec3d& friction(const uint64_t idx) { return m_data[idx].friction; }
  };


  struct analysis_buffers
  {
    template <typename T> using VectorT =  onika::memory::CudaMMVector<T>;
    VectorT<double> dn;
    VectorT<Vec3d>  cp;
    VectorT<Vec3d>  fn;
    VectorT<Vec3d>  ft;

    void resize (const int size)
    {
      if( size != 0 ) 
      {

        dn.resize(size);
        cp.resize(size);
        fn.resize(size);
        ft.resize(size);
      }
    }
  };

  /**
   * @brief Classifier for managing interactions categorized into different types.
   *
   * The Classifier struct manages interactions categorized into different types (up to 13 types).
   * It provides functionalities to store interactions in CUDA memory-managed vectors (`VectorT`).
   */
  struct Classifier
  {
    static constexpr int types = 13;
    template <typename T> using VectorT =  onika::memory::CudaMMVector<T>;
    std::vector<VectorT<exaDEM::Interaction>> waves; ///< Storage for interactions categorized by type.
    std::vector<analysis_buffers> buffers; ///< Storage for analysis. Empty if there is no analysis

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
    VectorT<exaDEM::Interaction>& get_wave(size_t id) {return waves[id];}

    /**
     * @brief Retrieves the pointer and size of the data stored in the CUDA memory-managed vector for a specific type.
     *
     * @param id Type identifier for the interaction wave.
     * @return Pair containing the pointer to the interaction data and the size of the data.
     */
    std::pair<exaDEM::Interaction*, size_t> get_info(size_t id) 
    {
      const unsigned int  data_size = onika::cuda::vector_size( waves[id]);
      exaDEM::Interaction* const data_ptr = onika::cuda::vector_data( waves[id] );
      return {data_ptr, data_size};
    }

    std::tuple<double*, Vec3d*,Vec3d*,Vec3d*> buffer_p(int id)
    {
      auto& analysis = buffers[id]; 
			// fit size if needed
      const int size = onika::cuda::vector_size( waves[id] );
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
    size_t number_of_waves() {return waves.size();}

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
						waves[w].insert(waves[w].end(), tmp[w].begin(), tmp[w].end());
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
          exaDEM::Interaction& item1 = wave[it];
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
    }
  };
}
