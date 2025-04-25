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

#pragma once

#include <exaDEM/interaction/interaction.hpp>
#include <exaDEM/classifier/interactionAOS.hpp>
#include <exaDEM/classifier/interactionSOA.hpp>
#include <exaDEM/interaction/grid_cell_interaction.hpp>
#include <exaDEM/itools/buffer.hpp>

namespace exaDEM
{
	constexpr int NumberOfInteractionTypes = 13;
	constexpr int NumberOfPolyhedronInteractionTypes = 4;
	using NumberOfInteractionPerTypes = ::onika::oarray_t<int, NumberOfInteractionTypes>;
	using NumberOfPolyhedronInteractionPerTypes = ::onika::oarray_t<int, NumberOfPolyhedronInteractionTypes>;

	enum ResizeClassifier
	{
		SPHERE,
		POLYHEDRON,
		DRIVER
	};


	template <typename GridT> inline bool filter_duplicates(const GridT &G, const exaDEM::Interaction &I)
	{
		if (I.type < 4) // polyhedron - polyhedron or sphere - sphere
		{
			if (G.is_ghost_cell(I.cell_j) && I.id_i > I.id_j)
			{
				return false;
			}
		}
		return true;
	}

	template <typename T> struct InteractionWrapper;

	template <> 
		struct InteractionWrapper<InteractionSOA>
		{
			// forces
			double * __restrict__ ft_x;
			double * __restrict__ ft_y;
			double * __restrict__ ft_z;
			// moment
			double * __restrict__ mom_x;
			double * __restrict__ mom_y;
			double * __restrict__ mom_z;
			// particle id
			uint64_t * __restrict__ id_i;
			uint64_t * __restrict__ id_j;
			// cell id
			uint32_t * __restrict__ cell_i;
			uint32_t * __restrict__ cell_j;
			// position into the cell
			uint16_t * __restrict__ p_i;
			uint16_t * __restrict__ p_j;
			// sub id
			uint16_t * __restrict__ sub_i;
			uint16_t * __restrict__ sub_j;
			uint16_t m_type;

			InteractionWrapper(InteractionSOA &data)
			{
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

				m_type = data.type;
			}

			ONIKA_HOST_DEVICE_FUNC inline exaDEM::Interaction operator()(const uint64_t idx) const
			{
				exaDEM::Interaction res = {
					{ft_x[idx], ft_y[idx], ft_z[idx]}, 
					{mom_x[idx], mom_y[idx], mom_z[idx]}, 
					id_i[idx], id_j[idx], 
					cell_i[idx], cell_j[idx], 
					p_i[idx], p_j[idx], 
					sub_i[idx], sub_j[idx], m_type};
				return res;
			}

			ONIKA_HOST_DEVICE_FUNC inline void update(const uint64_t idx, exaDEM::Interaction& item) const
			{
				ft_x[idx] = item.friction.x;
				ft_y[idx] = item.friction.y;
				ft_z[idx] = item.friction.z;

				mom_x[idx] = item.moment.x;
				mom_y[idx] = item.moment.y;
				mom_z[idx] = item.moment.z;
			}
		};

	template <> 
		struct InteractionWrapper<InteractionAOS>
		{
			exaDEM::Interaction * __restrict__ interactions;

			InteractionWrapper(InteractionAOS &data) { interactions = onika::cuda::vector_data(data.m_data); }

			ONIKA_HOST_DEVICE_FUNC inline exaDEM::Interaction operator()(const uint64_t idx) const { return interactions[idx]; }

			ONIKA_HOST_DEVICE_FUNC inline void update(const uint64_t idx, exaDEM::Interaction& item) const
			{
				auto &item2 = interactions[idx];
				item2.update_friction_and_moment(item);
			}
		};

	/**
	 * @brief Classifier for managing interactions categorized into different types.
	 *
	 * The Classifier struct manages interactions categorized into different types (up to 13 types).
	 * It provides functionalities to store interactions in CUDA memory-managed vectors (`VectorT`).
	 */
	template <typename T> struct Classifier
	{
		static constexpr int types = NumberOfInteractionTypes;
//		std::vector<T> waves;                             ///< Storage for interactions categorized by type.
		onika::memory::CudaMMVector<T> waves;             ///< Storage for interactions categorized by type.
		std::vector<itools::interaction_buffers> buffers; ///< Storage for analysis. Empty if there is no analysis

		/**
		 * @brief Default constructor.
		 *
		 * Initializes the waves vector to hold interactions for each type.
		 */
		Classifier()
		{
			waves.resize(types);
			buffers.resize(types);
		}

		/**
		 * @brief Initializes the waves vector to hold interactions for each type.
		 */
		void initialize()
		{
			waves.resize(types);
			buffers.resize(types);
		}

		/**
		 * @brief Clears all stored interactions in the waves vector.
		 */
		void reset_waves()
		{
			for (auto &wave : waves)
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
		T &get_wave(size_t id) { return waves[id]; }
		const T get_wave(size_t id) const { return waves[id]; }

		/**
		 * @brief Retrieves the pointer and size of the data stored in the CUDA memory-managed vector for a specific type.
		 *
		 * @param id Type identifier for the interaction wave.
		 * @return Pair containing the pointer to the interaction data and the size of the data.
		 */

		std::pair<T &, size_t> get_info(size_t id)
		{
			const unsigned int data_size = waves[id].size();
			T &data = waves[id];
			return {data, data_size};
		}

		const std::pair<const T &, const size_t> get_info(size_t id) const
		{
			const unsigned int data_size = waves[id].size();
			const T &data = waves[id];
			return {data, data_size};
		}

		std::tuple<double *, Vec3d *, Vec3d *, Vec3d *> buffer_p(int id)
		{
			auto &analysis = buffers[id];
			// fit size if needed
			const size_t size = waves[id].size();
			analysis.resize(size);
			double *const __restrict__ dnp = onika::cuda::vector_data(analysis.dn);
			Vec3d *const __restrict__ cpp = onika::cuda::vector_data(analysis.cp);
			Vec3d *const __restrict__ fnp = onika::cuda::vector_data(analysis.fn);
			Vec3d *const __restrict__ ftp = onika::cuda::vector_data(analysis.ft);
			return {dnp, cpp, fnp, ftp};
		}

		/**
		 * @brief Returns the number of interaction types managed by the classifier.
		 *
		 * @return Number of interaction types.
		 */
		size_t number_of_waves()
		{
			assert(types == waves.size());
			return types;
		}
		size_t number_of_waves() const
		{
			assert(types == waves.size());
			return types;
		}

		void prefetch_memory_on_gpu()
		{
			const int device_id = 0;
			onikaStream_t s;
			cudaStreamCreate(&s);
			for(int w = 0 ; w < types ; w++)
			{
				waves[w].prefetch_memory_on_gpu(device_id, s);
			}
		}

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
		void classify(GridCellParticleInteraction &ges, size_t *idxs, size_t size)
		{
			//std::array<size_t, types> previous_sizes;
			//for (int w = 0; w < types; w++) previous_sizes[w] = waves[w].size();

			//reset_waves();          // Clear existing waves
			auto &ces = ges.m_data; // Reference to cells containing interactions

      constexpr int s = 4;
      //constexpr int s = 0;

			size_t n_threads;
#     pragma omp parallel
			{
				n_threads = omp_get_num_threads();
			}

			std::vector< std::array<std::pair<size_t,size_t>, types> > bounds;
			bounds.resize(n_threads);

#     pragma omp parallel
			{
				size_t threads = omp_get_thread_num();
				std::array<std::vector<exaDEM::Interaction>, types> tmp; ///< Storage for interactions categorized by type.
																																 //for (int w = 0; w < types; w++) tmp[w].reserve( (2*previous_sizes[w]) / n_threads);

																																 // Partial
#       pragma omp for schedule(static) nowait
				for (size_t c = 0; c < size; c++)
				{
					auto &interactions = ces[idxs[c]];
					const unsigned int n_interactions_in_cell = interactions.m_data.size();
					exaDEM::Interaction *const __restrict__ data_ptr = interactions.m_data.data();
					// Place interactions into their respective waves
					for (size_t it = 0; it < n_interactions_in_cell; it++)
					{
						Interaction &item = data_ptr[it];
						const int t = item.type;
						tmp[t].push_back(item);
					}
				}

				for (int w = 0; w < types; w++) bounds[threads][w].second = tmp[w].size();

#pragma omp barrier   

				// All
				auto& bound = bounds[threads];
				for (int w = s; w < types; w++) 
				{
					size_t start = 0;
					for ( size_t i = 0 ; i < threads ; i++)
					{
						start += bounds[i][w].second;
					}
					bound[w].first = start;
				}

#pragma omp barrier

				// Partial
#pragma omp for
				//for (int w = 0; w < types; w++)
				for (int w = s; w < types; w++) // skip polyhedron
				{
					size_t size = bounds[n_threads-1][w].first + bounds[n_threads-1][w].second;
					waves[w].resize(size);
				}

#pragma omp barrier

				// All
				//for (int w = 0; w < types; w++)
				for (int w = s; w < types; w++) // skip polyhedron
				{
					waves[w].copy(bound[w].first, bound[w].second, tmp[w], w);
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
		void unclassify(GridCellParticleInteraction &ges)
		{
			Vec3d null = {0, 0, 0};
			auto &ces = ges.m_data; // Reference to cells containing interactions
															// Iterate through each wave
			/*
				 std::array<onikaStream_t, types> streams;
				 for (int w = 0; w < types; w++) 
				 {
				 int device_id = 0;
				 ONIKA_CU_CREATE_STREAM_NON_BLOCKING(streams[w]);
				 waves[w].prefetch_memory_on_gpu(cudaCpuDeviceId, streams[w]);
				 }
			 */

#     pragma omp parallel
			{
				for (int w = 0; w < types; w++)
				{
					auto &wave = waves[w];
					const unsigned int n1 = wave.size();
					//        ONIKA_CU_STREAM_SYNCHRONIZE(streams[w]);

					// Parallel loop to process interactions within a wave
#         pragma omp for schedule(guided) nowait
					for (size_t it = 0; it < n1; it++)
					{
						exaDEM::Interaction item1 = wave[it];
						// Check if interaction in wave has non-zero friction and moment
						if (item1.friction != null || item1.moment != null)
						{
							auto &cell = ces[item1.cell_i];
							const unsigned int n2 = onika::cuda::vector_size(cell.m_data);
							exaDEM::Interaction * __restrict__ data_ptr = onika::cuda::vector_data(cell.m_data);
							// Iterate through interactions in cell to find matching interaction
							for (size_t it2 = 0; it2 < n2; it2++)
							{
								exaDEM::Interaction &item2 = data_ptr[it2];
								if (item1 == item2)
								{
									item2.update_friction_and_moment(item1);
									break;
								}
							}
						}
					}
				}
			}
			//reset_waves(); keep the memory alive
		}

    template<typename NbOfIntPerTypes>
		void resize(int start_t, int end_t, const NbOfIntPerTypes& sizes)
		{
			assert(start_t < NumberOfInteractionTypes);
			assert(end_t < NumberOfInteractionTypes);
			for(int type = start_t; type <= end_t; type++)
			{
				waves[type].resize(sizes[type]); 
			}
		}

    template<typename NbOfIntPerTypes>
		void resize(const NbOfIntPerTypes& types, ResizeClassifier resize_type)
		{
			switch (resize_type)
			{
				case SPHERE:
					resize(0, 0, types);
					break;
				case POLYHEDRON:
					resize(0, 3, types);
					break;
				case DRIVER: 
					resize(4, 12, types);
					break;
			}
		}
	};
} // namespace exaDEM
