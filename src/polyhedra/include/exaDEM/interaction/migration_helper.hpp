#pragma once
#include <exanb/core/basic_types.h> 
#include <exaDEM/interaction/dynamic_data_storage.hpp>
#include <exaDEM/interaction/migration_buffer.hpp>

namespace exaDEM
{
	/**
	 * @brief Template struct representing a migration helper for extra dynamic data storage.
	 * This template struct serves as a migration helper for extra dynamic data storage.
	 * It contains references to the cell extra data and an outside-the-box (OTB) buffer for migrating data.
	 * @tparam T The type of extra dynamic data stored in the cell.
	 */
	template<typename T> struct ExtraDynamicDataStorageMigrationHelper
	{
		using uint = uint64_t;
		using info = std::tuple<uint, uint, uint>;

		// members
		onika::memory::CudaMMVector< CellExtraDynamicDataStorageT<T> > & m_cell_extra_data; /**< Reference to the cell extra data. */
		ExtraDynamicDataStorageCellMoveBufferT<T> & m_otb_buffer; /**< Reference to OTB buffer for migrating data. */

		inline const unsigned int cell_particles_data_size ( size_t cell_i )
		{
			return m_cell_extra_data[ cell_i ]. storage_size();
		}

		/**
		 * @brief Writes cell particles data into a buffer.
		 * Then, it calls the 'encode_cell_to_buffer' function of the corresponding cell extra data to encode the data into the buffer.
		 * @param buffer A pointer to the buffer where the cell particles data will be written.
		 * @param cell_i The index of the cell whose particles data is to be written into the buffer.
		 */
		inline void write_cell_particles_data_in_buffer(void* buffer, size_t cell_i) const
		{
			assert ( cell_i < m_cell_extra_data.size() );
			m_cell_extra_data[ cell_i ].encode_cell_to_buffer(buffer);
		}

		/**
		 * @brief Swaps particles in the outside-the-box (OTB) buffer.
		 * @param a The index of the first particle to swap.
		 * @param b The index of the second particle to swap.
		 * @see ExtraDynamicDataStorageCellMoveBufferT::swap
		 */
		inline void swap_otb_particles( size_t a, size_t b )
		{
			m_otb_buffer.swap(a, b);
		}

		/**
		 * @brief Calculates the storage size for an outside-the-box (OTB) range.
		 * This function calculates the storage size required to accommodate an outside-the-box (OTB) range of particles.
		 * @param pstart The start index of the OTB range.
		 * @param pend The end index of the OTB range.
		 * @return The storage size required for the OTB range.
		 */
		inline size_t storage_size_for_otb_range(size_t pstart, size_t pend)
		{
			assert ( pend >= pstart );  
			const size_t n_particles = pend - pstart;
			uint total_size = 0;

			// Since particles can be reordered, 
			// we have to iterate through the data to determine the item size.
			for(size_t p = pstart ; p < pend ; p++)
			{
				uint sz = m_otb_buffer.particle_number_of_items(p);
				total_size += sz * sizeof(T);
			}

			// Add global information and 'info' vector 
			total_size += n_particles * sizeof(info) + 2 * sizeof(uint);
			return total_size;
		}

		/**
		 * @brief Serializes an outside-the-box (OTB) range into a buffer.
		 * This function serializes the particles within the specified outside-the-box (OTB) range into a buffer.
		 * @param to_buff A pointer to the buffer where the serialized data will be written.
		 * @param pstart The start index of the OTB range.
		 * @param pend The end index of the OTB range.
		 * @return The number of bytes written into the buffer.
		 */
		inline size_t serialize_otb_range( void* to_buff, size_t pstart, size_t pend )
		{
			assert (m_otb_buffer.check_info_consistency() );
			const size_t n_particles = pend - pstart;
			const auto [from_glob_ptr, from_info_ptr, from_data_ptr] = m_otb_buffer.decode_pointers(n_particles);
			// Decode stream buffer pointers.
			uint* to_glob_ptr = (uint*) to_buff; // global information
			info* to_info_ptr = (info*) (to_glob_ptr + 2);
			T*    to_data_ptr = (T*) (to_info_ptr + n_particles);

			// Set global information, number of items will be updated after
			to_glob_ptr[0] = n_particles; 
			to_glob_ptr[1] = 0; // number of items

			uint total_size = n_particles * sizeof(info) + 2 * sizeof(uint); // total_size count in Word units
			// We need to set the correct offset in the buffer, it is not store in the OTB vector
			uint to_offset = 0;

			// Iterate over the OTB vector to add 'info' + extra data storage 
			for( size_t p = pstart ; p < pend ; p++)
			{
				const auto [from_offset, from_size, from_id] = m_otb_buffer.get_info(p); 
				to_glob_ptr[1] += from_size;
				total_size += from_size * sizeof(T);
				// update info	
				to_info_ptr[p] = {to_offset, from_size, from_id}; // fit offset
				// update data
				std::copy ( from_data_ptr + from_offset, from_data_ptr + from_offset + from_size, to_data_ptr + to_offset);
				to_offset += from_size;	
			}

			assert ( migration_test :: check_info_consistency( to_info_ptr, n_particles) );
			assert( total_size == storage_size_for_otb_range(pstart,pend) );
			return total_size;
		}

		/**
		 * @brief Clears the extra data associated with a specific cell.
		 * @param cell_i The index of the cell for which the data is to be cleared.
		 */
		inline void clear_cell_data(size_t cell_i)
		{
			if( ! m_cell_extra_data.empty() )
			{ 
				assert( cell_i < m_cell_extra_data.size() );
				m_cell_extra_data[cell_i].clear();
			}
		}

		// Default behavior used by the following function
		struct NullParticleIdFunc { inline constexpr uint64_t operator () (size_t) { return 0; } };

		/**
		 * @brief Appends data to the stream using specified functions for cell indexing, locking, and unlocking.
		 *
		 * @tparam CellIndexFuncT The type of the function object for cell indexing.
		 * @tparam CellLockFuncT The type of the function object for cell locking.
		 * @tparam CellUnlockFuncT The type of the function object for cell unlocking.
		 * @tparam ParticleIdFuncT The type of the function object for particle ID. Default is NullParticleIdFunc.
		 *
		 * @param datav The data stream.
		 * @param data_bytes The size of the data to be appended.
		 * @param part_seq_len The length of the particle sequence.
		 * @param cell_index_func The function object for cell indexing.
		 * @param cell_lock_func The function object for cell locking.
		 * @param cell_unlock_func The function object for cell unlocking.
		 * @param particle_id_func The function object for particle ID. Default is NullParticleIdFunc.
		 */
		template<class CellIndexFuncT, class CellLockFuncT, class CellUnlockFuncT, class ParticleIdFuncT = NullParticleIdFunc >
			inline void append_data_stream(const void* datav, size_t data_bytes, size_t part_seq_len, CellIndexFuncT cell_index_func, CellLockFuncT cell_lock_func, CellUnlockFuncT cell_unlock_func, ParticleIdFuncT particle_id_func = {} )
			{
				const char* dataw = (const char*) datav;
				assert( data_bytes % sizeof(char) == 0 ); // always true

				// decode header
				const uint* const __restrict__ buff = (const uint *) datav; 
				const uint n_particles = buff[0];

				// Defined to process out of the grid possibilities
				size_t cur_cell = std::numeric_limits<size_t>::max();
				size_t cur_cell_start_p = 0;

				size_t p=0;

				// This loop add information and update item in different cells.
				for( p = 0 ; p < n_particles ; p++)
				{
					size_t cell_index = cell_index_func(p);

					// This condition is used increment p while cur_cell_start_p is not updated
					if( cell_index != cur_cell )
					{
						// Process only Cases in the current grid
						if( cur_cell != std::numeric_limits<size_t>::max() )
						{
							assert( cur_cell_start_p < p );
							m_cell_extra_data[cur_cell].append_data_stream_range( dataw, cur_cell_start_p, p );
							cell_unlock_func( cur_cell );
						}
						cur_cell = cell_index;
						cell_lock_func( cur_cell);
						cur_cell_start_p = p;
					}
				}
				// Manage the last case
				if( cur_cell != std::numeric_limits<size_t>::max() )
				{
					//lout<<"import friction to cell #"<<cur_cell<<std::endl;
					m_cell_extra_data[cur_cell].append_data_stream_range( dataw, cur_cell_start_p, p );
					cell_unlock_func( cur_cell );
				}
			}

		/**
		 * @brief Sets the dimension of the cell extra data.
		 * @param dims The dimensions of the cell extra data, represented by an IJK grid object.
		 */
		inline void set_dimension( const exanb::IJK& dims )
		{
			m_cell_extra_data.clear();
			m_cell_extra_data.resize( dims.i * dims.j * dims.k );
		}
	};
}
