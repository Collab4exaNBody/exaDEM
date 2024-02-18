#pragma once

#include<tuple>

namespace exaDEM
{
	template<typename T>
		struct ExtraDynamicStorageGridMoveBufferT;

	template <typename T>
		struct ExtraDynamicDataStorageCellMoveBufferT
		{
			using uint = uint64_t;
			using info = std::tuple<uint, uint, uint>;
			std::vector< size_t > m_indirection; 
			std::vector< char > m_data; // global info,  offset, data, packed contiguously
			inline void clear()
			{
				m_indirection.clear();
				m_data.clear();
				m_data.resize(2 * sizeof(uint)); // number of particles, number of items 
				uint* global_information = (uint*) m_data.data();
				global_information[0] = 0;
				global_information[1] = 0;
			}

			std::tuple<const uint*, const info*, const T*> decode_pointers(const unsigned int n_particles) const
			{
				const uint* glob_info_ptr = (uint*) m_data.data();
				const info*      info_ptr = (info*) (glob_info_ptr + 2);
				const T*         data_ptr = (T*) (info_ptr + n_particles);
				return {glob_info_ptr, info_ptr, data_ptr};
			}

			std::tuple<uint*, info*, T*> decode_pointers(const unsigned int n_particles) 
			{
				uint* glob_info_ptr = (uint*) m_data.data();
				info*      info_ptr = (info*) (glob_info_ptr + 2);
				T*         data_ptr = (T*) (info_ptr + n_particles);
				return {glob_info_ptr, info_ptr, data_ptr};
			}

			inline info& get_info(const unsigned int p)
			{
				constexpr unsigned int global_information_shift = 2 * sizeof(uint);
        info * __restrict__ info_ptr = (info *) (m_data.data() + global_information_shift);
        return info_ptr[m_indirection[p]];
			}

			inline size_t number_of_particles() const
			{
				//uint n_particles = ((uint*) m_data.data()) [0]; // 0 -> n_particles, 1 -> n_items
				//assert ( m_indirection.size() == ((uint*) m_data.data()) [0] );
				return m_indirection.size();
			}

			inline void swap( size_t a, size_t b )
			{
				std::swap( m_indirection[a] , m_indirection[b] );
			}

			inline uint64_t particle_id(size_t p) const
			{
				constexpr unsigned int global_information_shift = 2 * sizeof(uint);
				const info * __restrict__ info_ptr = (const info *) (m_data.data() + global_information_shift);
				auto& particle_information = info_ptr[m_indirection[p]];
				return std::get<2> (particle_information);
			}

			inline size_t particle_number_of_items(size_t p) const
			{
				constexpr unsigned int global_information_shift = 2 * sizeof(uint);
				const info * __restrict__ info_ptr = (const info *) (m_data.data() + global_information_shift);
				auto& particle_information = info_ptr[ m_indirection[p] ];
				return std::get<1> (particle_information);
			}

			inline std::pair< const T* , const T* > particle_data_range(size_t p) const
			{
				constexpr unsigned int global_information_shift = 2 * sizeof(uint);
				uint n_particles = ((uint*) m_data.data()) [0]; // 0 -> n_particles, 1 -> n_items
				const info * __restrict__ info_ptr = (const info *) (m_data.data() + global_information_shift);
				const auto& [offset, size, id] = info_ptr[ m_indirection[p] ];
				const T* data_ptr = (const T*) (m_data.data() + global_information_shift + n_particles * sizeof (info)); 
				return { data_ptr + offset, data_ptr + offset + size -1 };
			}

			inline const T* item(size_t p, size_t i) const
			{
				constexpr unsigned int global_information_shift = 2 * sizeof(uint);
				uint n_particles = ((uint*) m_data.data()) [0]; // 0 -> n_particles, 1 -> n_items
				const info * __restrict__ info_ptr = (const info *) (m_data.data() + global_information_shift);
				const auto& [offset, size, id] = info_ptr[ m_indirection[p] ];
				const T* data_ptr = (const T*) (m_data.data() + global_information_shift + n_particles * sizeof (info)); 
				return data_ptr + offset + i;
			}

			inline bool check_info_consistency()
			{
				const auto [glob_ptr, info_ptr, data_ptr] = decode_pointers(number_of_particles());
				uint n_particles = glob_ptr[0];
				return migration_test::check_info_consistency ( info_ptr, n_particles) ;
			}

			inline void copy_incoming_particle( const ExtraDynamicStorageGridMoveBufferT<T>& opt_buffer, size_t cell_i, size_t p_i); // defined after ExtraDynamicStorageGridMoveBuffer 
		};

	template<typename T>
		struct ExtraDynamicStorageGridMoveBufferT
		{
			using uint = uint64_t;
			using info = std::tuple<uint,uint,uint>;
			using CellMoveBuffer = ExtraDynamicDataStorageCellMoveBufferT<T>;
			using CellStorage    = CellExtraDynamicDataStorageT<T>;
			onika::memory::CudaMMVector< CellExtraDynamicDataStorageT<T> > & m_cell_data;
			CellMoveBuffer & m_otb_buffer; // buffer for elements going outside of grid
			std::vector<CellMoveBuffer> m_cell_buffer; // incoming buffers for elements entering another cell of the grid

			inline CellMoveBuffer* otb_buffer() { return & m_otb_buffer; }
			inline CellMoveBuffer* cell_buffer( size_t cell_i ) { return & m_cell_buffer[cell_i]; }

			inline void initialize( size_t n_cells )
			{
				m_cell_buffer.resize( n_cells );
				for(size_t i=0;i<n_cells;i++) m_cell_buffer[i].clear();
				m_otb_buffer.clear();
			}

			// must be done following the same particle re-organisation as in move_particles_across_cells.h
			inline void pack_cell_particles( size_t cell_i, const std::vector<int32_t> & packed_particles, bool removed_particles = true )
			{
				assert( cell_i < m_cell_data.size() );
				auto & cell = m_cell_data[cell_i];

				const size_t n_incoming_particles = m_cell_buffer[cell_i].number_of_particles();
				const size_t total_particles = removed_particles ? ( packed_particles.size() + n_incoming_particles ) : ( cell.number_of_particles() + n_incoming_particles );

				CellStorage pf;
				pf.initialize( total_particles );

				// 1. pack existing partciles, overwritting removed ones
				size_t p = 0;
				uint offset = 0;
				if( removed_particles )
				{
					const size_t n_particles = packed_particles.size();
					for(p=0; p<n_particles ;p++)
					{
						// update information -> (offset , number of items, particle id) 
						const uint id = cell.particle_id( packed_particles[p] );
						const size_t n_items = cell.particle_number_of_items( packed_particles[p] );
						auto& particle_information = pf.m_info[p];
						particle_information = std::make_tuple(offset, n_items, id);
						// update data
						pf.m_data.resize (offset + n_items);
						for(size_t i = 0 ; i < n_items ; i++)
						{
							pf.set_item( offset++, cell.get_particle_item( packed_particles[p] , i ) );
						}
					}
				}
				else // copy original particles as is
				{
					const size_t n_particles = cell.number_of_particles();
					for( p=0 ; p < n_particles ; p++)
					{
						// update information
						const auto id = cell.particle_id( p );
						const size_t n_items = cell.particle_number_of_items( p );
						auto& particle_information = pf.m_info[p];
						particle_information = {offset, n_items, id};
						// update data
						pf.m_data.resize (offset + n_items);
						for(size_t i = 0 ; i < n_items ; i++)
						{
							pf.set_item( offset++, cell.get_particle_item( p , i ));
						}
					}
				}

				// add new items in the futur cell extra data storage
				// offset and p are incremented in the last loop 
				for(size_t i=0 ; i<n_incoming_particles ; i++)
				{
					// update information -> (offset , number of items, particle id) 
					auto id = m_cell_buffer[cell_i].particle_id(i);
					size_t n_items = m_cell_buffer[cell_i].particle_number_of_items(i);
					auto& particle_information = pf.m_info[p];
					particle_information = {offset, n_items, id};
					// update data
					pf.m_data.resize (offset + n_items);
					for(size_t j = 0 ; j < n_items ; j++) 
					{
						pf.set_item( offset++, *(m_cell_buffer[cell_i].item(i,j)) );
					}
					++ p;
				}

				assert (pf.check_info_consistency());
				// copy ad fit new dynamic extra data in the cell
				assert( p == total_particles );
				cell = std::move( pf );
				cell.shrink_to_fit();
			}
		};

	/**
	 * @brief Copies incoming particle data from a move buffer into the cell.
	 * @tparam T The type of the extra data storage in the cell.
	 * @param opt_buffer The move buffer containing incoming particle data.
	 * @param cell_i The index of the cell to which the incoming particle data is copied.
	 * @param p_i The index of the particle within the cell.
	 */
	template<typename T>
		inline void ExtraDynamicDataStorageCellMoveBufferT<T>::copy_incoming_particle( const ExtraDynamicStorageGridMoveBufferT<T>& opt_buffer, size_t cell_i, size_t p_i)
		{
			// Note : This function is not optimal, it does one std::move for every calls.
			// get cell information
			const CellExtraDynamicDataStorageT<T>& cell = opt_buffer.m_cell_data[cell_i];
			const auto [offset, size, id] = cell.m_info[p_i];
			uint n_items_to_copy = size;
			assert ( migration_test :: check_info_consistency (cell.m_info.data(), cell.m_info.size()) );

			// reminder : buffer layout : N/nb particles, M/nb items, info1, info2 ..., infoN, item1, ..., itemM
			constexpr uint global_information_shift = 2 * sizeof(uint);
			uint* global_information_ptr = (uint*) m_data.data();
			const uint buffer_n_particles = global_information_ptr [0]; // 0 -> n_particles, 1 -> n_items
			const uint buffer_n_items     = global_information_ptr [1]; // 0 -> n_particles, 1 -> n_items

			// create shift to insert a new info
			// -> new buffer size
			const uint new_size = global_information_shift + (buffer_n_particles + 1) * sizeof(info) + (buffer_n_items + n_items_to_copy) * sizeof(T);

			// -> resize data with new size
			m_data.resize(new_size); // m_data ptr can change here

			// update pointers due to resize
			global_information_ptr = (uint*) m_data.data();
			info * info_ptr = (info*) (m_data.data() + global_information_shift);

			// -> shift item data and new info
			char* old_item_ptr = m_data.data() + global_information_shift + buffer_n_particles * sizeof(info);
			char* new_item_ptr = m_data.data() + global_information_shift + (buffer_n_particles + 1) * sizeof(info);
			uint nb_bytes_n_items = buffer_n_items * sizeof(T);

			// --> create a space for info
			std::move(old_item_ptr, old_item_ptr + nb_bytes_n_items, new_item_ptr);

			// --> add info in this the new memory space
			if( buffer_n_particles != 0 )
			{
				info& old_last = info_ptr[buffer_n_particles - 1];
				info& new_last = info_ptr[buffer_n_particles];
				new_last = {std::get<0>(old_last) + std::get<1> (old_last), n_items_to_copy, id};
			}
			else
			{
				info_ptr[0] = {0, n_items_to_copy, id};
			}

			assert ( check_info_consistency () );

			// copy items at the end
			T* item_ptr = (T*) new_item_ptr;
			std::copy(
					cell.m_data.data() + offset, // offset and size are defined at the begining 
					cell.m_data.data() + offset + size,
					item_ptr + buffer_n_items // do not forget to shift here -> buffer_n_items is the old number of items on the buffer
					);				

			// update global information
			global_information_ptr[0] += 1;
			global_information_ptr[1] += n_items_to_copy;

			// update indirection vector
			m_indirection.push_back(m_indirection.size());
			assert ( m_indirection.size() == global_information_ptr[0] );
		}
}
