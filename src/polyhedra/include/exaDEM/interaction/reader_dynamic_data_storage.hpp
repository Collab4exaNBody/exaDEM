#pragma once

#include <exaDEM/interaction/dynamic_data_storage.hpp>

// ====================================================================================
// ================ dump reader helper for injection of extra data ====================
// ====================================================================================

namespace exaDEM
{
	template<typename ItemType>
		struct ExtraDynamicDataStorageReadHelper
		{
			using UIntType = uint64_t;
			using InfoType = std::tuple<UIntType, UIntType, UIntType>;
			std::vector< std::vector< std::vector< ItemType > > > m_out_item; /** [cell->particles->items] */
			std::vector<CellExtraDynamicDataStorageT < ItemType > > m_in_item; /** not grid cells, just chunks of item data */
			std::map< UIntType , std::pair<size_t,size_t> > m_id_map; /** makes relation between particle id and (cell, position in cell) */

			void initialize(size_t n_cells)
			{
				m_out_item.resize( n_cells );
				m_in_item.clear();
				m_id_map.clear();
			}

			inline void read_from_stream( const uint8_t* stream_start , size_t stream_size )
			{
				using UIntType = uint64_t;
				m_in_item.clear();
				m_id_map.clear();

				const uint8_t* stream = stream_start;
				constexpr int buffer_header_size = 2 * sizeof(UIntType);

				while( (stream-stream_start) < ssize_t(stream_size) )
				{
					CellExtraDynamicDataStorageT < ItemType > cell;
					cell.decode_buffer_to_cell((void*) stream );
					stream += cell.storage_size() + buffer_header_size; // shift ptr to decode the next cell
					assert( (stream-stream_start) <= ssize_t(stream_size) );
					m_in_item.push_back( cell );
				}
				assert( (stream-stream_start) == ssize_t(stream_size) );

				const size_t n_cells = m_in_item.size();
				for(size_t cell_i = 0 ; cell_i < n_cells ; cell_i++ )
				{
					const size_t n_particles = m_in_item[cell_i].number_of_particles();
					for(size_t p_i = 0 ; p_i < n_particles ; p_i++)
					{
						UIntType id = m_in_item[cell_i].particle_id( p_i );
						m_id_map[ id ] = std::pair<size_t,size_t>{ cell_i , p_i };
					}
				}
			}

			// This function is used particle_dump_filter.h
			inline void append_cell_particle( size_t cell_idx , size_t p_idx , uint64_t id )
			{
				if( m_id_map.empty() ) return;

				assert( cell_idx < m_out_item.size() );
				assert( p_idx == m_out_item[cell_idx].size() );
				auto & new_particle_item = m_out_item[cell_idx].emplace_back();

				auto it = m_id_map.find(id);
				if( it != m_id_map.end() )
				{
					auto [ c , p ] = it->second;
					auto [ offfset, n_items, item_id] = m_in_item[c].m_info[p];
					assert( item_id == id );
					new_particle_item.insert(new_particle_item.end(), &m_in_item[c].m_data[offfset], &m_in_item[c].m_data[offfset + n_items]);	
				}
			}

			template< class ParticleIdFuncT>
				inline void finalize( GridExtraDynamicDataStorageT<ItemType> & grid_item , ParticleIdFuncT particle_id )
				{
					grid_item.m_data.clear();
					const size_t n_cells = m_out_item.size();
					grid_item.m_data.resize( n_cells );

					// fill each cell dynamique extra storage
					for(size_t i = 0 ; i < n_cells ; i++)
					{
						const size_t n_particles = m_out_item[i].size();
						// get current cell of items
						auto& cell_item = grid_item.m_data[i];
						cell_item.initialize( n_particles );
						UIntType offset = 0;
						InfoType * __restrict__ info_ptr = cell_item.m_info.data();
						for(size_t p = 0 ; p < n_particles ; p++)
						{
							const UIntType id = particle_id( i , p );
							const UIntType n_items = m_out_item[i][p].size();
							info_ptr [p] = {offset, n_items, id};
							cell_item.m_data.insert( cell_item.m_data.end(), m_out_item[i][p].begin() , m_out_item[i][p].end());
							offset += n_items;
						}
						// Do some memory clean
						m_out_item[i].clear();
						m_out_item[i].shrink_to_fit(); // really free memory
						grid_item.m_data[i].shrink_to_fit(); // reallocate
					}
				}
		};
}
