#pragma once

#include <cassert>



namespace exaDEM
{
	namespace migration_test
	{
		using UIntType = uint64_t;
		using InfoType = std::tuple<UIntType,UIntType,UIntType>;

		inline bool check_info_consistency(const InfoType* __restrict__ info_ptr, const UIntType info_size)
		{
			for (size_t p = 0 ; p < info_size ; p++)
			{
				auto [offset, size, id] = info_ptr[p];
				if(p == 0)
				{
					if(offset != 0) return false;
				}
				else
				{
					auto [last_offset, last_size, last_id] = info_ptr[p-1];
					if(offset != last_offset + last_size) return false;
				}
			}
			return true;
		}

		inline bool check_info_doublon(const InfoType* __restrict__ info_ptr, const UIntType info_size)
		{
			for (size_t p1 = 0 ; p1 < info_size ; p1++)
			{
				for (size_t p2 = p1 + 1 ; p2 < info_size ; p2++)
				{
					if ( info_ptr[p1] == info_ptr[p2] )
					{
						return false;
					}
				}
			}
			return true;
		}
	}


	namespace interaction_test
	{
		using UIntType = uint64_t;
		using InfoType = std::tuple<UIntType,UIntType,UIntType>;
		inline bool check_extra_interaction_storage_consistency(int n_particles, InfoType* info_ptr, Interaction* data_ptr)
		{
			for( int p = 0 ; p < n_particles ; p++ )
			{
				auto [offset, size, id] = info_ptr[p];
				for(size_t i = offset ; i < offset + size ; i++ )
				{
					auto& item = data_ptr[i];
					if( item.id_i != id && item.id_j != id )
					{
						std::cout << "info says particle id = " << id << " and the interaction is between the particle id " << item.id_i << " and the particle id " << item.id_j << std::endl;
						return false;
					}
				}
			}

			return true;
		}
	}
}
