#pragma once

#include <cassert>

namespace exaDEM
{
	namespace migration_test
	{
		using uint = uint64_t;
		using info = std::tuple<uint,uint,uint>;

		inline bool check_info_consistency(const info* __restrict__ info_ptr, const uint info_size)
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
	}
}
