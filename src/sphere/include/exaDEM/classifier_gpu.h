#pragma once
#include <cuda.h>

namespace exaDEM
{
  template<typename Ker, typename... Args>
    __global__ void for_all(exaDEM::Interaction* ptr, size_t size, Ker kernel, Args... args)
    {
      int idx = blockIdx.x*blockDim.x+threadIdx.x;
      if (idx < size)
      {
	      exaDEM::Interaction& item = ptr[idx];
	      kernel(item, std::forward<Args>(args)...);
      }
    }

  template<typename... Args>
    inline void run_contact_law(int id, Classifier& ic, Args&&... args)
    {
      auto [ptr, size] = ic.get_info(id);
      if(size != 0)
      {
        const int blockSize = 128;
        const int gridSize = (int)ceil((float)size/blockSize);
        for_all<<<gridSize, blockSize>>>(ptr, size, args...);
      }
    }
}
