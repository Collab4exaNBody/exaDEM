#pragma once
#include <onika/parallel/parallel_execution_context.h>
#include <onika/parallel/block_parallel_for.h>


namespace exaDEM
{
	using namespace onika::parallel;
/*
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
*/
	namespace tuple_helper
	{
		template <int... Is>
			struct index {};

		template <int N, int... Is>
			struct gen_seq : gen_seq<N - 1, N - 1, Is...> {};

		template <int... Is>
			struct gen_seq<0, Is...> : index<Is...> {};
	}


	template<typename T, typename K, typename... Args>
		struct WrapperForAll
		{
			T* const ptr;
			const size_t size;
			const K kernel;
			const std::tuple<Args...> params;

			WrapperForAll(T* p, size_t s, K& k, Args... args) 
				: ptr(p), 
				size(s), 
				kernel(k), 
				params(std::tuple<Args...>(args...)) 
			{} 

			template <int... Is>
				ONIKA_HOST_DEVICE_FUNC inline void apply(T& item, tuple_helper::index<Is...> indexes) const
				{
					kernel(item, std::get<Is>(params)...);
				}

			ONIKA_HOST_DEVICE_FUNC inline void operator()(uint64_t i) const 
			{
				if(i < size)
				{
					T& item = ptr[i];
					apply(item, tuple_helper::gen_seq<sizeof...(Args)>{});
				}
			}
		};

	template<typename... Args>
		inline void run_contact_law(ParallelExecutionContext * exec_ctx, int id, Classifier& ic, Args... args)
		{
			auto [ptr, size] = ic.get_info(id);
			if(size != 0)
			{
				//WrapperForAll<exaDEM::Interaction, Args...> func(ptr, size,  args...);
				WrapperForAll func(ptr, size,  args...);
				block_parallel_for( size, func, exec_ctx );
			}
		}

/*
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
*/
}
