#pragma once
#include <onika/parallel/parallel_execution_context.h>
#include <onika/parallel/block_parallel_for.h>
#include <onika/parallel/parallel_for.h>


namespace exaDEM
{
	using namespace onika::parallel;

	namespace tuple_helper
	{
		template <size_t... Is>
			struct index {};

		template <size_t N, size_t... Is>
			struct gen_seq : gen_seq<N - 1, N - 1, Is...> {};

		template <size_t... Is>
			struct gen_seq<0, Is...> : index<Is...> {};
	}


	template<typename T, typename K, typename... Args>
		struct WrapperForAll
		{
			T* const ptr;
      uint64_t size;
			const K kernel;
			std::tuple<Args...> params;

			WrapperForAll(T* p, uint64_t s, K& k, Args... args) 
				: ptr(p), size(s), 
				kernel(k), 
				params(std::tuple<Args...>(args...)) 
			{} 

			template <size_t... Is>
				ONIKA_HOST_DEVICE_FUNC inline void apply(T& item, tuple_helper::index<Is...> indexes) const
				{
					kernel(item, std::get<Is>(params)...);
				}

			ONIKA_HOST_DEVICE_FUNC inline void operator()(uint64_t i) const 
			{
				T& item = ptr[i];
				apply(item, tuple_helper::gen_seq<sizeof...(Args)>{});
			}
		};
}

namespace onika
{
  namespace parallel
	{
  	template<typename T, typename K, typename... Args> struct ParallelForFunctorTraits < exaDEM::WrapperForAll <T,K,Args... > >
  	{
	  	static inline constexpr bool CudaCompatible = true;
	  };
  }
}

namespace exaDEM
{
	using namespace onika::parallel;
	template<typename Kernel, typename... Args>
		static inline ParallelExecutionWrapper run_contact_law(ParallelExecutionContext * exec_ctx, int type, Classifier& ic, Kernel& kernel, Args&&... args)
		{
			auto [ptr, size] = ic.get_info(type);
			WrapperForAll func(ptr, size, kernel, args...);
			return parallel_for( size, func, exec_ctx);
		}
}
