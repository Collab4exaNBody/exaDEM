#pragma once
#include <onika/parallel/parallel_execution_context.h>
#include <onika/parallel/block_parallel_for.h>


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
      const K kernel;
      const std::tuple<Args...> params;

      WrapperForAll(T* p, K& k, Args... args) 
        : ptr(p), 
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

  template<typename... Args>
    static inline ParallelExecutionWrapper run_contact_law(ParallelExecutionContext * exec_ctx, int type, Classifier& ic, Args&&... args)
		{
			auto [ptr, size] = ic.get_info(type);
			WrapperForAll func(ptr,  args...);
		  return block_parallel_for( size, func, exec_ctx);
		}
}
