#pragma once
#include <onika/parallel/parallel_execution_context.h>
#include <onika/parallel/block_parallel_for.h>
#include <onika/parallel/parallel_for.h>


namespace exaDEM
{
  using namespace onika::parallel;

  /**
   * @brief Namespace for utilities related to tuple manipulation.
   */
  namespace tuple_helper
  {
    template <size_t... Is>
      struct index {};

    template <size_t N, size_t... Is>
      struct gen_seq : gen_seq<N - 1, N - 1, Is...> {};

    template <size_t... Is>
      struct gen_seq<0, Is...> : index<Is...> {};
  }

  struct AnalysisDataPackerNull
  {
    template<typename... Args>
      ONIKA_HOST_DEVICE_FUNC inline void operator() (Args&&...  args) const { /* do nothing */ }
  };

  struct AnalysisDataPacker
  {
    Vec3d* cop;
    Vec3d* fnp;
    Vec3d* ftp;

    AnalysisDataPacker(Classifier& ic, int type)
    {
      auto [_cop, _fnp, _ftp] = ic.buffer_p(type);
      cop = _cop;
      fnp = _fnp;
      ftp = _ftp;
    }

    ONIKA_HOST_DEVICE_FUNC inline void operator() (const uint64_t idx, const Vec3d& contact, const Vec3d& fn, const Vec3d& ft) const
    { 
      cop[idx] = contact;
      fnp[idx] = fn;
      ftp[idx] = ft;
    }
  };

  /**
   * @brief Wrapper for applying a kernel function to elements of an array in parallel.
   *
   * This struct provides a mechanism to apply a kernel function to each element of
   * an array in parallel. It stores a pointer to the array (`ptr`),
   * the kernel function (`kernel`), and a tuple of parameters (`params`) to be passed
   * to the kernel function along with each element.
   *
   * @tparam T Type of elements in the array.
   * @tparam K Type of the kernel function.
   * @tparam AnalysisDataPacker Used to pack any kind of data.
   * @tparam Args Types of additional parameters passed to the kernel function.
   */
  template<typename K, typename AnalysisDataPacker, typename... Args>
    struct WrapperForAll
    {
      InteractionWrapper data;    /**< Wrapper that contains a pointer to the array of elements. */
      const K kernel;             /**< Kernel function to be applied. */
      AnalysisDataPacker packer;  /**< Kernel function to be applied. */
      std::tuple<Args...> params; /**< Tuple of parameters to be passed to the kernel function. */

      /**
       * @brief Constructor to initialize the WrapperForAll struct.
       *
       * @param d Wrapper that contains the array of elements.
       * @param k Kernel function to be applied.
       * @param args Additional parameters passed to the kernel function.
       */
      WrapperForAll(InteractionWrapper& d, K& k, AnalysisDataPacker& p,  Args... args) 
        : data(std::move(d)),
        kernel(k),
        packer(p), 
        params(std::tuple<Args...>(args...)) 
      {} 


      /**
       * @brief Helper function to apply the kernel function to a single element.
       *
       * @tparam Is Index sequence for unpacking the parameter tuple.
       * @param item Reference to the element from the array.
       * @param indexes Index sequence to unpack the parameter tuple.
       */
      template <size_t... Is>
        ONIKA_HOST_DEVICE_FUNC inline void apply(uint64_t i, tuple_helper::index<Is...> indexes) const
        {
          exaDEM::Interaction& item = data(i);
          const auto [pos, fn, ft] = kernel(item, std::get<Is>(params)...); 
          packer(i, pos, fn, ft); // packer is used to store interaction data 
        }


      /**
       * @brief Functor operator to apply the kernel function to each element in the array.
       *
       * @param i Index of the element in the array.
       */
      ONIKA_HOST_DEVICE_FUNC inline void operator()(uint64_t i) const 
      {
        apply(i, tuple_helper::gen_seq<sizeof...(Args)>{});
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
    static inline ParallelExecutionWrapper run_contact_law(ParallelExecutionContext * exec_ctx, int type, Classifier& ic, Kernel& kernel, bool dataPacker, Args&&... args)
    {
      ParallelForOptions opts;
      opts.omp_scheduling = OMP_SCHED_STATIC;
      auto [ptr, size] = ic.get_info(type);
      InteractionWrapper interactions = {type, ptr};
      if( !dataPacker )
      {
        AnalysisDataPackerNull nop;     
        WrapperForAll func(interactions, kernel, nop, args...);
        return parallel_for( size, func, exec_ctx, opts);
      }
      else
      {
        AnalysisDataPacker packer(ic, type);
        WrapperForAll func(interactions, kernel, packer, args...);
        return parallel_for( size, func, exec_ctx, opts);
      }
    }
}
