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
    template <size_t... Is> struct index
    {
    };

    template <size_t N, size_t... Is> struct gen_seq : gen_seq<N - 1, N - 1, Is...>
    {
    };

    template <size_t... Is> struct gen_seq<0, Is...> : index<Is...>
    {
    };
  } // namespace tuple_helper

  struct AnalysisDataPackerNull
  {
    template <typename... Args> ONIKA_HOST_DEVICE_FUNC inline void operator()(Args &&...args) const
    { /* do nothing */
    }
  };

  /**
   * @struct AnalysisDataPacker
   * @brief Packs analysis data for interactions, including overlap (dn), contact point, and forces.
   *
   * This structure stores interaction-related data such as overlap (dn), contact point,
   * normal force (fn), and tangential force (ft) into their respective buffers.
   */
  struct AnalysisDataPacker
  {
    double *dnp; /**< Pointer to the buffer storing overlap (dn) values between particles. */
    Vec3d *cpp;  /**< Pointer to the buffer storing contact point positions. */
    Vec3d *fnp;  /**< Pointer to the buffer storing normal force vectors (fn). */
    Vec3d *ftp;  /**< Pointer to the buffer storing tangential force vectors (ft). */

    /**
     * @brief Constructor that initializes the data packer with buffers from a classifier.
     *
     * @param ic The classifier used to access the buffers for a given interaction type.
     * @param type The interaction type identifier to retrieve the appropriate buffers.
     *
     * The constructor retrieves the buffers for overlap (dn), contact points, normal forces,
     * and tangential forces from the classifier based on the interaction type.
     */
    template <typename T> AnalysisDataPacker(Classifier<T> &ic, int type)
    {
      auto [_dnp, _cpp, _fnp, _ftp] = ic.buffer_p(type);
      dnp = _dnp;
      cpp = _cpp;
      fnp = _fnp;
      ftp = _ftp;
    }

    /**
     * @brief Stores the interaction data at a specific index in the corresponding buffers.
     *
     * @param idx The index of the interaction.
     * @param dn The overlap between interacting particles.
     * @param contact The contact point vector.
     * @param fn The normal force vector at the contact point.
     * @param ft The tangential force vector at the contact point.
     *
     * This operator writes the given interaction data to the respective buffers
     * for overlap, contact point, normal force, and tangential force.
     */
    ONIKA_HOST_DEVICE_FUNC inline void operator()(const uint64_t idx, const double dn, const Vec3d &contact, const Vec3d &fn, const Vec3d &ft) const
    {
      dnp[idx] = dn;
      cpp[idx] = contact;
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
  template <typename T, typename K, typename AnalysisDataPacker, typename... Args> struct WrapperForAll
  {
    InteractionWrapper<T> data;
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
    WrapperForAll(InteractionWrapper<T> &d, K &k, AnalysisDataPacker &p, Args... args) : data(std::move(d)), kernel(k), packer(p), params(std::tuple<Args...>(args...)) {}

    /**
     * @brief Helper function to apply the kernel function to a single element.
     *
     * @tparam Is Index sequence for unpacking the parameter tuple.
     * @param item Reference to the element from the array.
     * @param indexes Index sequence to unpack the parameter tuple.
     */
    template <size_t... Is> ONIKA_HOST_DEVICE_FUNC inline void apply(uint64_t i, tuple_helper::index<Is...> indexes) const
    {
      exaDEM::Interaction item = data(i);
      const auto [dn, pos, fn, ft] = kernel(std::forward<exaDEM::Interaction>(item), std::get<Is>(params)...);
      data.update(i, item);
      packer(i, dn, pos, fn, ft); // packer is used to store interaction data
    }

    /**
     * @brief Functor operator to apply the kernel function to each element in the array.
     *
     * @param i Index of the element in the array.
     */
    ONIKA_HOST_DEVICE_FUNC inline void operator()(uint64_t i) const { apply(i, tuple_helper::gen_seq<sizeof...(Args)>{}); }
  };
} // namespace exaDEM

namespace onika
{
  namespace parallel
  {
    template <typename T, typename K, typename A, typename... Args> struct ParallelForFunctorTraits<exaDEM::WrapperForAll<T, K, A, Args...>>
    {
      static inline constexpr bool CudaCompatible = true;
    };
  } // namespace parallel
} // namespace onika

namespace exaDEM
{
  using namespace onika::parallel;

  /**
   * @brief Executes the contact law in parallel for a given interaction type.
   *
   * @tparam Kernel The type of the kernel functor that performs operations for the contact law.
   * @tparam Args Variadic template arguments for additional parameters.
   * @param exec_ctx The parallel execution context used to run the kernel.
   * @param type The interaction type identifier.
   * @param ic The classifier that manages interaction classification and provides data pointers.
   * @param kernel The kernel functor that defines the contact law to be applied.
   * @param dataPacker A boolean flag that determines whether to use a data packer or a null packer.
   * @param args Additional arguments forwarded to the kernel.
   *
   * @return A `ParallelExecutionWrapper` that represents the parallel execution of the contact law.
   *
   * The function retrieves interaction data based on the interaction type and runs the kernel
   * in parallel using the specified execution context. Depending on the `dataPacker` flag, it either
   * applies a null data packer or uses a data packer to process interaction data.
   */
  template <typename Kernel, typename T, typename... Args> static inline ParallelExecutionWrapper run_contact_law(ParallelExecutionContext *exec_ctx, int type, Classifier<T> &ic, Kernel &kernel, bool dataPacker, Args &&...args)
  {
    ParallelForOptions opts;
    opts.omp_scheduling = OMP_SCHED_STATIC;
    auto [data, size] = ic.get_info(type);
    InteractionWrapper<T> interactions(data);
    if (!dataPacker)
    {
      AnalysisDataPackerNull nop;
      WrapperForAll func(interactions, kernel, nop, args...);
      return parallel_for(size, func, exec_ctx, opts);
    }
    else
    {
      AnalysisDataPacker packer(ic, type);
      WrapperForAll func(interactions, kernel, packer, args...);
      return parallel_for(size, func, exec_ctx, opts);
    }
  }
} // namespace exaDEM
