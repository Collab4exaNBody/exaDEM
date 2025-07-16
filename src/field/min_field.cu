/*
   Licensed to the Apache Software Foundation (ASF) under one
   or more contributor license agreements.  See the NOTICE file
   distributed with this work for additional information
   regarding copyright ownership.  The ASF licenses this file
   to you under the Apache License, Version 2.0 (the
   "License"); you may not use this file except in compliance
   with the License.  You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.  See the License for the
specific language governing permissions and limitations
under the License.
 */

#include <onika/scg/operator.h>
#include <onika/scg/operator_slot.h>
#include <onika/scg/operator_factory.h>
#include <onika/mpi/data_types.h>
#include <exanb/core/make_grid_variant_operator.h>
#include <exanb/core/parallel_grid_algorithm.h>
#include <exanb/compute/reduce_cell_particles.h>
#include <exanb/core/grid.h>
#include <exaDEM/traversal.h>
#include <limits>


namespace exaDEM
{
  /** It does not work for quaternion or Vec3d */
  using namespace exanb;
  template<typename T>
    struct ReduceMinFieldSet
    {
      ONIKA_HOST_DEVICE_FUNC inline void operator()(T &local, T field, reduce_thread_local_t = {}) const
      {
        local = field;
      }

      ONIKA_HOST_DEVICE_FUNC inline void operator()(T &global, const T& local, reduce_thread_block_t) const
      {
        ONIKA_CU_ATOMIC_MIN(global, local);
      }

      ONIKA_HOST_DEVICE_FUNC inline void operator()(T &global, const T& local, reduce_global_t) const
      {
        ONIKA_CU_ATOMIC_MIN(global, local);
      }
    };
}

namespace exanb
{
  template <typename T> struct ReduceCellParticlesTraits<exaDEM::ReduceMinFieldSet<T>>
  {
    static inline constexpr bool RequiresBlockSynchronousCall = true;
    static inline constexpr bool RequiresCellParticleIndex = false;
    static inline constexpr bool CudaCompatible = true;
  };
} // namespace exanb


namespace exaDEM
{
  using namespace exanb;

  template <typename T, class FieldSetT, typename GridT> class ReduceMinFieldOP : public OperatorNode
  {
    using ReduceField = FieldSet<FieldSetT>;
    static constexpr ReduceField reduce_field{};
    ADD_SLOT(GridT, grid, INPUT_OUTPUT, REQUIRED);
    ADD_SLOT(T, result, OUTPUT);
    ADD_SLOT(bool, print_value, INPUT, false, DocString({"Enable to print the reduced value"}));
    ADD_SLOT(MPI_Comm, mpi, INPUT, MPI_COMM_WORLD);
    ADD_SLOT(Traversal, traversal_real, INPUT, REQUIRED, DocString{"list of non empty cells within the current grid"});

    // -----------------------------------------------
    // ----------- Operator documentation ------------
    inline std::string documentation() const override final
    {
      return R"EOF(
        This operator compute the minimal value of a field. It does not work for quaternion or vec3d. 
        )EOF";
    }

    public:

    inline void execute() override final
    {
      T local = std::numeric_limits<T>::max();
      ReduceMinFieldSet<T> func;
      const ReduceCellParticlesOptions rcpo = traversal_real->get_reduce_cell_particles_options();
lout << " size " << rcpo.m_num_cell_indices << std::endl;
      if(rcpo.m_num_cell_indices > 0)
      {
        reduce_cell_particles(*grid, false, func, local, reduce_field, parallel_execution_context(), {}, rcpo);
        ONIKA_CU_DEVICE_SYNCHRONIZE();
      }
      T global;
      MPI_Allreduce(&local, &global, 1, onika::mpi::mpi_datatype<T>(), MPI_MIN, *mpi);
      if(*print_value) lout << "min result: " << global << std::endl;
      *result = global; 
    }
  };

  template <class GridT> using MinMassTmpl = ReduceMinFieldOP<double, field::_mass, GridT>;
  template <class GridT> using MinRxTmpl = ReduceMinFieldOP<double, field::_rx, GridT>;
  template <class GridT> using MinRyTmpl = ReduceMinFieldOP<double, field::_ry, GridT>;
  template <class GridT> using MinRzTmpl = ReduceMinFieldOP<double, field::_rz, GridT>;

  // === register factories ===
  ONIKA_AUTORUN_INIT(min_mass) { OperatorNodeFactory::instance()->register_factory("min_mass", make_grid_variant_operator<MinMassTmpl>); }
  ONIKA_AUTORUN_INIT(min_rx)   { OperatorNodeFactory::instance()->register_factory("min_rx", make_grid_variant_operator<MinRxTmpl>); }
  ONIKA_AUTORUN_INIT(min_ry)   { OperatorNodeFactory::instance()->register_factory("min_ry", make_grid_variant_operator<MinRyTmpl>); }
  ONIKA_AUTORUN_INIT(min_rz)   { OperatorNodeFactory::instance()->register_factory("min_rz", make_grid_variant_operator<MinRzTmpl>); }

} // namespace exaDEM
