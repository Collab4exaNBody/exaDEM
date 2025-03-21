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
#include <mpi.h>
#include <onika/scg/operator.h>
#include <onika/scg/operator_slot.h>
#include <onika/scg/operator_factory.h>
#include <exanb/core/grid.h>
#include <exanb/core/make_grid_variant_operator.h>
#include <exanb/compute/reduce_cell_particles.h>
#include <memory>

#include <exaDEM/traversal.h>


namespace exaDEM
{
  using namespace exanb;

  ONIKA_HOST_DEVICE_FUNC inline void ATOMIC_ADD(Vec3d& a, const Vec3d& b)
  {
     ONIKA_CU_ATOMIC_ADD(a.x, b.x);
     ONIKA_CU_ATOMIC_ADD(a.y, b.y);
     ONIKA_CU_ATOMIC_ADD(a.z, b.z);
  }

  struct ParticleVelMassValue
  {
    double m_tot; // the number of particles of a type
    Vec3d v_m_tot; // sum of the v_i * m_i
    void print() 
    { 
      lout << "{mass tot: " << m_tot << ", velocity * mass tot: " << v_m_tot << "}" << std::endl; 
    }
  };

  struct ReduceParticleVelMassFunctor
  {
    ONIKA_HOST_DEVICE_FUNC inline void operator()(ParticleVelMassValue& local, const double vx, const double vy, const double vz, const double mass, reduce_thread_local_t = {}) const
    {
      Vec3d v = {vx, vy, vz};
      local.v_m_tot += v * mass; 
      local.m_tot += mass;
    }

    ONIKA_HOST_DEVICE_FUNC inline void operator()(ParticleVelMassValue& global, const ParticleVelMassValue& local, reduce_thread_block_t) const
    {
      ONIKA_CU_ATOMIC_ADD(global.m_tot, local.m_tot);
      ATOMIC_ADD(global.v_m_tot, local.v_m_tot);
    }

    ONIKA_HOST_DEVICE_FUNC inline void operator()(ParticleVelMassValue& global, const ParticleVelMassValue& local, reduce_global_t) const
    {
      ONIKA_CU_ATOMIC_ADD(global.m_tot, local.m_tot);
      ATOMIC_ADD(global.v_m_tot, local.v_m_tot);
    }
  };
}

namespace exanb
{
  template <> struct ReduceCellParticlesTraits<exaDEM::ReduceParticleVelMassFunctor>
  {
    static inline constexpr bool RequiresBlockSynchronousCall = false;
    static inline constexpr bool RequiresCellParticleIndex = false;
    static inline constexpr bool CudaCompatible = true;
  };
} // namespace exanb

namespace exaDEM
{
  using namespace exanb;

  template <typename GridT, class = AssertGridHasFields<GridT, field::_vx, field::_vy, field::_vz,  field::_mass>> 
  class GetAvgVelMass : public OperatorNode
  {
    using ReduceFields = FieldSet<field::_vx, field::_vy, field::_vz, field::_mass>;
    static constexpr ReduceFields reduce_field_set{};
    ADD_SLOT(GridT, grid, INPUT_OUTPUT);
    ADD_SLOT(MPI_Comm, mpi, INPUT, MPI_COMM_WORLD);
    ADD_SLOT(Traversal, traversal_real, INPUT_OUTPUT, DocString{"list of non empty cells within the current grid"});
    ADD_SLOT(Vec3d, out, OUTPUT, DocString("Sum[v_i*m_i] / mass_{systeme}"));
    // Remark : do not hesite to use rebind to rename the output variable

    inline std::string documentation() const override final
    {
      return R"EOF(
        This operator returns out = Sum_{particles p}(v_p*m_p) / mass_{systeme}.
        )EOF";
    }

  public:
    inline void execute() override final
    {
      auto [data, size] = traversal_real->info();

      // Reduce over the subdomain
      ParticleVelMassValue value = {0.0, Vec3d{0.0,0.0,0.0}}; 
      ReduceParticleVelMassFunctor func;
      reduce_cell_particles(*grid, false, func, value, reduce_field_set, parallel_execution_context(), {}, data, size);

      // Reduce over MPI processes
      double local[4] = {value.m_tot, value.v_m_tot.x, value.v_m_tot.y, value.v_m_tot.z};
      double global[4] = {0.0, 0.0, 0.0, 0.0}; // count, x, y, z
      MPI_Allreduce(&local, &global, 4, MPI_DOUBLE, MPI_SUM, *mpi);

      assert(global[0] != 0.0);
      Vec3d v_m = {global[1] / global[0], global[2] / global[0], global[3] / global[0]};

      // Set the result into the output slot
      *out = -v_m; // 
    }
  };

  template <class GridT> using GetAvgVelMassTmpl = GetAvgVelMass<GridT>;

  // === register factories ===
  ONIKA_AUTORUN_INIT(avg_vel_m) { OperatorNodeFactory::instance()->register_factory("avg_v_m", make_grid_variant_operator<GetAvgVelMassTmpl>); }

} // namespace exaDEM
