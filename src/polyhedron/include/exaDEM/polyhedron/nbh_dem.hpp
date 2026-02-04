#pragma once

#include <exaDEM/shapes.hpp>
#include <exaDEM/polyhedron/nbh_runner.hpp>

namespace exaDEM {

struct brute_force_storage {
  onika::memory::CudaMMVector<uint64_t> m_data;  // store number of interactions per cell
};

struct skip_test {
  onika::memory::CudaMMVector<std::pair<uint64_t, uint64_t>> fragmentation;
};

struct ApplyNbhBruteForceFunc {

  template<typename TMPLC> ONIKA_HOST_DEVICE_FUNC
  inline void operator()(uint64_t cell_id_a, uint64_t cell_id_b,
                         TMPLC cells, uint64_t* const global_nb_inters,
                         const double rcut_inc, const shape* const shps) const {
    auto& cell_a = cells[cell_id_a];
    uint64_t local_nb_inters = 0;

    for(size_t pa = 0; pa < cell_a.size() ; pa++) {

      const double rxa = cell_a[field::rx][pa];
      const double rya = cell_a[field::ry][pa];
      const double rza = cell_a[field::rz][pa];
      Vec3d ra = {rxa, rya, rza};
      const auto ta = cell_a[field::type][pa];
      const double rada = cell_a[field::radius][pa];

      AABB aabb_particle_a = {ra- rada - rcut_inc, ra + rada + rcut_inc};

      const Quaternion& quata = cell_a[field::orient][pa];
      const double& ha = cell_a[field::homothety][pa];
      OBB obb_a = compute_obb(shps[ta].obb, ra, quata, ha);
      obb_a.enlarge(rcut_inc);

      auto& cell_b = cells[cell_id_b];

      for(size_t pb = 0; pb < cell_b.size() ; pb++) {
        const double rxb = cell_b[field::rx][pb];
        const double ryb = cell_b[field::ry][pb];
        const double rzb = cell_b[field::rz][pb];
        const Vec3d rb = {rxb, ryb, rzb};
        const double radb = cell_b[field::radius][pb];
        // very coarse test
        if( !is_inside_threshold(aabb_particle_a, rb, radb)) {
          continue;
        }

        Vec3d r = rb - ra;
        double rmax = rada + radb + rcut_inc;

        // basic tests
        if( exanb::dot(r,r) > rmax*rmax ) {
          continue;
        }

        // now test OBB
        const uint16_t tb = cell_b[field::type][pb];
        const Quaternion& quatb = cell_b[field::orient][pb];
        const double hb = cell_b[field::homothety][pb];
        OBB obb_b = compute_obb(shps[tb].obb, rb, quatb, hb);

        if( !obb_a.intersect(obb_b) ) {
          continue;
        }

        // TODO 
        local_nb_inters += 1; 
      }
    }
    ONIKA_CU_ATOMIC_ADD(global_nb_inters[cell_id_a], local_nb_inters);
  }
};

template<>
struct NeighborRunnerFunctorTraits<ApplyNbhBruteForceFunc> {
  static inline constexpr bool RequiresBlockSynchronousCall = false;
  static inline constexpr bool CudaCompatible = true;
};
}  // namespace exaDEM
