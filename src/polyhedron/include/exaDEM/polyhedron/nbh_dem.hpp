#pragma once

#include <exaDEM/shapes.hpp>
#include <exaDEM/polyhedron/nbh_runner.hpp>

namespace exaDEM {

static constexpr int ParticleParticleSize = 4;
typedef std::array<size_t, ParticleParticleSize> InteractionTypePerCellCounter;
//typedef InteractionTypePerCellCounter std::array<size_t, ParticleParticleSize>;

void debug_print(InteractionTypePerCellCounter& in)
{
  std::cout << "VertexVertex = " << in[InteractionTypeId::VertexVertex] << std::endl;
  std::cout << "VertexEdge   = " << in[InteractionTypeId::VertexEdge] << std::endl;
  std::cout << "VertexFace   = " << in[InteractionTypeId::VertexFace] << std::endl;
  std::cout << "EdgeEdge     = " << in[InteractionTypeId::EdgeEdge] << std::endl;
}

void debug_print(InteractionTypePerCellCounter& in1, InteractionTypePerCellCounter& in2)
{
  InteractionTypePerCellCounter sum;
  for(size_t i=0 ; i<ParticleParticleSize ; i++) {
    sum[i] = in1[i] + in2[i];
  }
  debug_print(sum);
}

struct brute_force_storage {
  onika::memory::CudaMMVector<InteractionTypePerCellCounter> m_data;  // store number of interactions per cell
};

struct skip_test {
  onika::memory::CudaMMVector<std::pair<uint64_t, uint64_t>> fragmentation;
};

struct PrefxSumInteractionTypePerCellCounter {
  InteractionTypePerCellCounter* const offset;
  InteractionTypePerCellCounter* const size;
  size_t n_elem;

  ONIKA_HOST_DEVICE_FUNC inline void operator()(uint64_t id) const {
    offset[0][id] = 0;
    for (size_t i=1 ; i<n_elem ; i++) {
      offset[i][id] = offset[i-1][id] + size[i-1][id];
    }
  }
};

struct ApplyNbhBruteForceFunc {
  template<typename TMPLC> ONIKA_HOST_DEVICE_FUNC
  inline void operator()(uint64_t cell_id_a, uint64_t cell_id_b,
                         TMPLC cells, InteractionTypePerCellCounter* const global_nb_inters,
                         const double rcut_inc, const shape* const shps,
                         VertexField* const vertex_fields) const {
    auto& cell_a = cells[cell_id_a];
    InteractionTypePerCellCounter counter = {0,0,0,0};
    auto& vertex_cell_a = vertex_fields[cell_id_a];
    for(size_t pa = 0; pa < cell_a.size() ; pa++) {
      const double rxa = cell_a[field::rx][pa];
      const double rya = cell_a[field::ry][pa];
      const double rza = cell_a[field::rz][pa];
      const uint64_t ida = cell_a[field::id][pa];
      Vec3d ra = {rxa, rya, rza};
      const auto ta = cell_a[field::type][pa];
      const double rada = cell_a[field::radius][pa];

      AABB aabb_particle_a = {ra- rada - rcut_inc, ra + rada + rcut_inc};

      const Quaternion& quata = cell_a[field::orient][pa];
      const double& ha = cell_a[field::homothety][pa];
      OBB obb_a = compute_obb(shps[ta].obb, ra, quata, ha);
      obb_a.enlarge(rcut_inc);
      ParticleVertexView vertices_a = {pa, vertex_cell_a};

      auto& cell_b = cells[cell_id_b];
      VertexField& vertex_cell_b = vertex_fields[cell_id_a];
      auto& shpa = shps[ta];

      for(size_t pb = 0; pb < cell_b.size() ; pb++) {
        const uint64_t idb = cell_b[field::id][pb];
        if (ida>=idb) {  // TODO add ghost
          continue;
        }
        //if (cell_id_a == cell_id_b && pb == pa) continue;
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
        if (exanb::dot(r,r) > rmax*rmax) {
          continue;
        }

        // now test OBB
        const uint16_t tb = cell_b[field::type][pb];
        const Quaternion& quatb = cell_b[field::orient][pb];
        const double hb = cell_b[field::homothety][pb];
        auto& shpb = shps[tb];
        OBB obb_b = compute_obb(shpb.obb, rb, quatb, hb);

        if (!obb_a.intersect(obb_b)) {
          continue;
        }

        obb_b.enlarge(rcut_inc);
        ParticleVertexView vertices_b = {pb, vertex_cell_b};
        // get particle j data.
        const int nva = shpa.get_number_of_vertices();
        const int nea = shpa.get_number_of_edges();
        const int nfa = shpa.get_number_of_faces();
        const int nvb = shpb.get_number_of_vertices();
        const int neb = shpb.get_number_of_edges();
        const int nfb = shpb.get_number_of_faces();

#define PARAMETERS_SWAP_FALSE rcut_inc, vertices_a, ha, i, &shps[ta], vertices_b, hb, j, &shps[tb]
#define PARAMETERS_SWAP_TRUE rcut_inc, vertices_b, hb, j, &shps[tb], vertices_a, ha, i, &shps[ta]

        // exclude possibilities with obb
        for (int i = 0; i < nva; i++) {
          auto vi = vertices_a[i];
          OBB obbvi;
          obbvi.center = {vi.x, vi.y, vi.z};
          obbvi.enlarge(shpa.minskowski(ha));
          if (obb_b.intersect(obbvi)) {
            for (int j = 0; j < nvb; j++) {
              if (filter_vertex_vertex(PARAMETERS_SWAP_FALSE)) {
                counter[InteractionTypeId::VertexVertex]++;
              }
            }
            for (int j = 0; j < neb; j++) {
              if (filter_vertex_edge(PARAMETERS_SWAP_FALSE)) {
                counter[InteractionTypeId::VertexEdge]++;
              }
            }
            for (int j = 0; j < nfb; j++) {
              if (filter_vertex_face(PARAMETERS_SWAP_FALSE)) {
                counter[InteractionTypeId::VertexFace]++;
              }
            }
          }
        }

        for (int i = 0; i < nea; i++) {
          for (int j = 0; j < neb; j++) {
            if (filter_edge_edge(PARAMETERS_SWAP_FALSE)) {
              counter[InteractionTypeId::EdgeEdge]++;
            }
          }
        }

        for (int j = 0; j < nvb; j++) {
          auto vbj = vertices_b[j];
          OBB obbvj;
          obbvj.center = {vbj.x, vbj.y, vbj.z};
          obbvj.enlarge(shpb.minskowski(hb));

          if (obb_a.intersect(obbvj)) {
            for (int i = 0; i < nea; i++) {
              if (filter_vertex_edge(PARAMETERS_SWAP_TRUE)) {
                counter[InteractionTypeId::VertexEdge]++;
              }
            }
            for (int i = 0; i < nfa; i++) {
              if (filter_vertex_face(PARAMETERS_SWAP_TRUE)) {
                counter[InteractionTypeId::VertexFace]++;
              }
            }
          }
        }
#undef PARAMETERS_SWAP_FALSE
#undef PARAMETERS_SWAP_TRUE
      }
    }
    auto& res = global_nb_inters[cell_id_a];
    for (int type_id = 0 ; type_id < ParticleParticleSize ; type_id++) {
      if (counter[type_id]>0) {
        ONIKA_CU_ATOMIC_ADD(res[type_id], counter[type_id]);
      }
    } 
  }
};

template<>
struct NeighborRunnerFunctorTraits<ApplyNbhBruteForceFunc> {
  static inline constexpr bool RequiresBlockSynchronousCall = false;
  static inline constexpr bool CudaCompatible = true;
};
}  // namespace exaDEM

namespace onika {
namespace parallel {
template<>
struct ParallelForFunctorTraits<exaDEM::PrefxSumInteractionTypePerCellCounter> {
  static inline constexpr bool RequiresBlockSynchronousCall = false;
  static inline constexpr bool CudaCompatible = true;
};
}  // namespace parallel
}  // namespace onika
