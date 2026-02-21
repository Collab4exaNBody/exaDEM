#pragma once

#include <exaDEM/shapes.hpp>
#include <exaDEM/polyhedron/nbh_runner.hpp>
#include <cub/block/block_scan.cuh>

namespace exaDEM {

struct ParticleDetectPack {
  Quaternion quat;
  Vec3d r;
  uint64_t id;
  ParticleTypeInt type;
  double radius;
  double homothety;
};

template<typename TMPLC> ONIKA_HOST_DEVICE_FUNC
inline ParticleDetectPack load(TMPLC& cell, size_t i) {
  ParticleDetectPack p;
  p.quat = cell[field::orient][i];
  p.r.x = cell[field::rx][i];
  p.r.y = cell[field::ry][i];
  p.r.z = cell[field::rz][i];
  p.id = cell[field::id][i];
  p.type = cell[field::type][i];
  p.radius = cell[field::radius][i];
  p.homothety = cell[field::homothety][i];
  return p;
}

template<typename Func> ONIKA_HOST_DEVICE_FUNC
inline void detection(Func& func,
                      const double rcut_inc,
                      ParticleDetectPack& a,
                      ParticleVertexView& vertices_a,
                      const shape& shpa,
                      AABB& aabb,
                      OBB& obb_a,
                      ParticleDetectPack& b,
                      ParticleVertexView& vertices_b,
                      const shape& shpb) {
  if (a.id>=b.id) {  // TODO add ghost
    return;
  }
  // very coarse test
  if( !is_inside_threshold(aabb, b.r, b.radius)) {
    return;
  }

  Vec3d r = b.r - a.r;
  double rmax = a.radius + b.radius + rcut_inc;

  // basic tests
  if (exanb::dot(r,r) > rmax*rmax) {
    return;
  }

  // now test OBB
  OBB obb_b = compute_obb(shpb.obb, b.r, b.quat, b.homothety);
  if (!obb_a.intersect(obb_b)) {
    return;
  }

  obb_b.enlarge(rcut_inc);
  // get particle j data.
  const int nva = shpa.get_number_of_vertices();
  const int nea = shpa.get_number_of_edges();
  const int nfa = shpa.get_number_of_faces();
  const int nvb = shpb.get_number_of_vertices();
  const int neb = shpb.get_number_of_edges();
  const int nfb = shpb.get_number_of_faces();

#define PARAMETERS_SWAP_FALSE rcut_inc, vertices_a, a.homothety, i, &shpa, vertices_b, b.homothety, j, &shpb
#define PARAMETERS_SWAP_TRUE rcut_inc, vertices_b, b.homothety, j, &shpb, vertices_a, a.homothety, i, &shpa

  for (int i = 0; i < nva; i++) {
    auto vi = vertices_a[i];
    // exclude possibilities with obb
    OBB obbvi;
    obbvi.center = {vi.x, vi.y, vi.z};
    obbvi.enlarge(shpa.minskowski(a.homothety));
    if (obb_b.intersect(obbvi)) {
      for (int j = 0; j < nvb; j++) {
        if (filter_vertex_vertex(PARAMETERS_SWAP_FALSE)) {
          func(i, j, InteractionTypeId::VertexVertex, false);
        }
      }
      for (int j = 0; j < neb; j++) {
        if (filter_vertex_edge(PARAMETERS_SWAP_FALSE)) {
          func(i, j, InteractionTypeId::VertexEdge, false);
        }
      }
      for (int j = 0; j < nfb; j++) {
        if (filter_vertex_face(PARAMETERS_SWAP_FALSE)) {
          func(i, j, InteractionTypeId::VertexFace, false);
        }
      }
    }
  }

  for (int j = 0; j < nvb; j++) {
    auto vbj = vertices_b[j];
    OBB obbvj;
    obbvj.center = {vbj.x, vbj.y, vbj.z};
    obbvj.enlarge(shpb.minskowski(b.homothety));

    if (obb_a.intersect(obbvj)) {
      for (int i = 0; i < nea; i++) {
        if (filter_vertex_edge(PARAMETERS_SWAP_TRUE)) {
          func(i, j, InteractionTypeId::VertexEdge, true);
        }
      }
      for (int i = 0; i < nfa; i++) {
        if (filter_vertex_face(PARAMETERS_SWAP_TRUE)) {
          func(i, j, InteractionTypeId::VertexFace, true);
        }
      }
    }
  }
#undef PARAMETERS_SWAP_FALSE
#undef PARAMETERS_SWAP_TRUE
}


template<typename TMPLC>
struct ApplyNbhFunc {
  TMPLC cells;
  NbhManagerAcessor accessor;
  const double rcut_inc;
  const shape* const shps;
  VertexField* const vertex_fields;
  ONIKA_HOST_DEVICE_FUNC
      inline void operator()(uint64_t idx) const {
        size_t cell_id_a = accessor.owner_cell[idx];
        size_t cell_id_b = accessor.partner_cell[idx];
        auto& cell_a = cells[cell_id_a];
        auto& cell_b = cells[cell_id_b];
        InteractionTypePerCellCounter counter = {0,0,0,0};
        VertexField& vertex_cell_a = vertex_fields[cell_id_a];
        VertexField& vertex_cell_b = vertex_fields[cell_id_b];

        // used by detection
        auto func = [&counter] (size_t i, size_t j, int InteractionType, bool swap) {
          counter[InteractionType]++;
        };

        for(size_t pa = 0; pa < cell_a.size() ; pa++) {
          // load data relative to the particle a
          auto body_a = load(cell_a, pa); 
          ParticleVertexView vertices_a = {pa, vertex_cell_a};
          auto& shpa = shps[body_a.type];

          // setup geometric test prerequis
          AABB aabb_body_a = { body_a.r - body_a.radius - rcut_inc,
            body_a.r + body_a.radius + rcut_inc};

          OBB obb_a = compute_obb(shpa.obb,
                                  body_a.r, body_a.quat,
                                  body_a.homothety);
          obb_a.enlarge(rcut_inc);

          for(size_t pb = 0; pb < cell_b.size() ; pb++) {
            // load data relative to the particle b
            auto body_b = load(cell_b, pb);
            auto& shpb = shps[body_b.type];
            ParticleVertexView vertices_b = {pb, vertex_cell_b};
            detection(func, rcut_inc, body_a, vertices_a,
                      shpa, aabb_body_a, obb_a,
                      body_b, vertices_b, shpb);
          }
        }
        auto& res = accessor.size[idx];
        for (int type_id = 0 ; type_id < ParticleParticleSize ; type_id++) {
          if (counter[type_id]>0) {
            accessor.skip[idx] = false;
            ONIKA_CU_ATOMIC_ADD(res[type_id], counter[type_id]);
          }
        } 
      }
};

/*
template<size_t BLOCKX, size_t, BLOCKY, typename TMPLC>
struct ApplyClassifierFunc {  // Second pass 
  TMPLC cells;
  NbhManagerAcessor accessor;
  const double rcut_inc;
  const shape* const shps;
  VertexField* const vertex_fields;
  InteractionWrapper<ParticleParticle> interactions[ParticleParticleSize];
  ONIKA_HOST_DEVICE_FUNC
      inline void operator()(uint64_t idx) const {
        using BlockScan = cub::BlockScan<int, BLOCKX, cub::BLOCK_SCAN_RAKING, BLOCKY>;

        if (accessor.skip[idx]) {  // set by the first pass
          return;
        }

        // cub stuff
        ONIKA_CU_BLOCK_SHARED typename BlockScan::TempStorage temp_storage;

        size_t cell_id_a = accessor.owner_cell[idx];
        size_t cell_id_b = accessor.partner_cell[idx];
        auto& cell_a = cells[cell_id_a];
        auto& cell_b = cells[cell_id_b];
        InteractionTypePerCellCounter counter = {0,0,0,0};
        VertexField& vertex_cell_a = vertex_fields[cell_id_a];
        VertexField& vertex_cell_b = vertex_fields[cell_id_b];



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
#define PARAMETERS_SWAPParticleTypeInt_TRUE rcut_inc, vertices_b, hb, j, &shps[tb], vertices_a, ha, i, &shps[ta]

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
        auto& res = accessor.size[idx];
        for (int type_id = 0 ; type_id < ParticleParticleSize ; type_id++) {
          if (counter[type_id]>0) {
            lout << "add type id " << type_id << " = " << counter[type_id] << std::endl;
            ONIKA_CU_ATOMIC_ADD(res[type_id], counter[type_id]);
          }
        } 
      }
};
*/

}  // namespace exaDEM

namespace onika {
namespace parallel {
template<typename TMPLC>
struct ParallelForFunctorTraits<exaDEM::ApplyNbhFunc<TMPLC>> {
  static inline constexpr bool RequiresBlockSynchronousCall = false;
  static inline constexpr bool CudaCompatible = true;
};
}  // namespace parallel
}  // namespace onika
