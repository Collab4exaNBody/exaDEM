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

template<typename T>
ONIKA_HOST_DEVICE_FUNC inline void gpu_swap(T& a, T& b) {
  T tmp = a;
  a = b;
  b = tmp;
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

  func.set_ghost(InteractionPair::NotGhost);

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

  func.swap_ij();

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
        VertexField& vertex_cell_a = vertex_fields[cell_id_a];
        VertexField& vertex_cell_b = vertex_fields[cell_id_b];

        // used by detection
        struct counter_func {
          InteractionTypePerCellCounter counter;
          ONIKA_HOST_DEVICE_FUNC counter_func() : counter({0,0,0,0}) {}
          ONIKA_HOST_DEVICE_FUNC void set_ghost(bool g) {}
          ONIKA_HOST_DEVICE_FUNC void swap_ij() {}
          ONIKA_HOST_DEVICE_FUNC inline void operator() (
              size_t i, size_t j, int InteractionType, bool swap) {
            counter[InteractionType]++;
          }
        };

        counter_func func;

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
          if (func.counter[type_id]>0) {
            accessor.skip[idx] = false;
            ONIKA_CU_ATOMIC_ADD(res[type_id], func.counter[type_id]);
          }
        } 
      }
};

template<size_t BLOCKX, size_t BLOCKY, typename TMPLC>
struct ApplyClassifierFunc {  // Second pass 
  using array_i = onika::oarray_t<InteractionWrapper<ParticleParticle>, ParticleParticleSize>;
  TMPLC cells;
  NbhManagerAcessor accessor;
  const double rcut_inc;
  const shape* const shps;
  VertexField* const vertex_fields;
  array_i interactions;
  ONIKA_HOST_DEVICE_FUNC inline void operator()(uint64_t idx) const {
    using BlockScan = cub::BlockScan<int, BLOCKX, cub::BLOCK_SCAN_RAKING, BLOCKY>;

    if (accessor.skip[idx]) {  // set by the first pass
      return;
    }
    // cub stuff
    ONIKA_CU_BLOCK_SHARED typename BlockScan::TempStorage temp_storage;
    // used by detection
    struct counter_func {
      InteractionTypePerCellCounter counter = {0,0,0,0};
      ONIKA_HOST_DEVICE_FUNC void set_ghost(bool g) {}
      ONIKA_HOST_DEVICE_FUNC void swap_ij() {}
      ONIKA_HOST_DEVICE_FUNC inline void operator() (
          size_t i, size_t j, int InteractionType, bool swap) {
        counter[InteractionType]++;
      }
    };

    struct add_interaction {
      const array_i& data; 
      PlaceholderInteraction item;
      InteractionTypePerCellCounter prefix;

      ONIKA_HOST_DEVICE_FUNC
      add_interaction(const array_i& in) : data(in), prefix({0,0,0,0}) {};

      ONIKA_HOST_DEVICE_FUNC
          void set_ghost(int level_of_ghost) {
            item.pair.ghost = level_of_ghost;
          }

      ONIKA_HOST_DEVICE_FUNC
          inline void operator() (size_t i, size_t j, int InteractionType, bool swap) {
            item.pair.swap = swap;
            item.pair.pi.sub = i;
            item.pair.pj.sub = j;
            item.pair.type = InteractionType;
            data[InteractionType].set(prefix[InteractionType]++, item);
          }


      ONIKA_HOST_DEVICE_FUNC inline void swap_ij() {
        gpu_swap(item.pair.pi.id, item.pair.pj.id);
        gpu_swap(item.pair.pi.cell, item.pair.pj.cell);
        gpu_swap(item.pair.pi.p, item.pair.pj.p);
      }
    };

    counter_func func;
    size_t cell_id_a = accessor.owner_cell[idx];
    size_t cell_id_b = accessor.partner_cell[idx];
    auto& cell_a = cells[cell_id_a];
    auto& cell_b = cells[cell_id_b];
    VertexField& vertex_cell_a = vertex_fields[cell_id_a];
    VertexField& vertex_cell_b = vertex_fields[cell_id_b];
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

    add_interaction adder(interactions);
    auto& sdata = accessor.offset[idx];
    for(int type = 0 ; type < ParticleParticleSize ; type++)
    {
      BlockScan(temp_storage).ExclusiveSum(func.counter[type], adder.prefix[type]);
      ONIKA_CU_BLOCK_SYNC();
      adder.prefix[type] += sdata[type];
    }

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
        // do not forget to reset the interaction
        adder.item.pair.pi.id = body_a.id;
        adder.item.pair.pi.p = pa;
        adder.item.pair.pi.cell = cell_id_a;
        adder.item.pair.pj.id = body_b.id;
        adder.item.pair.pj.p = pb;
        adder.item.pair.pj.cell = cell_id_b;
        detection(func, rcut_inc, body_a, vertices_a,
                  shpa, aabb_body_a, obb_a,
                  body_b, vertices_b, shpb);
      }
    }
  }
};

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
