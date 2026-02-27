#pragma once

#include <cub/block/block_scan.cuh>

namespace exaDEM {
/**
 * @brief Packed particle data for detection and initialization.
 */
struct ParticleDetectPack {
  Quaternion quat;    ///< Particle orientation as a quaternion
  Vec3d r;            ///< Particle position
  uint64_t id;        ///< Unique particle ID
  ParticleTypeInt type; ///< Particle type (integer code)
  double radius;      ///< Particle radius
  double homothety;   ///< Scaling factor applied to particle size
};

/**
 * @brief Load a ParticleDetectPack from a cell container at index i.
 * @tparam TMPLC Type of the cell container (must support field access via operator[])
 * @param cell Reference to the cell container
 * @param i Index of the particle in the cell
 * @return ParticleDetectPack with all particle information
 */
template<typename TMPLC>
ONIKA_HOST_DEVICE_FUNC
inline ParticleDetectPack load(TMPLC& cell, size_t i) {
  ParticleDetectPack p;

  // Load orientation
  p.quat = cell[field::orient][i];

  // Load position
  p.r.x = cell[field::rx][i];
  p.r.y = cell[field::ry][i];
  p.r.z = cell[field::rz][i];

  // Load identification and type
  p.id   = cell[field::id][i];
  p.type = cell[field::type][i];

  // Load radius and scaling factor
  p.radius    = cell[field::radius][i];
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

  if (a.id>=b.id) {
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


  const onikaDim3_t& block = ONIKA_CU_BLOCK_DIMS;
  const onikaDim3_t& thread = ONIKA_CU_THREAD_COORD;

  for (int i = thread.x; i < nva; i+=block.x) {
    auto vi = vertices_a[i];
    // exclude possibilities with obb
    OBB obbvi;
    obbvi.center = {vi.x, vi.y, vi.z};
    obbvi.enlarge(shpa.minskowski(a.homothety));
    if (obb_b.intersect(obbvi)) {
      if (!func.skip(InteractionTypeId::VertexVertex)) {
        for (int j = thread.y; j < nvb; j+= block.y) {
          if (filter_vertex_vertex(PARAMETERS_SWAP_FALSE)) {
            func(i, j, InteractionTypeId::VertexVertex, false);
          }
        }  // thread.y
      }  // VertexVertex
      if (!func.skip(InteractionTypeId::VertexEdge)) {
        for (int j = thread.y; j < neb; j+= block.y) {
          if (filter_vertex_edge(PARAMETERS_SWAP_FALSE)) {
            func(i, j, InteractionTypeId::VertexEdge, false);
          }
        }  // thread.y
      }  // VertexEdge
      if (!func.skip(InteractionTypeId::VertexFace)) {
        for (int j = thread.y; j < nfb; j+= block.y) {
          if (filter_vertex_face(PARAMETERS_SWAP_FALSE)) {
            func(i, j, InteractionTypeId::VertexFace, false);
          }
        }  // thread.y
      }  // VertexFace
    }  // if obb
  }  // end thread.x

  func.swap_ij();

  for (int j = thread.y; j < nvb; j+= block.y) {
    auto vbj = vertices_b[j];
    OBB obbvj;
    obbvj.center = {vbj.x, vbj.y, vbj.z};
    obbvj.enlarge(shpb.minskowski(b.homothety));

    if (obb_a.intersect(obbvj)) {
      if (!func.skip(InteractionTypeId::VertexEdge)) {
        for (int i = thread.x; i < nea; i+= block.x) {
          if (filter_vertex_edge(PARAMETERS_SWAP_TRUE)) {
            func(j, i, InteractionTypeId::VertexEdge, true);
          }
        }  // thread.x
      }  // VertexEdge
      if (!func.skip(InteractionTypeId::VertexFace)) {
        for (int i = thread.x; i < nfa; i+= block.x) {
          if (filter_vertex_face(PARAMETERS_SWAP_TRUE)) {
            func(j, i, InteractionTypeId::VertexFace, true);
          }
        }  // thread.x 
      }  // VertexFace
    }  // if obb
  }  // thread.y
#undef PARAMETERS_SWAP_FALSE
#undef PARAMETERS_SWAP_TRUE
}

template<typename TMPLC>
struct ApplyNbhFunc {
  TMPLC cells;
  NbhCellAccessor accessor;
  const double rcut_inc;
  const shape* const shps;
  VertexField* const vertex_fields;
  ONIKA_HOST_DEVICE_FUNC inline void operator()(onikaInt3_t coord) const {
    long idx = coord.x;
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
      ONIKA_HOST_DEVICE_FUNC inline bool skip(uint8_t i) { return false ; }
      ONIKA_HOST_DEVICE_FUNC void swap_ij() {}
      ONIKA_HOST_DEVICE_FUNC inline void operator() (
          int i, int j, int InteractionType, bool swap) {
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
    for (int typeID = get_first_id<InteractionType::ParticleParticle>() ;
         typeID <= get_last_id<InteractionType::ParticleParticle>() ; typeID++) {
      if (func.counter[typeID]>0) {
        accessor.skip[idx] = false;
        //printf("do not skip cell pair %ld\n", idx); 
        ONIKA_CU_ATOMIC_ADD(res[typeID], func.counter[typeID]);
      }
    } 
  }
};

template<size_t BLOCKX, size_t BLOCKY, typename TMPLC>
struct ApplyClassifierFunc {  // Second pass 
                              // Note: This operator is quite demanding in terms of memory.
                              // Do not increase the number of members.
                              // That's why we only recover useful wrappers.
  TMPLC cells;
  NbhCellAccessor accessor;
  const double rcut_inc;
  const shape* const shps;
  VertexField* const vertex_fields;
  InteractionParticleAccessor interactions;

  ONIKA_HOST_DEVICE_FUNC inline void operator()(onikaInt3_t coord) const {
    using BlockScan = cub::BlockScan<int, BLOCKX, cub::BLOCK_SCAN_RAKING, BLOCKY>;
    long idx = coord.x;
    //std::assert(coord.y == 0);
    if (accessor.skip[idx]) {  // set by the first pass
      return;
    }

    // cub stuff
    ONIKA_CU_BLOCK_SHARED typename BlockScan::TempStorage temp_storage;
    // used by detection
    struct counter_func {
      InteractionTypePerCellCounter counter = {0,0,0,0};
      ONIKA_HOST_DEVICE_FUNC inline void swap_ij() {}
      ONIKA_HOST_DEVICE_FUNC inline bool skip(uint8_t i) { return false; }
      ONIKA_HOST_DEVICE_FUNC inline void operator() (
          int i, int j, int InteractionType, bool swap) {
        counter[InteractionType]++;
      }
    };

    struct AddInteractionFunc {
      const InteractionParticleAccessor& data; 
      PlaceholderInteraction item;
      InteractionTypePerCellCounter prefix;

      ONIKA_HOST_DEVICE_FUNC
          AddInteractionFunc(const InteractionParticleAccessor& in):
              data(in), prefix({0,0,0,0}) {};

      ONIKA_HOST_DEVICE_FUNC
          void set_ghost(int level_of_ghost) {
            item.pair.ghost = level_of_ghost;
          }

      ONIKA_HOST_DEVICE_FUNC
          inline void operator() (int i, int j, int InteractionType, bool swap) {
            item.pair.swap = swap;
            item.pair.pi.sub = i;
            item.pair.pj.sub = j;
            auto& PJ = item.pair.pj;
            PJ.sub = j;
            item.pair.type = InteractionType;
            /*
               printf("adder interaction %d at place %d = "
               "idi: %llu idj: %llu, subi: %u, subj: %u, swap %d\n",
               InteractionType,
               prefix[InteractionType],
               (unsigned long long)item.pair.pi.id,
               (unsigned long long)item.pair.pj.id,
               item.pair.pi.sub,
               item.pair.pj.sub,
               (int) item.pair.swap);
               */
            data[InteractionType].set(prefix[InteractionType]++, item);
          }

      ONIKA_HOST_DEVICE_FUNC inline bool skip(uint8_t i) {
        return false;
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

    AddInteractionFunc adder(interactions);
    // These Interactions are tagged to be copied in ghost areas
    adder.set_ghost(accessor.ghost[idx]);

    auto& sdata = accessor.offset[idx];
    for (int typeID = get_first_id<InteractionType::ParticleParticle>() ;
         typeID <= get_last_id<InteractionType::ParticleParticle>() ; typeID++) {
      BlockScan(temp_storage).ExclusiveSum(func.counter[typeID], adder.prefix[typeID]);
      ONIKA_CU_BLOCK_SYNC();
      adder.prefix[typeID] += sdata[typeID];
      //printf("adder.prefix[%d] = %d\n", typeID, adder.prefix[typeID]);
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
        adder.item.pair.pj.id = static_cast<uint64_t>(body_b.id);
        adder.item.pair.pj.p = static_cast<uint16_t>(pb);
        adder.item.pair.pj.cell = static_cast<size_t>(cell_id_b);
        detection(adder, rcut_inc, body_a, vertices_a,
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
struct BlockParallelForFunctorTraits<exaDEM::ApplyNbhFunc<TMPLC>> {
  static inline constexpr bool RequiresBlockSynchronousCall = false;
  static inline constexpr bool CudaCompatible = true;
};
template<size_t BX, size_t BY, typename TMPLC>
struct BlockParallelForFunctorTraits<exaDEM::ApplyClassifierFunc<BX,BY,TMPLC>> {
  static inline constexpr bool RequiresBlockSynchronousCall = false;
  static inline constexpr bool CudaCompatible = true;
};
}  // namespace parallel
}  // namespace onika
