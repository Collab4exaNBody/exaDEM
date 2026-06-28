#pragma once

#include <cub/block/block_scan.cuh>

namespace exaDEM {
/**
 * @brief Packed particle data for detection and initialization.
 */
struct ParticleDetectPack {
  Quaternion quat_;       ///< Particle orientation as a quaternion
  Vec3d r_;               ///< Particle position
  uint64_t id_;           ///< Unique particle ID
  ParticleTypeInt type_;  ///< Particle type (integer code)
  double radius_;         ///< Particle radius
  double homothety_;      ///< Scaling factor applied to particle size
};

/**
 * @brief Load a ParticleDetectPack from a cell container at index i.
 * @tparam TMPLC Type of the cell container (must support field access via operator[])
 * @param cell Reference to the cell container
 * @param i Index of the particle in the cell
 * @return ParticleDetectPack with all particle information
 */
template <typename TMPLC>
ONIKA_HOST_DEVICE_FUNC inline ParticleDetectPack load(TMPLC& cell, size_t i) {
  ParticleDetectPack p;

  // Load orientation
  p.quat_ = cell[field::orient][i];

  // Load position
  p.r_.x = cell[field::rx][i];
  p.r_.y = cell[field::ry][i];
  p.r_.z = cell[field::rz][i];

  // Load identification and type
  p.id_ = cell[field::id][i];
  p.type_ = cell[field::type][i];

  // Load radius and scaling factor
  p.radius_ = cell[field::radius][i];
  p.homothety_ = cell[field::homothety][i];

  return p;
}

template <typename Func>
ONIKA_HOST_DEVICE_FUNC inline void detection(Func& func, const double rcut_inc, ParticleDetectPack& a,
                                             ParticleVertexView& vertices_a, const shape& shpa, AABB& aabb, OBB& obb_a,
                                             ParticleDetectPack& b, ParticleVertexView& vertices_b, const shape& shpb) {
  if (a.id_ >= b.id_) {
    return;
  }

  // very coarse test
  if (!is_inside_threshold(aabb, b.r_, b.radius_)) {
    return;
  }

  Vec3d r = b.r_ - a.r_;
  double rmax = a.radius_ + b.radius_ + rcut_inc;

  // basic tests
  if (exanb::dot(r, r) > rmax * rmax) {
    return;
  }

  // now test OBB
  OBB obb_b = compute_obb(shpb.obb_, b.r_, b.quat_, b.homothety_);
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

#define PARAMETERS_SWAP_FALSE rcut_inc, vertices_a, a.homothety_, i, &shpa, vertices_b, b.homothety_, j, &shpb
#define PARAMETERS_SWAP_TRUE rcut_inc, vertices_b, b.homothety_, j, &shpb, vertices_a, a.homothety_, i, &shpa

  const onikaDim3_t& block = ONIKA_CU_BLOCK_DIMS;
  const onikaDim3_t& thread = ONIKA_CU_THREAD_COORD;

  for (int i = thread.x; i < nva; i += block.x) {
    auto vi = vertices_a[i];
    // exclude possibilities with obb
    OBB obbvi;
    obbvi.center = {vi.x, vi.y, vi.z};
    obbvi.enlarge(shpa.minkowski(a.homothety_));
    if (obb_b.intersect(obbvi)) {
      if (!func.skip(InteractionTypeId::VertexVertex)) {
        for (int j = thread.y; j < nvb; j += block.y) {
          if (filter_vertex_vertex(PARAMETERS_SWAP_FALSE)) {
            func(i, j, InteractionTypeId::VertexVertex, false);
          }
        }  // thread.y
      }  // VertexVertex
      if (!func.skip(InteractionTypeId::VertexEdge)) {
        for (int j = thread.y; j < neb; j += block.y) {
          if (filter_vertex_edge(PARAMETERS_SWAP_FALSE)) {
            func(i, j, InteractionTypeId::VertexEdge, false);
          }
        }  // thread.y
      }  // VertexEdge
      if (!func.skip(InteractionTypeId::VertexFace)) {
        for (int j = thread.y; j < nfb; j += block.y) {
          if (filter_vertex_face(PARAMETERS_SWAP_FALSE)) {
            func(i, j, InteractionTypeId::VertexFace, false);
          }
        }  // thread.y
      }  // VertexFace
    }  // if obb
  }  // end thread.x

  for (int i = thread.x; i < nea; i += block.x) {
    for (int j = thread.y; j < neb; j += block.y) {
      if (filter_edge_edge(PARAMETERS_SWAP_FALSE)) {
        func(i, j, InteractionTypeId::EdgeEdge, false);
      }
    }
  }

  func.swap_ij();

  for (int j = thread.y; j < nvb; j += block.y) {
    auto vbj = vertices_b[j];
    OBB obbvj;
    obbvj.center = {vbj.x, vbj.y, vbj.z};
    obbvj.enlarge(shpb.minkowski(b.homothety_));

    if (obb_a.intersect(obbvj)) {
      if (!func.skip(InteractionTypeId::VertexEdge)) {
        for (int i = thread.x; i < nea; i += block.x) {
          if (filter_vertex_edge(PARAMETERS_SWAP_TRUE)) {
            func(j, i, InteractionTypeId::VertexEdge, true);
          }
        }  // thread.x
      }  // VertexEdge
      if (!func.skip(InteractionTypeId::VertexFace)) {
        for (int i = thread.x; i < nfa; i += block.x) {
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

template <typename TMPLC>
struct ApplyNbhFunc {
  TMPLC cells_;
  NbhCellAccessor accessor_;
  const double rcut_inc_;
  const shape* const shps_;
  VertexField* const vertex_fields_;
  ONIKA_HOST_DEVICE_FUNC inline void operator()(onikaInt3_t coord) const {
    long idx = coord.x;
    size_t cell_id_a = accessor_.owner_cell_[idx];
    size_t cell_id_b = accessor_.partner_cell_[idx];
    auto& cell_a = cells_[cell_id_a];
    auto& cell_b = cells_[cell_id_b];
    VertexField& vertex_cell_a = vertex_fields_[cell_id_a];
    VertexField& vertex_cell_b = vertex_fields_[cell_id_b];

    // used by detection
    struct counter_func {
      InteractionTypePerCellCounter counter_;
      ONIKA_HOST_DEVICE_FUNC counter_func() : counter_({0, 0, 0, 0}) {}
      ONIKA_HOST_DEVICE_FUNC inline bool skip(uint8_t i) { return false; }
      ONIKA_HOST_DEVICE_FUNC void swap_ij() {}
      ONIKA_HOST_DEVICE_FUNC inline void operator()(int i, int j, int InteractionType, bool swap) {
        counter_[InteractionType]++;
      }
    };

    counter_func func;

    for (size_t pa = 0; pa < cell_a.size(); pa++) {
      // load data relative to the particle a
      auto body_a = load(cell_a, pa);
      ParticleVertexView vertices_a = {pa, vertex_cell_a};
      auto& shpa = shps_[body_a.type_];

      // setup geometric test prerequis
      AABB aabb_body_a = {body_a.r_ - body_a.radius_ - rcut_inc_, body_a.r_ + body_a.radius_ + rcut_inc_};

      OBB obb_a = compute_obb(shpa.obb_, body_a.r_, body_a.quat_, body_a.homothety_);
      obb_a.enlarge(rcut_inc_);

      for (size_t pb = 0; pb < cell_b.size(); pb++) {
        // load data relative to the particle b
        auto body_b = load(cell_b, pb);
        auto& shpb = shps_[body_b.type_];
        ParticleVertexView vertices_b = {pb, vertex_cell_b};
        detection(func, rcut_inc_, body_a, vertices_a, shpa, aabb_body_a, obb_a, body_b, vertices_b, shpb);
      }
    }
    auto& res = accessor_.size_[idx];
    for (int typeID = get_first_id<InteractionType::ParticleParticle>();
         typeID <= get_last_id<InteractionType::ParticleParticle>(); typeID++) {
      if (func.counter_[typeID] > 0) {
        accessor_.skip_[idx] = false;
        // printf("do not skip cell pair %ld\n", idx);
        ONIKA_CU_ATOMIC_ADD(res[typeID], func.counter_[typeID]);
      }
    }
  }
};

template <size_t BLOCKX, size_t BLOCKY, typename TMPLC>
struct ApplyClassifierFunc {  // Second pass
                              // Note: This operator is quite demanding in terms of memory.
                              // Do not increase the number of members.
                              // That's why we only recover useful wrappers.
  TMPLC cells_;
  NbhCellAccessor accessor_;
  const double rcut_inc_;
  const shape* const shps_;
  VertexField* const vertex_fields_;
  InteractionParticleAccessor interactions_;

  ONIKA_HOST_DEVICE_FUNC inline void operator()(onikaInt3_t coord) const {
    using BlockScan = cub::BlockScan<int, BLOCKX, cub::BLOCK_SCAN_RAKING, BLOCKY>;
    long idx = coord.x;
    // std::assert(coord.y == 0);
    if (accessor_.skip_[idx]) {  // set by the first pass
      return;
    }

    // cub stuff
    ONIKA_CU_BLOCK_SHARED typename BlockScan::TempStorage temp_storage;
    // used by detection
    struct counter_func {
      InteractionTypePerCellCounter counter_ = {0, 0, 0, 0};
      ONIKA_HOST_DEVICE_FUNC inline void swap_ij() {}
      ONIKA_HOST_DEVICE_FUNC inline bool skip(uint8_t i) { return false; }
      ONIKA_HOST_DEVICE_FUNC inline void operator()(int i, int j, int InteractionType, bool swap) {
        counter_[InteractionType]++;
      }
    };

    struct AddInteractionFunc {
      const InteractionParticleAccessor& data_;
      PlaceholderInteraction item_;
      InteractionTypePerCellCounter prefix_;

      ONIKA_HOST_DEVICE_FUNC
      AddInteractionFunc(const InteractionParticleAccessor& in) : data_(in), item_{}, prefix_({0, 0, 0, 0}) {}

      ONIKA_HOST_DEVICE_FUNC
      void set_ghost(int level_of_ghost) { item_.pair_.ghost_ = level_of_ghost; }

      ONIKA_HOST_DEVICE_FUNC
      inline void operator()(int i, int j, int InteractionType, bool swap) {
        item_.pair_.swap_ = swap;
        item_.pair_.pi_.sub_ = i;
        item_.pair_.pj_.sub_ = j;
        auto& PJ = item_.pair_.pj_;
        PJ.sub_ = j;
        item_.pair_.type_ = InteractionType;
        /*
           printf("adder interaction %d at place %d = "
           "idi: %llu idj: %llu, subi: %u, subj: %u, swap %d\n",
           InteractionType,
           prefix_[InteractionType],
           (unsigned long long)item_.pair_.pi_.id_,
           (unsigned long long)item_.pair_.pj_.id_,
           item_.pair_.pi_.sub_,
           item_.pair_.pj_.sub_,
           (int) item_.pair_.swap_);
           */
        data_[InteractionType].set(prefix_[InteractionType]++, item_);
      }

      ONIKA_HOST_DEVICE_FUNC inline bool skip(uint8_t i) { return false; }

      ONIKA_HOST_DEVICE_FUNC inline void swap_ij() {
        gpu_swap(item_.pair_.pi_.id_, item_.pair_.pj_.id_);
        gpu_swap(item_.pair_.pi_.cell_, item_.pair_.pj_.cell_);
        gpu_swap(item_.pair_.pi_.p_, item_.pair_.pj_.p_);
      }
    };

    counter_func func;
    size_t cell_id_a = accessor_.owner_cell_[idx];
    size_t cell_id_b = accessor_.partner_cell_[idx];
    auto& cell_a = cells_[cell_id_a];
    auto& cell_b = cells_[cell_id_b];
    VertexField& vertex_cell_a = vertex_fields_[cell_id_a];
    VertexField& vertex_cell_b = vertex_fields_[cell_id_b];

    for (size_t pa = 0; pa < cell_a.size(); pa++) {
      // load data relative to the particle a
      auto body_a = load(cell_a, pa);
      ParticleVertexView vertices_a = {pa, vertex_cell_a};
      auto& shpa = shps_[body_a.type_];

      // setup geometric test prerequis
      AABB aabb_body_a = {body_a.r_ - body_a.radius_ - rcut_inc_, body_a.r_ + body_a.radius_ + rcut_inc_};

      OBB obb_a = compute_obb(shpa.obb_, body_a.r_, body_a.quat_, body_a.homothety_);
      obb_a.enlarge(rcut_inc_);

      for (size_t pb = 0; pb < cell_b.size(); pb++) {
        // load data relative to the particle b
        auto body_b = load(cell_b, pb);
        auto& shpb = shps_[body_b.type_];
        ParticleVertexView vertices_b = {pb, vertex_cell_b};
        detection(func, rcut_inc_, body_a, vertices_a, shpa, aabb_body_a, obb_a, body_b, vertices_b, shpb);
      }
    }

    AddInteractionFunc adder(interactions_);
    // These Interactions are tagged to be copied in ghost areas
    adder.set_ghost(accessor_.ghost_[idx]);

    auto& sdata = accessor_.offset_[idx];
    for (int typeID = get_first_id<InteractionType::ParticleParticle>();
         typeID <= get_last_id<InteractionType::ParticleParticle>(); typeID++) {
      BlockScan(temp_storage).ExclusiveSum(func.counter_[typeID], adder.prefix_[typeID]);
      ONIKA_CU_BLOCK_SYNC();
      adder.prefix_[typeID] += sdata[typeID];
      // printf("adder.prefix_[%d] = %d\n", typeID, adder.prefix_[typeID]);
    }

    for (size_t pa = 0; pa < cell_a.size(); pa++) {
      // load data relative to the particle a
      auto body_a = load(cell_a, pa);
      ParticleVertexView vertices_a = {pa, vertex_cell_a};
      auto& shpa = shps_[body_a.type_];

      // setup geometric test prerequis
      AABB aabb_body_a = {body_a.r_ - body_a.radius_ - rcut_inc_, body_a.r_ + body_a.radius_ + rcut_inc_};

      OBB obb_a = compute_obb(shpa.obb_, body_a.r_, body_a.quat_, body_a.homothety_);
      obb_a.enlarge(rcut_inc_);

      for (size_t pb = 0; pb < cell_b.size(); pb++) {
        // load data relative to the particle b
        auto body_b = load(cell_b, pb);
        auto& shpb = shps_[body_b.type_];
        ParticleVertexView vertices_b = {pb, vertex_cell_b};
        // do not forget to reset the interaction
        adder.item_.pair_.pi_.id_ = body_a.id_;
        adder.item_.pair_.pi_.p_ = pa;
        adder.item_.pair_.pi_.cell_ = cell_id_a;
        adder.item_.pair_.pj_.id_ = static_cast<uint64_t>(body_b.id_);
        adder.item_.pair_.pj_.p_ = static_cast<uint16_t>(pb);
        adder.item_.pair_.pj_.cell_ = static_cast<size_t>(cell_id_b);
        detection(adder, rcut_inc_, body_a, vertices_a, shpa, aabb_body_a, obb_a, body_b, vertices_b, shpb);
      }
    }
  }
};
}  // namespace exaDEM

namespace onika {
namespace parallel {
template <typename TMPLC>
struct BlockParallelForFunctorTraits<exaDEM::ApplyNbhFunc<TMPLC>> {
  static inline constexpr bool RequiresBlockSynchronousCall = false;
  static inline constexpr bool CudaCompatible = true;
};
template <size_t BX, size_t BY, typename TMPLC>
struct BlockParallelForFunctorTraits<exaDEM::ApplyClassifierFunc<BX, BY, TMPLC>> {
  static inline constexpr bool RequiresBlockSynchronousCall = false;
  static inline constexpr bool CudaCompatible = true;
};
}  // namespace parallel
}  // namespace onika
