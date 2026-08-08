#pragma once
#include <cub/cub.cuh>
#include <exaDEM/experimental/filter_pair_particle.hpp>
#include <exaDEM/experimental/polyhedron/nbh_gpu/nbh_storage.hpp>

namespace exaDEM {

/* Run an exclusive prefix sum on device memory using CUB. */
template <typename T>
inline void exclusive_scan_device(const T* input, T* output, size_t count, onikaStream_t st = 0) {
  void* d_tmp = nullptr;
  size_t tmp_bytes = 0;
  cub::DeviceScan::ExclusiveSum(d_tmp, tmp_bytes, input, output, count, st);
  cudaMalloc(&d_tmp, tmp_bytes);
  cub::DeviceScan::ExclusiveSum(d_tmp, tmp_bytes, input, output, count, st);
  cudaFree(d_tmp);
}

/**
 * @brief Storage for particle pairs (output of neighbor search).
 *
 * Holds the cell and particle indices of each detected pair, along with
 * the ghost tag and a back-reference to the originating cell pair.
 */
struct ParticlePairStorage {
  template <typename T>
  using VectorT = onika::memory::CudaMMVector<T>;
  VectorT<uint32_t> cell_i_;         ///< Cell index of particle i
  VectorT<uint32_t> cell_j_;         ///< Cell index of particle j
  VectorT<uint16_t> p_i_;            ///< Index of particle i within its cell
  VectorT<uint16_t> p_j_;            ///< Index of particle j within its cell
  VectorT<uint8_t> ghost_;           ///< Ghost tag per pair
  VectorT<uint32_t> cell_pair_idx_;  ///< Index of the cell pair this particle pair came from
  size_t size_ = 0;                  ///< Current number of stored particle pairs

  /**
   * @brief Resize all internal vectors to hold n particle pairs.
   * @param n New number of particle pairs.
   */
  void resize(size_t n) {
    cell_i_.resize(n);
    cell_j_.resize(n);
    p_i_.resize(n);
    p_j_.resize(n);
    ghost_.resize(n);
    cell_pair_idx_.resize(n);
    size_ = n;
  }
};

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
 * @tparam CellsT Type of the cell container (must support field access via operator[])
 * @param cell Reference to the cell container
 * @param i Index of the particle in the cell
 * @return ParticleDetectPack with all particle information
 */
template <typename CellsT>
ONIKA_HOST_DEVICE_FUNC inline ParticleDetectPack load(CellsT& cell, size_t i) {
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

// ============================================================
// Stage 1: Count particle pairs per cell pair
// 1 block = 1 cell pair, threads iterate particle pairs
// ============================================================
template <int BLOCKX, int BLOCKY, bool IGNORE_PAIR, typename CellsT>
__global__ __launch_bounds__(64, 8) void CountParticlePairsKernel(CellsT cells, size_t* __restrict__ owner_cells,
                                                                  size_t* __restrict__ partner_cells,
                                                                  uint8_t* __restrict__ ghost_flags, double rcut_inc,
                                                                  const shape* __restrict__ shps,
                                                                  VertexField* __restrict__ vertex_fields,
                                                                  int* __restrict__ pair_counts, size_t num_cell_pairs,
                                                                  IgnorePairsGPU::View ignore_pairs) {
  using BlockReduce = cub::BlockReduce<int, BLOCKX, cub::BLOCK_REDUCE_RAKING, BLOCKY>;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  size_t idx = blockIdx.x;
  if (idx >= num_cell_pairs) return;

  uint32_t cell_a = owner_cells[idx];
  uint32_t cell_b = partner_cells[idx];
  auto& cA = cells[cell_a];
  auto& cB = cells[cell_b];
  size_t nA = cA.size();
  size_t nB = cB.size();

  int count = 0;

  for (size_t pa = threadIdx.y; pa < nA; pa += blockDim.y) {
    auto body_a = load(cA, pa);
    const auto& shpa = shps[body_a.type_];

    AABB aabb_a = {body_a.r_ - body_a.radius_ - rcut_inc, body_a.r_ + body_a.radius_ + rcut_inc};

    for (size_t pb = threadIdx.x; pb < nB; pb += blockDim.x) {
      if constexpr (IGNORE_PAIR) {  // Skip pairs that are flagged to be ignored
        if (ignore_pairs(cell_a, body_a.id_, cB[field::id][pb])) continue;
      }
      auto body_b = load(cB, pb);

      if (body_a.id_ >= body_b.id_) continue;

      if (!is_inside_threshold(aabb_a, body_b.r_, body_b.radius_)) continue;

      Vec3d r = body_b.r_ - body_a.r_;
      double rmax = body_a.radius_ + body_b.radius_ + rcut_inc;
      if (exanb::dot(r, r) > rmax * rmax) continue;

      // OBB test
      const auto& shpb = shps[body_b.type_];
      OBB obb_a = compute_obb(shpa.obb_, body_a.r_, body_a.quat_, body_a.homothety_);
      obb_a.enlarge(rcut_inc);
      OBB obb_b = compute_obb(shpb.obb_, body_b.r_, body_b.quat_, body_b.homothety_);
      if (obb_a.intersect(obb_b)) {
        count++;
      }
    }
  }

  int aggregate = BlockReduce(temp_storage).Sum(count);
  __syncthreads();
  if (threadIdx.x == 0 && threadIdx.y == 0) pair_counts[idx] = aggregate;
}

// ============================================================
// Stage 2: Fill particle pair arrays
// 1 block = 1 cell pair
// ============================================================
template <int BLOCKX, int BLOCKY, bool IGNORE_PAIR, typename CellsT>
__global__ __launch_bounds__(64, 8) void FillParticlePairsKernel(
    CellsT cells, size_t* __restrict__ owner_cells, size_t* __restrict__ partner_cells,
    uint8_t* __restrict__ ghost_flags, double rcut_inc, const shape* __restrict__ shps,
    VertexField* __restrict__ vertex_fields, int* __restrict__ pair_offsets,
    // output
    uint32_t* __restrict__ out_cell_i, uint32_t* __restrict__ out_cell_j, uint16_t* __restrict__ out_p_i,
    uint16_t* __restrict__ out_p_j, uint8_t* __restrict__ out_ghost, uint32_t* __restrict__ out_cell_pair_idx,
    size_t num_cell_pairs, IgnorePairsGPU::View ignore_pairs) {
  using BlockScan = cub::BlockScan<int, BLOCKX, cub::BLOCK_SCAN_RAKING, BLOCKY>;
  __shared__ typename BlockScan::TempStorage temp_storage;

  size_t idx = blockIdx.x;
  if (idx >= num_cell_pairs) return;

  uint32_t cell_a = owner_cells[idx];
  uint32_t cell_b = partner_cells[idx];
  uint8_t ghost_flag = ghost_flags[idx];
  auto& cA = cells[cell_a];
  auto& cB = cells[cell_b];
  size_t nA = cA.size();
  size_t nB = cB.size();

  int count = 0;

  // First pass: count (for BlockScan)
  for (size_t pa = threadIdx.y; pa < nA; pa += blockDim.y) {
    auto body_a = load(cA, pa);
    const auto& shpa = shps[body_a.type_];
    AABB aabb_a = {body_a.r_ - body_a.radius_ - rcut_inc, body_a.r_ + body_a.radius_ + rcut_inc};

    for (size_t pb = threadIdx.x; pb < nB; pb += blockDim.x) {
      if constexpr (IGNORE_PAIR) {  // Skip pairs that are flagged to be ignored
        if (ignore_pairs(cell_a, body_a.id_, cB[field::id][pb])) continue;
      }
      auto body_b = load(cB, pb);
      // if (body_a.id_ >= body_b.id_ && ghost_flag == 0) continue;
      if (body_a.id_ >= body_b.id_) continue;
      if (!is_inside_threshold(aabb_a, body_b.r_, body_b.radius_)) continue;
      Vec3d r = body_b.r_ - body_a.r_;
      double rmax = body_a.radius_ + body_b.radius_ + rcut_inc;
      if (exanb::dot(r, r) > rmax * rmax) continue;
      const auto& shpb = shps[body_b.type_];
      OBB obb_a = compute_obb(shpa.obb_, body_a.r_, body_a.quat_, body_a.homothety_);
      obb_a.enlarge(rcut_inc);
      OBB obb_b = compute_obb(shpb.obb_, body_b.r_, body_b.quat_, body_b.homothety_);
      if (obb_a.intersect(obb_b)) count++;
    }
  }

  int prefix = 0;
  BlockScan(temp_storage).ExclusiveSum(count, prefix);
  __syncthreads();
  prefix += pair_offsets[idx];

  // Second pass: fill
  int write_idx = 0;
  for (size_t pa = threadIdx.y; pa < nA; pa += blockDim.y) {
    auto body_a = load(cA, pa);
    const auto& shpa = shps[body_a.type_];
    AABB aabb_a = {body_a.r_ - body_a.radius_ - rcut_inc, body_a.r_ + body_a.radius_ + rcut_inc};

    for (size_t pb = threadIdx.x; pb < nB; pb += blockDim.x) {
      if constexpr (IGNORE_PAIR) {  // Skip pairs that are flagged to be ignored
        if (ignore_pairs(cell_a, body_a.id_, cB[field::id][pb])) continue;
      }
      auto body_b = load(cB, pb);
      // if (body_a.id_ >= body_b.id_ && ghost_flag == 0) continue;
      if (body_a.id_ >= body_b.id_) continue;
      if (!is_inside_threshold(aabb_a, body_b.r_, body_b.radius_)) continue;
      Vec3d r = body_b.r_ - body_a.r_;
      double rmax = body_a.radius_ + body_b.radius_ + rcut_inc;
      if (exanb::dot(r, r) > rmax * rmax) continue;
      const auto& shpb = shps[body_b.type_];
      OBB obb_a = compute_obb(shpa.obb_, body_a.r_, body_a.quat_, body_a.homothety_);
      obb_a.enlarge(rcut_inc);
      OBB obb_b = compute_obb(shpb.obb_, body_b.r_, body_b.quat_, body_b.homothety_);
      if (obb_a.intersect(obb_b)) {
        int pos = prefix + write_idx;
        out_cell_i[pos] = cell_a;
        out_cell_j[pos] = cell_b;
        out_p_i[pos] = static_cast<uint16_t>(pa);
        out_p_j[pos] = static_cast<uint16_t>(pb);
        out_ghost[pos] = ghost_flag;
        out_cell_pair_idx[pos] = static_cast<uint32_t>(idx);
        write_idx++;
      }
    }
  }
}

// ============================================================
// Stage 3: Count interactions per particle pair
// 1 block = 1 particle pair (PCCP)
// ============================================================
template <int BLOCKX, int BLOCKY, typename CellsT>
__global__ void CountInteractionsPPKernel(CellsT cells, VertexField* __restrict__ vertex_fields,
                                          const shape* __restrict__ shps, double rcut_inc,
                                          uint32_t* __restrict__ pp_cell_i, uint32_t* __restrict__ pp_cell_j,
                                          uint16_t* __restrict__ pp_p_i, uint16_t* __restrict__ pp_p_j,
                                          InteractionTypePerCellCounter* __restrict__ count_data, size_t num_pairs) {
  using BlockReduce = cub::BlockReduce<int, BLOCKX, cub::BLOCK_REDUCE_RAKING, BLOCKY>;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  size_t idx = blockIdx.x;
  if (idx >= num_pairs) return;

  uint32_t cell_a = pp_cell_i[idx];
  uint32_t cell_b = pp_cell_j[idx];
  uint16_t pa = pp_p_i[idx];
  uint16_t pb = pp_p_j[idx];

  auto body_a = load(cells[cell_a], pa);
  auto body_b = load(cells[cell_b], pb);
  const auto& shpa = shps[body_a.type_];
  const auto& shpb = shps[body_b.type_];
  ParticleVertexView vertices_a = {pa, vertex_fields[cell_a]};
  ParticleVertexView vertices_b = {pb, vertex_fields[cell_b]};

  const int nva = shpa.get_number_of_vertices();
  const int nea = shpa.get_number_of_edges();
  const int nfa = shpa.get_number_of_faces();
  const int nvb = shpb.get_number_of_vertices();
  const int neb = shpb.get_number_of_edges();
  const int nfb = shpb.get_number_of_faces();

  int countVV = 0, countVE = 0, countVF = 0, countEE = 0;

  // A→B: vertex tests
  OBB obb_b = compute_obb(shpb.obb_, body_b.r_, body_b.quat_, body_b.homothety_);
  obb_b.enlarge(rcut_inc);

  for (int i = threadIdx.y; i < nva; i += blockDim.y) {
    for (int j = threadIdx.x; j < nvb; j += blockDim.x) {
      if (filter_vertex_vertex(rcut_inc, vertices_a, body_a.homothety_, i, &shpa, vertices_b, body_b.homothety_, j,
                               &shpb))
        countVV++;
    }
    for (int j = threadIdx.x; j < neb; j += blockDim.x) {
      if (filter_vertex_edge(rcut_inc, vertices_a, body_a.homothety_, i, &shpa, vertices_b, body_b.homothety_, j,
                             &shpb))
        countVE++;
    }
    for (int j = threadIdx.x; j < nfb; j += blockDim.x) {
      if (filter_vertex_face(rcut_inc, vertices_a, body_a.homothety_, i, &shpa, vertices_b, body_b.homothety_, j,
                             &shpb))
        countVF++;
    }
  }

  // Edge-Edge
  for (int i = threadIdx.y; i < nea; i += blockDim.y) {
    for (int j = threadIdx.x; j < neb; j += blockDim.x) {
      if (filter_edge_edge(rcut_inc, vertices_a, body_a.homothety_, i, &shpa, vertices_b, body_b.homothety_, j, &shpb))
        countEE++;
    }
  }

  // B→A: reverse VE, VF
  for (int j = threadIdx.y; j < nvb; j += blockDim.y) {
    for (int i = threadIdx.x; i < nea; i += blockDim.x) {
      if (filter_vertex_edge(rcut_inc, vertices_b, body_b.homothety_, j, &shpb, vertices_a, body_a.homothety_, i,
                             &shpa))
        countVE++;
    }
    for (int i = threadIdx.x; i < nfa; i += blockDim.x) {
      if (filter_vertex_face(rcut_inc, vertices_b, body_b.homothety_, j, &shpb, vertices_a, body_a.homothety_, i,
                             &shpa))
        countVF++;
    }
  }

  // Block reduce
  int types[4] = {countVV, countVE, countVF, countEE};
  for (int t = 0; t < 4; t++) {
    int agg = BlockReduce(temp_storage).Sum(types[t]);
    __syncthreads();
    if (threadIdx.x == 0 && threadIdx.y == 0) count_data[idx][t] = agg;
  }
}

// ============================================================
// Stage 4: Fill Classifier per particle pair
// 1 block = 1 particle pair (PCCP)
// ============================================================
template <int BLOCKX, int BLOCKY, typename CellsT>
__global__ __launch_bounds__(64, 10) void FillInteractionsPPKernel(
    CellsT cells, VertexField* __restrict__ vertex_fields, const shape* __restrict__ shps, double rcut_inc,
    uint32_t* __restrict__ pp_cell_i, uint32_t* __restrict__ pp_cell_j, uint16_t* __restrict__ pp_p_i,
    uint16_t* __restrict__ pp_p_j, uint8_t* __restrict__ pp_ghost,
    InteractionTypePerCellCounter* __restrict__ prefix_data, InteractionParticleAccessor interactions,
    size_t num_pairs) {
  using BlockScan = cub::BlockScan<int, BLOCKX, cub::BLOCK_SCAN_RAKING, BLOCKY>;
  __shared__ typename BlockScan::TempStorage temp_storage;

  size_t idx = blockIdx.x;
  if (idx >= num_pairs) return;

  uint32_t cell_a = pp_cell_i[idx];
  uint32_t cell_b = pp_cell_j[idx];
  uint16_t pa = pp_p_i[idx];
  uint16_t pb = pp_p_j[idx];
  uint8_t ghost_flag = pp_ghost[idx];

  auto body_a = load(cells[cell_a], pa);
  auto body_b = load(cells[cell_b], pb);
  const auto& shpa = shps[body_a.type_];
  const auto& shpb = shps[body_b.type_];
  ParticleVertexView vertices_a = {pa, vertex_fields[cell_a]};
  ParticleVertexView vertices_b = {pb, vertex_fields[cell_b]};

  const int nva = shpa.get_number_of_vertices();
  const int nea = shpa.get_number_of_edges();
  const int nfa = shpa.get_number_of_faces();
  const int nvb = shpb.get_number_of_vertices();
  const int neb = shpb.get_number_of_edges();
  const int nfb = shpb.get_number_of_faces();

  // Count pass with directional counters
  int count1 = 0;  // VV A→B
  int count2 = 0;  // VE A→B
  int count3 = 0;  // VF A→B
  int count4 = 0;  // EE
  int count5 = 0;  // VE B→A
  int count6 = 0;  // VF B→A

  for (int i = threadIdx.y; i < nva; i += blockDim.y) {
    for (int j = threadIdx.x; j < nvb; j += blockDim.x)
      if (filter_vertex_vertex(rcut_inc, vertices_a, body_a.homothety_, i, &shpa, vertices_b, body_b.homothety_, j,
                               &shpb))
        count1++;
    for (int j = threadIdx.x; j < neb; j += blockDim.x)
      if (filter_vertex_edge(rcut_inc, vertices_a, body_a.homothety_, i, &shpa, vertices_b, body_b.homothety_, j,
                             &shpb))
        count2++;
    for (int j = threadIdx.x; j < nfb; j += blockDim.x)
      if (filter_vertex_face(rcut_inc, vertices_a, body_a.homothety_, i, &shpa, vertices_b, body_b.homothety_, j,
                             &shpb))
        count3++;
  }
  for (int i = threadIdx.y; i < nea; i += blockDim.y)
    for (int j = threadIdx.x; j < neb; j += blockDim.x)
      if (filter_edge_edge(rcut_inc, vertices_a, body_a.homothety_, i, &shpa, vertices_b, body_b.homothety_, j, &shpb))
        count4++;
  for (int j = threadIdx.y; j < nvb; j += blockDim.y) {
    for (int i = threadIdx.x; i < nea; i += blockDim.x)
      if (filter_vertex_edge(rcut_inc, vertices_b, body_b.homothety_, j, &shpb, vertices_a, body_a.homothety_, i,
                             &shpa))
        count5++;
    for (int i = threadIdx.x; i < nfa; i += blockDim.x)
      if (filter_vertex_face(rcut_inc, vertices_b, body_b.homothety_, j, &shpb, vertices_a, body_a.homothety_, i,
                             &shpa))
        count6++;
  }

  // BlockScan for prefix per type
  int counts[4] = {count1, count2 + count5, count3 + count6, count4};
  int prefix[4];
  InteractionTypePerCellCounter sdata = prefix_data[idx];
  for (int t = 0; t < 4; t++) {
    BlockScan(temp_storage).ExclusiveSum(counts[t], prefix[t]);
    __syncthreads();
    prefix[t] += sdata[t];
  }

  // Prepare interaction item
  PlaceholderInteraction item = {};
  item.pair_.pi_.id_ = body_a.id_;
  item.pair_.pi_.cell_ = cell_a;
  item.pair_.pi_.p_ = pa;
  item.pair_.pj_.id_ = body_b.id_;
  item.pair_.pj_.cell_ = cell_b;
  item.pair_.pj_.p_ = pb;
  item.pair_.ghost_ = ghost_flag;
  item.pair_.swap_ = false;

  // Fill pass A→B (skip loops if no interactions)
  if (count1 > 0 || count2 > 0 || count3 > 0) {
    for (int i = threadIdx.y; i < nva; i += blockDim.y) {
      if (count1 > 0) {
        for (int j = threadIdx.x; j < nvb; j += blockDim.x) {
          if (filter_vertex_vertex(rcut_inc, vertices_a, body_a.homothety_, i, &shpa, vertices_b, body_b.homothety_, j,
                                   &shpb)) {
            item.pair_.pi_.sub_ = i;
            item.pair_.pj_.sub_ = j;
            item.pair_.type_ = InteractionTypeId::VertexVertex;
            item.pair_.swap_ = false;
            interactions[InteractionTypeId::VertexVertex].set(prefix[0]++, item);
          }
        }
      }
      if (count2 > 0) {
        for (int j = threadIdx.x; j < neb; j += blockDim.x) {
          if (filter_vertex_edge(rcut_inc, vertices_a, body_a.homothety_, i, &shpa, vertices_b, body_b.homothety_, j,
                                 &shpb)) {
            item.pair_.pi_.sub_ = i;
            item.pair_.pj_.sub_ = j;
            item.pair_.type_ = InteractionTypeId::VertexEdge;
            item.pair_.swap_ = false;
            interactions[InteractionTypeId::VertexEdge].set(prefix[1]++, item);
          }
        }
      }
      if (count3 > 0) {
        for (int j = threadIdx.x; j < nfb; j += blockDim.x) {
          if (filter_vertex_face(rcut_inc, vertices_a, body_a.homothety_, i, &shpa, vertices_b, body_b.homothety_, j,
                                 &shpb)) {
            item.pair_.pi_.sub_ = i;
            item.pair_.pj_.sub_ = j;
            item.pair_.type_ = InteractionTypeId::VertexFace;
            item.pair_.swap_ = false;
            interactions[InteractionTypeId::VertexFace].set(prefix[2]++, item);
          }
        }
      }
    }
  }

  // EE
  if (count4 > 0) {
    for (int i = threadIdx.y; i < nea; i += blockDim.y) {
      for (int j = threadIdx.x; j < neb; j += blockDim.x) {
        if (filter_edge_edge(rcut_inc, vertices_a, body_a.homothety_, i, &shpa, vertices_b, body_b.homothety_, j,
                             &shpb)) {
          item.pair_.pi_.sub_ = i;
          item.pair_.pj_.sub_ = j;
          item.pair_.type_ = InteractionTypeId::EdgeEdge;
          item.pair_.swap_ = false;
          interactions[InteractionTypeId::EdgeEdge].set(prefix[3]++, item);
        }
      }
    }
  }

  // Swap for B→A
  if (count5 > 0 || count6 > 0) {
    gpu_swap(item.pair_.pi_.id_, item.pair_.pj_.id_);
    gpu_swap(item.pair_.pi_.cell_, item.pair_.pj_.cell_);
    gpu_swap(item.pair_.pi_.p_, item.pair_.pj_.p_);
    item.pair_.swap_ = true;

    for (int j = threadIdx.y; j < nvb; j += blockDim.y) {
      if (count5 > 0) {
        for (int i = threadIdx.x; i < nea; i += blockDim.x) {
          if (filter_vertex_edge(rcut_inc, vertices_b, body_b.homothety_, j, &shpb, vertices_a, body_a.homothety_, i,
                                 &shpa)) {
            item.pair_.pi_.sub_ = j;
            item.pair_.pj_.sub_ = i;
            item.pair_.type_ = InteractionTypeId::VertexEdge;
            interactions[InteractionTypeId::VertexEdge].set(prefix[1]++, item);
          }
        }
      }
      if (count6 > 0) {
        for (int i = threadIdx.x; i < nfa; i += blockDim.x) {
          if (filter_vertex_face(rcut_inc, vertices_b, body_b.homothety_, j, &shpb, vertices_a, body_a.homothety_, i,
                                 &shpa)) {
            item.pair_.pi_.sub_ = j;
            item.pair_.pj_.sub_ = i;
            item.pair_.type_ = InteractionTypeId::VertexFace;
            interactions[InteractionTypeId::VertexFace].set(prefix[2]++, item);
          }
        }
      }
    }
  }
}

/// @brief Zeroes info_cell_pair.offset_/size_ for every cell pair, ahead of the atomic accumulation pass.
struct ResetCellPairCountsFunc {
  mutable onika::cuda::span<InteractionTypePerCellCounter> offset_;
  mutable onika::cuda::span<InteractionTypePerCellCounter> size_;

  ONIKA_HOST_DEVICE_FUNC inline void operator()(size_t cp) const {
    for (int t = 0; t < InteractionTypeId::NTypes; t++) {
      offset_[cp][t] = 0;
      size_[cp][t] = 0;
    }
  }
};

/// @brief Scatter-adds each particle pair's per-type interaction counts onto its owning cell pair.
struct AccumulateCellPairCountsFunc {
  onika::cuda::span<uint32_t> cell_pair_idx_;
  const InteractionTypePerCellCounter* __restrict__ count_per_pp_;
  mutable onika::cuda::span<InteractionTypePerCellCounter> size_;

  ONIKA_HOST_DEVICE_FUNC inline void operator()(size_t pp) const {
    const uint32_t cp = cell_pair_idx_[pp];
    for (int t = 0; t < InteractionTypeId::NTypesPP; t++) {
      ONIKA_CU_ATOMIC_ADD(size_[cp][t], count_per_pp_[pp][t]);
    }
  }
};

struct ExtractInteractionCountsFunc {
  onika::cuda::span<const InteractionTypePerCellCounter> counts_;
  // WARNING (TEMPORARY): todo remove mutable
  mutable onika::cuda::span<int> vv_;
  mutable onika::cuda::span<int> ve_;
  mutable onika::cuda::span<int> vf_;
  mutable onika::cuda::span<int> ee_;

  ONIKA_HOST_DEVICE_FUNC inline void operator()(size_t i) const {
    vv_[i] = counts_[i][0];
    ve_[i] = counts_[i][1];
    vf_[i] = counts_[i][2];
    ee_[i] = counts_[i][3];
  }
};

struct PackInteractionPrefixFunc {
  // WARNING (TEMPORARY): todo remove mutable
  mutable onika::cuda::span<InteractionTypePerCellCounter> prefix_;
  onika::cuda::span<const int> vv_;
  onika::cuda::span<const int> ve_;
  onika::cuda::span<const int> vf_;
  onika::cuda::span<const int> ee_;

  ONIKA_HOST_DEVICE_FUNC inline void operator()(size_t i) const {
    prefix_[i][0] = vv_[i];
    prefix_[i][1] = ve_[i];
    prefix_[i][2] = vf_[i];
    prefix_[i][3] = ee_[i];
  }
};

template <typename ExecCtx>
inline void reconstruct_cell_pair_offsets(
    ParticlePairStorage& pp_storage, InteractionTypePerCellCounter* count_per_pp, size_t num_particle_pairs,
    size_t num_cell_pairs, CellPairStorage& info_cell_pair,
    onika::memory::CudaMMVector<int> (&cp_type_counts)[InteractionTypeId::NTypesPP],
    onika::memory::CudaMMVector<int> (&cp_type_prefix)[InteractionTypeId::NTypesPP],
    onika::parallel::ParallelExecutionQueue& queue, int lane, onikaStream_t st, ExecCtx& exec_ctx,
    const onika::parallel::ParallelForOptions& opts) {
  using onika::cuda::make_const_span;
  using onika::cuda::make_span;
  using onika::cuda::span;
  using onika::parallel::flush;
  using onika::parallel::set_lane;
  auto cp_accessor = info_cell_pair.view();

  for (int t = 0; t < InteractionTypeId::NTypesPP; t++) {
    cp_type_counts[t].resize(num_cell_pairs);
    cp_type_prefix[t].resize(num_cell_pairs);
  }

  ResetCellPairCountsFunc reset_func{cp_accessor.offset_, cp_accessor.size_};
  AccumulateCellPairCountsFunc accumulate_func{to_span(pp_storage.cell_pair_idx_), count_per_pp, cp_accessor.size_};
  ExtractInteractionCountsFunc extract_func{
      span<const InteractionTypePerCellCounter>{cp_accessor.size_.data(), num_cell_pairs}, make_span(cp_type_counts[0]),
      make_span(cp_type_counts[1]), make_span(cp_type_counts[2]), make_span(cp_type_counts[3])};

  queue << set_lane(lane) << parallel_for(num_cell_pairs, reset_func, exec_ctx("nbh_gpu::reset_cell_pair_counts"), opts)
        << parallel_for(num_particle_pairs, accumulate_func, exec_ctx("nbh_gpu::accumulate_cell_pair_counts"), opts)
        << parallel_for(num_cell_pairs, extract_func, exec_ctx("nbh_gpu::extract_cell_pair_counts"), opts) << flush;

  // Same stream as `lane` above: stream-ordered after extract_func, no host sync needed here.
  for (int t = 0; t < InteractionTypeId::NTypesPP; t++) {
    exclusive_scan_device(cp_type_counts[t].data(), cp_type_prefix[t].data(), num_cell_pairs, st);
  }

  PackInteractionPrefixFunc pack_func{cp_accessor.offset_, make_const_span(cp_type_prefix[0]),
                                      make_const_span(cp_type_prefix[1]), make_const_span(cp_type_prefix[2]),
                                      make_const_span(cp_type_prefix[3])};
  queue << set_lane(lane) << parallel_for(num_cell_pairs, pack_func, exec_ctx("nbh_gpu::pack_cell_pair_prefix"), opts)
        << flush;
}

}  // namespace exaDEM

namespace onika {
namespace parallel {
template <>
struct ParallelForFunctorTraits<exaDEM::ResetCellPairCountsFunc> {
  static inline constexpr bool RequiresBlockSynchronousCall = false;
  static inline constexpr bool CudaCompatible = true;
};
template <>
struct ParallelForFunctorTraits<exaDEM::AccumulateCellPairCountsFunc> {
  static inline constexpr bool RequiresBlockSynchronousCall = false;
  static inline constexpr bool CudaCompatible = true;
};
template <>
struct ParallelForFunctorTraits<exaDEM::ExtractInteractionCountsFunc> {
  static inline constexpr bool RequiresBlockSynchronousCall = false;
  static inline constexpr bool CudaCompatible = true;
};
template <>
struct ParallelForFunctorTraits<exaDEM::PackInteractionPrefixFunc> {
  static inline constexpr bool RequiresBlockSynchronousCall = false;
  static inline constexpr bool CudaCompatible = true;
};
}  // namespace parallel
}  // namespace onika