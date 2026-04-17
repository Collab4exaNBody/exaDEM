#pragma once
#include <cub/cub.cuh>
#include <exaDEM/experimental/polyhedron/nbh_gpu/nbh_storage.hpp>
#include <exaDEM/experimental/polyhedron/nbh_gpu/nbh_gpu.hpp>

namespace exaDEM {

// Storage for particle pairs (output of neighbor search)
struct ParticlePairStorage {
  template<typename T> using VectorT = onika::memory::CudaMMVector<T>;
  VectorT<uint32_t> cell_i;
  VectorT<uint32_t> cell_j;
  VectorT<uint16_t> p_i;
  VectorT<uint16_t> p_j;
  VectorT<uint8_t>  ghost;       // ghost tag per pair
  VectorT<uint32_t> cell_pair_idx; // which cell-pair this came from
  size_t size = 0;

  void resize(size_t n) {
    cell_i.resize(n);
    cell_j.resize(n);
    p_i.resize(n);
    p_j.resize(n);
    ghost.resize(n);
    cell_pair_idx.resize(n);
    size = n;
  }
};

// ============================================================
// Stage 1: Count particle pairs per cell pair
// 1 block = 1 cell pair, threads iterate particle pairs
// ============================================================
template<int BLOCKX, int BLOCKY, typename TMPLC>
__global__ __launch_bounds__(64, 8)
void CountParticlePairsKernel(
    TMPLC cells,
    size_t* __restrict__ owner_cells,
    size_t* __restrict__ partner_cells,
    uint8_t* __restrict__ ghost_flags,
    double rcut_inc,
    const shape* __restrict__ shps,
    VertexField* __restrict__ vertex_fields,
    int* __restrict__ pair_counts,
    size_t num_cell_pairs)
{
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
    const auto& shpa = shps[body_a.type];

    AABB aabb_a = {body_a.r - body_a.radius - rcut_inc,
                   body_a.r + body_a.radius + rcut_inc};

    for (size_t pb = threadIdx.x; pb < nB; pb += blockDim.x) {
      auto body_b = load(cB, pb);

      //if (body_a.id >= body_b.id && ghost_flags[idx] == 0)
      //  continue;
      
	if (body_a.id >= body_b.id)
          continue;

      if (!is_inside_threshold(aabb_a, body_b.r, body_b.radius))
        continue;

      Vec3d r = body_b.r - body_a.r;
      double rmax = body_a.radius + body_b.radius + rcut_inc;
      if (exanb::dot(r, r) > rmax * rmax)
        continue;

      // OBB test
      const auto& shpb = shps[body_b.type];
      OBB obb_a = compute_obb(shpa.obb, body_a.r, body_a.quat, body_a.homothety);
      obb_a.enlarge(rcut_inc);
      OBB obb_b = compute_obb(shpb.obb, body_b.r, body_b.quat, body_b.homothety);
      if (obb_a.intersect(obb_b)) {
        count++;
      }
    }
  }

  int aggregate = BlockReduce(temp_storage).Sum(count);
  __syncthreads();
  if (threadIdx.x == 0 && threadIdx.y == 0)
    pair_counts[idx] = aggregate;
}

// ============================================================
// Stage 2: Fill particle pair arrays
// 1 block = 1 cell pair
// ============================================================
template<int BLOCKX, int BLOCKY, typename TMPLC>
__global__ __launch_bounds__(64, 8)
void FillParticlePairsKernel(
    TMPLC cells,
    size_t* __restrict__ owner_cells,
    size_t* __restrict__ partner_cells,
    uint8_t* __restrict__ ghost_flags,
    double rcut_inc,
    const shape* __restrict__ shps,
    VertexField* __restrict__ vertex_fields,
    int* __restrict__ pair_offsets,
    // output
    uint32_t* __restrict__ out_cell_i,
    uint32_t* __restrict__ out_cell_j,
    uint16_t* __restrict__ out_p_i,
    uint16_t* __restrict__ out_p_j,
    uint8_t*  __restrict__ out_ghost,
    uint32_t* __restrict__ out_cell_pair_idx,
    size_t num_cell_pairs)
{
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
    const auto& shpa = shps[body_a.type];
    AABB aabb_a = {body_a.r - body_a.radius - rcut_inc,
                   body_a.r + body_a.radius + rcut_inc};

    for (size_t pb = threadIdx.x; pb < nB; pb += blockDim.x) {
      auto body_b = load(cB, pb);
      //if (body_a.id >= body_b.id && ghost_flag == 0) continue;
       if (body_a.id >= body_b.id) continue;
      if (!is_inside_threshold(aabb_a, body_b.r, body_b.radius)) continue;
      Vec3d r = body_b.r - body_a.r;
      double rmax = body_a.radius + body_b.radius + rcut_inc;
      if (exanb::dot(r, r) > rmax * rmax) continue;
      const auto& shpb = shps[body_b.type];
      OBB obb_a = compute_obb(shpa.obb, body_a.r, body_a.quat, body_a.homothety);
      obb_a.enlarge(rcut_inc);
      OBB obb_b = compute_obb(shpb.obb, body_b.r, body_b.quat, body_b.homothety);
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
    const auto& shpa = shps[body_a.type];
    AABB aabb_a = {body_a.r - body_a.radius - rcut_inc,
                   body_a.r + body_a.radius + rcut_inc};

    for (size_t pb = threadIdx.x; pb < nB; pb += blockDim.x) {
      auto body_b = load(cB, pb);
      //if (body_a.id >= body_b.id && ghost_flag == 0) continue;
       if (body_a.id >= body_b.id) continue;
      if (!is_inside_threshold(aabb_a, body_b.r, body_b.radius)) continue;
      Vec3d r = body_b.r - body_a.r;
      double rmax = body_a.radius + body_b.radius + rcut_inc;
      if (exanb::dot(r, r) > rmax * rmax) continue;
      const auto& shpb = shps[body_b.type];
      OBB obb_a = compute_obb(shpa.obb, body_a.r, body_a.quat, body_a.homothety);
      obb_a.enlarge(rcut_inc);
      OBB obb_b = compute_obb(shpb.obb, body_b.r, body_b.quat, body_b.homothety);
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
template<int BLOCKX, int BLOCKY, typename TMPLC>
__global__ void CountInteractionsPPKernel(
    TMPLC cells,
    VertexField* __restrict__ vertex_fields,
    const shape* __restrict__ shps,
    double rcut_inc,
    uint32_t* __restrict__ pp_cell_i,
    uint32_t* __restrict__ pp_cell_j,
    uint16_t* __restrict__ pp_p_i,
    uint16_t* __restrict__ pp_p_j,
    InteractionTypePerCellCounter* __restrict__ count_data,
    size_t num_pairs)
{
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
  const auto& shpa = shps[body_a.type];
  const auto& shpb = shps[body_b.type];
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
  OBB obb_b = compute_obb(shpb.obb, body_b.r, body_b.quat, body_b.homothety);
  obb_b.enlarge(rcut_inc);

  for (int i = threadIdx.y; i < nva; i += blockDim.y) {
    for (int j = threadIdx.x; j < nvb; j += blockDim.x) {
      if (filter_vertex_vertex(rcut_inc, vertices_a, body_a.homothety, i, &shpa,
                               vertices_b, body_b.homothety, j, &shpb))
        countVV++;
    }
    for (int j = threadIdx.x; j < neb; j += blockDim.x) {
      if (filter_vertex_edge(rcut_inc, vertices_a, body_a.homothety, i, &shpa,
                             vertices_b, body_b.homothety, j, &shpb))
        countVE++;
    }
    for (int j = threadIdx.x; j < nfb; j += blockDim.x) {
      if (filter_vertex_face(rcut_inc, vertices_a, body_a.homothety, i, &shpa,
                             vertices_b, body_b.homothety, j, &shpb))
        countVF++;
    }
  }

  // Edge-Edge
  for (int i = threadIdx.y; i < nea; i += blockDim.y) {
    for (int j = threadIdx.x; j < neb; j += blockDim.x) {
      if (filter_edge_edge(rcut_inc, vertices_a, body_a.homothety, i, &shpa,
                           vertices_b, body_b.homothety, j, &shpb))
        countEE++;
    }
  }

  // B→A: reverse VE, VF
  for (int j = threadIdx.y; j < nvb; j += blockDim.y) {
    for (int i = threadIdx.x; i < nea; i += blockDim.x) {
      if (filter_vertex_edge(rcut_inc, vertices_b, body_b.homothety, j, &shpb,
                             vertices_a, body_a.homothety, i, &shpa))
        countVE++;
    }
    for (int i = threadIdx.x; i < nfa; i += blockDim.x) {
      if (filter_vertex_face(rcut_inc, vertices_b, body_b.homothety, j, &shpb,
                             vertices_a, body_a.homothety, i, &shpa))
        countVF++;
    }
  }

  // Block reduce
  int types[4] = {countVV, countVE, countVF, countEE};
  for (int t = 0; t < 4; t++) {
    int agg = BlockReduce(temp_storage).Sum(types[t]);
    __syncthreads();
    if (threadIdx.x == 0 && threadIdx.y == 0)
      count_data[idx][t] = agg;
  }
}

// ============================================================
// Stage 4: Fill Classifier per particle pair
// 1 block = 1 particle pair (PCCP)
// ============================================================
template<int BLOCKX, int BLOCKY, typename TMPLC>
__global__ __launch_bounds__(64, 10) 
void FillInteractionsPPKernel(
    TMPLC cells,
    VertexField* __restrict__ vertex_fields,
    const shape* __restrict__ shps,
    double rcut_inc,
    uint32_t* __restrict__ pp_cell_i,
    uint32_t* __restrict__ pp_cell_j,
    uint16_t* __restrict__ pp_p_i,
    uint16_t* __restrict__ pp_p_j,
    uint8_t*  __restrict__ pp_ghost,
    InteractionTypePerCellCounter* __restrict__ prefix_data,
    InteractionParticleAccessor interactions,
    size_t num_pairs)
{
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
  const auto& shpa = shps[body_a.type];
  const auto& shpb = shps[body_b.type];
  ParticleVertexView vertices_a = {pa, vertex_fields[cell_a]};
  ParticleVertexView vertices_b = {pb, vertex_fields[cell_b]};

  const int nva = shpa.get_number_of_vertices();
  const int nea = shpa.get_number_of_edges();
  const int nfa = shpa.get_number_of_faces();
  const int nvb = shpb.get_number_of_vertices();
  const int neb = shpb.get_number_of_edges();
  const int nfb = shpb.get_number_of_faces();

// Count pass with directional counters
  int count1 = 0; // VV A→B
  int count2 = 0; // VE A→B
  int count3 = 0; // VF A→B
  int count4 = 0; // EE
  int count5 = 0; // VE B→A
  int count6 = 0; // VF B→A

  for (int i = threadIdx.y; i < nva; i += blockDim.y) {
    for (int j = threadIdx.x; j < nvb; j += blockDim.x)
      if (filter_vertex_vertex(rcut_inc, vertices_a, body_a.homothety, i, &shpa,
                               vertices_b, body_b.homothety, j, &shpb)) count1++;
    for (int j = threadIdx.x; j < neb; j += blockDim.x)
      if (filter_vertex_edge(rcut_inc, vertices_a, body_a.homothety, i, &shpa,
                             vertices_b, body_b.homothety, j, &shpb)) count2++;
    for (int j = threadIdx.x; j < nfb; j += blockDim.x)
      if (filter_vertex_face(rcut_inc, vertices_a, body_a.homothety, i, &shpa,
                             vertices_b, body_b.homothety, j, &shpb)) count3++;
  }
  for (int i = threadIdx.y; i < nea; i += blockDim.y)
    for (int j = threadIdx.x; j < neb; j += blockDim.x)
      if (filter_edge_edge(rcut_inc, vertices_a, body_a.homothety, i, &shpa,
                           vertices_b, body_b.homothety, j, &shpb)) count4++;
  for (int j = threadIdx.y; j < nvb; j += blockDim.y) {
    for (int i = threadIdx.x; i < nea; i += blockDim.x)
      if (filter_vertex_edge(rcut_inc, vertices_b, body_b.homothety, j, &shpb,
                             vertices_a, body_a.homothety, i, &shpa)) count5++;
    for (int i = threadIdx.x; i < nfa; i += blockDim.x)
      if (filter_vertex_face(rcut_inc, vertices_b, body_b.homothety, j, &shpb,
                             vertices_a, body_a.homothety, i, &shpa)) count6++;
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
  item.pair.pi.id = body_a.id;
  item.pair.pi.cell = cell_a;
  item.pair.pi.p = pa;
  item.pair.pj.id = body_b.id;
  item.pair.pj.cell = cell_b;
  item.pair.pj.p = pb;
  item.pair.ghost = ghost_flag;
  item.pair.swap = false;

  // Fill pass A→B (skip loops if no interactions)
  if (count1 > 0 || count2 > 0 || count3 > 0) {
    for (int i = threadIdx.y; i < nva; i += blockDim.y) {
      if (count1 > 0) {
        for (int j = threadIdx.x; j < nvb; j += blockDim.x) {
          if (filter_vertex_vertex(rcut_inc, vertices_a, body_a.homothety, i, &shpa,
                                   vertices_b, body_b.homothety, j, &shpb)) {
            item.pair.pi.sub = i;
            item.pair.pj.sub = j;
            item.pair.type = InteractionTypeId::VertexVertex;
            item.pair.swap = false;
            interactions[InteractionTypeId::VertexVertex].set(prefix[0]++, item);
          }
        }
      }
      if (count2 > 0) {
        for (int j = threadIdx.x; j < neb; j += blockDim.x) {
          if (filter_vertex_edge(rcut_inc, vertices_a, body_a.homothety, i, &shpa,
                                 vertices_b, body_b.homothety, j, &shpb)) {
            item.pair.pi.sub = i;
            item.pair.pj.sub = j;
            item.pair.type = InteractionTypeId::VertexEdge;
            item.pair.swap = false;
            interactions[InteractionTypeId::VertexEdge].set(prefix[1]++, item);
          }
        }
      }
      if (count3 > 0) {
        for (int j = threadIdx.x; j < nfb; j += blockDim.x) {
          if (filter_vertex_face(rcut_inc, vertices_a, body_a.homothety, i, &shpa,
                                 vertices_b, body_b.homothety, j, &shpb)) {
            item.pair.pi.sub = i;
            item.pair.pj.sub = j;
            item.pair.type = InteractionTypeId::VertexFace;
            item.pair.swap = false;
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
        if (filter_edge_edge(rcut_inc, vertices_a, body_a.homothety, i, &shpa,
                             vertices_b, body_b.homothety, j, &shpb)) {
          item.pair.pi.sub = i;
          item.pair.pj.sub = j;
          item.pair.type = InteractionTypeId::EdgeEdge;
          item.pair.swap = false;
          interactions[InteractionTypeId::EdgeEdge].set(prefix[3]++, item);
        }
      }
    }
  }

  // Swap for B→A
  if (count5 > 0 || count6 > 0) {
    gpu_swap(item.pair.pi.id, item.pair.pj.id);
    gpu_swap(item.pair.pi.cell, item.pair.pj.cell);
    gpu_swap(item.pair.pi.p, item.pair.pj.p);
    item.pair.swap = true;

    for (int j = threadIdx.y; j < nvb; j += blockDim.y) {
      if (count5 > 0) {
        for (int i = threadIdx.x; i < nea; i += blockDim.x) {
          if (filter_vertex_edge(rcut_inc, vertices_b, body_b.homothety, j, &shpb,
                                 vertices_a, body_a.homothety, i, &shpa)) {
            item.pair.pi.sub = j;
            item.pair.pj.sub = i;
            item.pair.type = InteractionTypeId::VertexEdge;
            interactions[InteractionTypeId::VertexEdge].set(prefix[1]++, item);
          }
        }
      }
      if (count6 > 0) {
        for (int i = threadIdx.x; i < nfa; i += blockDim.x) {
          if (filter_vertex_face(rcut_inc, vertices_b, body_b.homothety, j, &shpb,
                                 vertices_a, body_a.homothety, i, &shpa)) {
            item.pair.pi.sub = j;
            item.pair.pj.sub = i;
            item.pair.type = InteractionTypeId::VertexFace;
            interactions[InteractionTypeId::VertexFace].set(prefix[2]++, item);
          }
        }
      }
    }
  }
}
// ============================================================
// Reconstruct per-cell-pair offsets from particle-pair data
// Needed for UpdateHistoryFunc compatibility
// ============================================================
/*inline void reconstruct_cell_pair_offsets(
    ParticlePairStorage& pp_storage,
    InteractionTypePerCellCounter* count_per_pp, // [num_particle_pairs]
    size_t num_particle_pairs,
    size_t num_cell_pairs,
    NbhCellStorage& info_cell_pair)
{
  // Reset offsets and sizes
  for (size_t cp = 0; cp < num_cell_pairs; cp++) {
    for (int t = 0; t < InteractionTypeId::NTypes; t++) {
      info_cell_pair.offset[cp][t] = 0;
      info_cell_pair.size[cp][t] = 0;
    }
  }

  // Accumulate: for each particle pair, add its counts to its cell pair
  for (size_t pp = 0; pp < num_particle_pairs; pp++) {
    uint32_t cp = pp_storage.cell_pair_idx[pp];
    for (int t = 0; t < 4; t++) {
      info_cell_pair.size[cp][t] += count_per_pp[pp][t];
    }
  }

  // Prefix sum per type across cell pairs to get offsets
  InteractionTypePerCellCounter running;
  for (int t = 0; t < InteractionTypeId::NTypes; t++) running[t] = 0;

  for (size_t cp = 0; cp < num_cell_pairs; cp++) {
    for (int t = 0; t < InteractionTypeId::NTypes; t++) {
      info_cell_pair.offset[cp][t] = running[t];
      running[t] += info_cell_pair.size[cp][t];
    }
  }
}*/
inline void reconstruct_cell_pair_offsets(
    ParticlePairStorage& pp_storage,
    InteractionTypePerCellCounter* count_per_pp,
    size_t num_particle_pairs,
    size_t num_cell_pairs,
    NbhCellStorage& info_cell_pair)
{
  // Reset (parallel)
  #pragma omp parallel for
  for (size_t cp = 0; cp < num_cell_pairs; cp++) {
    for (int t = 0; t < InteractionTypeId::NTypes; t++) {
      info_cell_pair.offset[cp][t] = 0;
      info_cell_pair.size[cp][t] = 0;
    }
  }

  // Accumulate (parallel with atomics)
  #pragma omp parallel for
  for (size_t pp = 0; pp < num_particle_pairs; pp++) {
    uint32_t cp = pp_storage.cell_pair_idx[pp];
    for (int t = 0; t < 4; t++) {
      #pragma omp atomic
      info_cell_pair.size[cp][t] += count_per_pp[pp][t];
    }
  }

  // Prefix sum (sequential, but only over cell pairs ~few thousands)
  InteractionTypePerCellCounter running;
  for (int t = 0; t < InteractionTypeId::NTypes; t++) running[t] = 0;

  for (size_t cp = 0; cp < num_cell_pairs; cp++) {
    for (int t = 0; t < InteractionTypeId::NTypes; t++) {
      info_cell_pair.offset[cp][t] = running[t];
      running[t] += info_cell_pair.size[cp][t];
    }
  }
}

// ============================================================
// Helper kernels for GPU prefix sum
// ============================================================
__global__
void ExtractInteractionCounts(
    const InteractionTypePerCellCounter* __restrict__ counts,
    int* __restrict__ vv, int* __restrict__ ve,
    int* __restrict__ vf, int* __restrict__ ee,
    size_t n)
{
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  vv[i] = counts[i][0];
  ve[i] = counts[i][1];
  vf[i] = counts[i][2];
  ee[i] = counts[i][3];
}

__global__
void PackInteractionPrefix(
    InteractionTypePerCellCounter* __restrict__ prefix,
    const int* __restrict__ vv, const int* __restrict__ ve,
    const int* __restrict__ vf, const int* __restrict__ ee,
    size_t n)
{
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  prefix[i][0] = vv[i];
  prefix[i][1] = ve[i];
  prefix[i][2] = vf[i];
  prefix[i][3] = ee[i];
}

} // namespace exaDEM
