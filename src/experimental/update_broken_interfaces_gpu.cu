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

#ifdef ONIKA_CUDA_VERSION

#include <exanb/core/grid.h>
#include <exanb/core/make_grid_variant_operator.h>
#include <mpi.h>
#include <onika/cuda/cuda_context.h>
#include <onika/scg/operator.h>
#include <onika/scg/operator_factory.h>
#include <onika/scg/operator_slot.h>

#include <algorithm>
#include <exaDEM/classifier/classifier.hpp>
#include <exaDEM/experimental/polyhedron/nbh_gpu/nbh_gpu_pccp.hpp>
#include <exaDEM/experimental/polyhedron/nbh_gpu/nbh_utils.hpp>
#include <exaDEM/forcefield/contact_force_accessor.hpp>
#include <exaDEM/forcefield/contact_parameters.hpp>
#include <exaDEM/forcefield/multimat_parameters.hpp>
#include <exaDEM/interface/interface.hpp>
#include <exaDEM/shapes.hpp>
#include <numeric>
#include <vector>

namespace exaDEM {
using namespace exanb;

namespace broken_gpu {

constexpr int kThreads = 128;
constexpr int kBlockX = 8;
constexpr int kBlockY = 8;

inline int nblocks(size_t n) { return static_cast<int>((n + kThreads - 1) / kThreads); }

/** @brief Raw device buffer (no host initialization, unlike CudaMMVector). */
template <typename T>
struct DeviceBuffer {
  T* ptr_ = nullptr;
  size_t capacity_ = 0;

  DeviceBuffer() = default;
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  ~DeviceBuffer() {
    if (ptr_) cudaFree(ptr_);
  }
  T* reserve(size_t n) {
    if (n > capacity_) {
      if (ptr_) ONIKA_CU_CHECK_ERRORS(cudaFree(ptr_));
      capacity_ = std::max(n, 2 * capacity_);
      ONIKA_CU_CHECK_ERRORS(cudaMalloc(&ptr_, capacity_ * sizeof(T)));
    }
    return ptr_;
  }
  T* data() const { return ptr_; }
};

template <typename T>
inline T read_device(const T* ptr, onikaStream_t st) {
  T value;
  ONIKA_CU_CHECK_ERRORS(cudaMemcpyAsync(&value, ptr, sizeof(T), cudaMemcpyDeviceToHost, st));
  ONIKA_CU_STREAM_SYNCHRONIZE(st);
  return value;
}

/** @brief Broken bond of an OwnerGhost interface, sent to the PartnerGhost copies (48 bytes). */
struct BrokenGhostBond {
  uint64_t id_i;
  uint64_t id_j;
  uint32_t sub_i;
  uint32_t sub_j;
  Vec3d friction;
};
static_assert(sizeof(BrokenGhostBond) == 48);

/** @brief Raw pointers on the InnerBond fields used by this pipeline. */
struct BondFields {
  const uint64_t* id_i;
  const uint64_t* id_j;
  const uint32_t* cell_i;
  const uint32_t* cell_j;
  const uint16_t* p_i;
  const uint16_t* p_j;
  const uint32_t* sub_i;
  const uint32_t* sub_j;
  const uint8_t* swap;
  const uint8_t* ghost;
  Vec3d* friction;
  uint8_t* unbroken;

  ONIKA_HOST_DEVICE_FUNC inline void owner_partner(size_t b, uint32_t& cell_a, uint16_t& p_a, uint32_t& cell_b,
                                                   uint16_t& p_b) const {
    if (!swap[b]) {
      cell_a = cell_i[b], p_a = p_i[b], cell_b = cell_j[b], p_b = p_j[b];
    } else {
      cell_a = cell_j[b], p_a = p_j[b], cell_b = cell_i[b], p_b = p_i[b];
    }
  }
};

inline BondFields make_bond_fields(Classifier::WaveIB::View& v) {
  return {v[attr::id_i].data(),  v[attr::id_j].data(),  v[attr::cell_i].data(), v[attr::cell_j].data(),
          v[attr::p_i].data(),   v[attr::p_j].data(),   v[attr::sub_i].data(),  v[attr::sub_j].data(),
          v[attr::swap].data(),  v[attr::ghost].data(), v[attr::friction].data(), v[attr::unbroken].data()};
}

/** @brief Where the friction history of a broken pair comes from: local bonds or received ghost bonds. */
struct PairHistoryRef {
  uint32_t begin;
  uint32_t size;
  uint8_t from_ghost;
};

/** @brief Device compaction: out[k] = i for every i < n such that pred(i). Returns the count. */
struct Compactor {
  DeviceBuffer<uint32_t> flags_;
  DeviceBuffer<uint32_t> scan_;

  template <typename PredT>
  size_t run(size_t n, PredT pred, DeviceBuffer<uint32_t>& out, onikaStream_t st);
};

template <typename PredT>
__global__ void flag_kernel(size_t n, PredT pred, uint32_t* flags) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
    flags[i] = pred(i) ? 1 : 0;
  }
}

__global__ void scatter_kernel(size_t n, const uint32_t* flags, const uint32_t* scan, uint32_t* out) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
    if (flags[i]) out[scan[i]] = i;
  }
}

template <typename PredT>
size_t Compactor::run(size_t n, PredT pred, DeviceBuffer<uint32_t>& out, onikaStream_t st) {
  if (n == 0) return 0;
  uint32_t* flags = flags_.reserve(n);
  uint32_t* scan = scan_.reserve(n);
  flag_kernel<<<nblocks(n), kThreads, 0, st>>>(n, pred, flags);
  ONIKA_CU_CHECK_ERRORS(cudaGetLastError());  // exclusive_scan_device clears pending errors
  exclusive_scan_device(flags, scan, n, st);
  const size_t count = read_device(scan + n - 1, st) + read_device(flags + n - 1, st);
  if (count > 0) {
    scatter_kernel<<<nblocks(n), kThreads, 0, st>>>(n, flags, scan, out.reserve(count));
  }
  return count;
}

struct IsBroken {
  const uint8_t* break_interface;
  __device__ inline bool operator()(size_t i) const { return break_interface[i] != 0; }
};

struct IsOwnerGhost {
  const uint32_t* broken;
  const Interface* interfaces;
  const uint8_t* ghost;
  __device__ inline bool operator()(size_t i) const {
    return ghost[interfaces[broken[i]].loc_] == InteractionPair::OwnerGhost;
  }
};

struct IsMatched {
  const int* first;
  __device__ inline bool operator()(size_t i) const { return first[i] >= 0; }
};

struct IsKept {
  const uint8_t* break_interface;
  __device__ inline bool operator()(size_t i) const { return break_interface[i] == 0; }
};

// Same rule as broken_interface_friction_updater.
template <typename CellsT, typename CpT>
__global__ void friction_update_kernel(size_t k, const uint32_t* broken, const Interface* interfaces, BondFields b,
                                       CellsT cells, CpT cp, const double* dn, const Vec3d* fn, const Vec3d* ft) {
  for (size_t t = blockIdx.x * blockDim.x + threadIdx.x; t < k; t += gridDim.x * blockDim.x) {
    const Interface itf = interfaces[broken[t]];
    const size_t o = itf.loc_;
    const auto type_a = cells[b.cell_i[o]][field::group][b.p_i[o]];
    const auto type_b = cells[b.cell_j[o]][field::group][b.p_j[o]];
    const double mu = cp(type_a, type_b).mu_;
    for (size_t j = o; j < o + itf.size_; j++) {
      b.unbroken[j] = false;
      const double ftn = norm(ft[j]);
      const double threshold = mu * norm(fn[j]);
      if (ftn > threshold && ftn > 0) b.friction[j] = ft[j] * threshold / ftn;
      if (dn[j] > 0) b.friction[j] = Vec3d{0, 0, 0};
    }
  }
}

__global__ void ghost_sizes_kernel(size_t m, const uint32_t* ghost_sel, const uint32_t* broken,
                                   const Interface* interfaces, uint32_t* sizes) {
  for (size_t t = blockIdx.x * blockDim.x + threadIdx.x; t < m; t += gridDim.x * blockDim.x) {
    sizes[t] = interfaces[broken[ghost_sel[t]]].size_;
  }
}

__global__ void pack_ghost_kernel(size_t m, const uint32_t* ghost_sel, const uint32_t* broken,
                                  const Interface* interfaces, const uint32_t* offsets, BondFields b,
                                  BrokenGhostBond* send) {
  for (size_t t = blockIdx.x * blockDim.x + threadIdx.x; t < m; t += gridDim.x * blockDim.x) {
    const Interface itf = interfaces[broken[ghost_sel[t]]];
    BrokenGhostBond* out = send + offsets[t];
    for (size_t j = 0; j < itf.size_; j++) {
      const size_t s = itf.loc_ + j;
      out[j] = {b.id_i[s], b.id_j[s], b.sub_i[s], b.sub_j[s], b.friction[s]};
    }
  }
}

ONIKA_HOST_DEVICE_FUNC inline bool key_less(const BrokenGhostBond& a, uint64_t id_i, uint64_t id_j) {
  return a.id_i < id_i || (a.id_i == id_i && a.id_j < id_j);
}

// recv is sorted by (id_i, id_j): first[p] = first received bond of the PartnerGhost interface p, or -1.
__global__ void match_partner_kernel(size_t n_partner, const Interface* partner, BondFields b,
                                     const BrokenGhostBond* recv, size_t n_recv, int* first, uint32_t* count) {
  for (size_t p = blockIdx.x * blockDim.x + threadIdx.x; p < n_partner; p += gridDim.x * blockDim.x) {
    first[p] = -1;
    const Interface itf = partner[p];
    const size_t o = itf.loc_;
    if (!b.unbroken[o]) continue;
    const uint64_t id_i = b.id_i[o], id_j = b.id_j[o];
    size_t lo = 0, hi = n_recv;
    while (lo < hi) {
      const size_t mid = (lo + hi) / 2;
      if (key_less(recv[mid], id_i, id_j)) lo = mid + 1;
      else hi = mid;
    }
    if (lo == n_recv || recv[lo].id_i != id_i || recv[lo].id_j != id_j) continue;
    size_t end = lo;
    while (end < n_recv && recv[end].id_i == id_i && recv[end].id_j == id_j) end++;
    first[p] = lo;
    count[p] = end - lo;
    for (size_t j = o; j < o + itf.size_; j++) {
      b.unbroken[j] = false;
      b.friction[j] = Vec3d{0, 0, 0};
    }
  }
}

struct PairArrays {
  uint32_t* cell_i;
  uint32_t* cell_j;
  uint16_t* p_i;
  uint16_t* p_j;
  uint8_t* ghost;
  PairHistoryRef* hist;
};

__global__ void local_pairs_kernel(size_t k, const uint32_t* broken, const Interface* interfaces, BondFields b,
                                   PairArrays pairs) {
  for (size_t t = blockIdx.x * blockDim.x + threadIdx.x; t < k; t += gridDim.x * blockDim.x) {
    const Interface itf = interfaces[broken[t]];
    const size_t o = itf.loc_;
    b.owner_partner(o, pairs.cell_i[t], pairs.p_i[t], pairs.cell_j[t], pairs.p_j[t]);
    pairs.ghost[t] = b.ghost[o];
    pairs.hist[t] = {static_cast<uint32_t>(o), static_cast<uint32_t>(itf.size_), 0};
  }
}

__global__ void partner_pairs_kernel(size_t q, size_t shift, const uint32_t* matched, const Interface* partner,
                                     const int* first, const uint32_t* count, BondFields b, PairArrays pairs) {
  for (size_t t = blockIdx.x * blockDim.x + threadIdx.x; t < q; t += gridDim.x * blockDim.x) {
    const uint32_t p = matched[t];
    const size_t o = partner[p].loc_;
    const size_t d = shift + t;
    b.owner_partner(o, pairs.cell_i[d], pairs.p_i[d], pairs.cell_j[d], pairs.p_j[d]);
    pairs.ghost[d] = InteractionPair::PartnerGhost;
    pairs.hist[d] = {static_cast<uint32_t>(first[p]), count[p], 1};
  }
}

__global__ void split_counts_kernel(size_t n, const InteractionTypePerCellCounter* counts, int* vv, int* ve, int* vf,
                                    int* ee) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
    vv[i] = counts[i][0];
    ve[i] = counts[i][1];
    vf[i] = counts[i][2];
    ee[i] = counts[i][3];
  }
}

struct TypeBase {
  int v[InteractionTypeId::NTypesPP];
};

__global__ void pack_prefix_kernel(size_t n, InteractionTypePerCellCounter* prefix, const int* vv, const int* ve,
                                   const int* vf, const int* ee, TypeBase base) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
    prefix[i][0] = vv[i] + base.v[0];
    prefix[i][1] = ve[i] + base.v[1];
    prefix[i][2] = vf[i] + base.v[2];
    prefix[i][3] = ee[i] + base.v[3];
  }
}

// Same matching rules as add_broken_interface_contacts (local) and add_broken_interface_ghost_contacts (ghost).
struct BrokenPairHistory {
  const PairHistoryRef* hist;
  BondFields b;
  const BrokenGhostBond* recv;

  ONIKA_HOST_DEVICE_FUNC inline void operator()(size_t pair, PlaceholderInteraction& item) const {
    const PairHistoryRef r = hist[pair];
    const auto& pi = item.pair_.pi_;
    const auto& pj = item.pair_.pj_;
    for (uint32_t j = r.begin; j < r.begin + r.size; j++) {
      if (r.from_ghost) {
        if (recv[j].sub_i == pi.sub_ && recv[j].sub_j == pj.sub_) {
          item.as<Interaction>()[attr::friction] = recv[j].friction;
          return;
        }
      } else if (!b.swap[j] && b.id_i[j] == pi.id_ && b.id_j[j] == pj.id_ && b.sub_i[j] == pi.sub_ &&
                 b.sub_j[j] == pj.sub_) {
        item.as<Interaction>()[attr::friction] = b.friction[j];
        return;
      }
    }
  }
};

__global__ void clear_friction_kernel(size_t k, const uint32_t* broken, const Interface* interfaces, Vec3d* friction) {
  for (size_t t = blockIdx.x * blockDim.x + threadIdx.x; t < k; t += gridDim.x * blockDim.x) {
    const Interface itf = interfaces[broken[t]];
    for (size_t j = itf.loc_; j < itf.loc_ + itf.size_; j++) friction[j] = Vec3d{0, 0, 0};
  }
}

__global__ void gather_interfaces_kernel(size_t n, const uint32_t* kept, const Interface* in, Interface* out) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) out[i] = in[kept[i]];
}

struct Scratch {
  Compactor compactor_;
  DeviceBuffer<uint32_t> broken_;     // broken local interfaces
  DeviceBuffer<uint32_t> ghost_sel_;  // indices (in broken_) of the OwnerGhost ones
  DeviceBuffer<uint32_t> sizes_;
  DeviceBuffer<uint32_t> offsets_;
  DeviceBuffer<BrokenGhostBond> send_;
  DeviceBuffer<BrokenGhostBond> recv_;
  DeviceBuffer<int> first_;
  DeviceBuffer<uint32_t> count_;
  DeviceBuffer<uint32_t> matched_;
  DeviceBuffer<uint32_t> cell_i_, cell_j_;
  DeviceBuffer<uint16_t> p_i_, p_j_;
  DeviceBuffer<uint8_t> ghost_;
  DeviceBuffer<PairHistoryRef> hist_;
  DeviceBuffer<InteractionTypePerCellCounter> counts_, prefix_;
  DeviceBuffer<int> type_counts_[InteractionTypeId::NTypesPP];
  DeviceBuffer<int> type_prefix_[InteractionTypeId::NTypesPP];
  DeviceBuffer<uint32_t> kept_;
  DeviceBuffer<Interface> interfaces_;
  std::vector<BrokenGhostBond> host_send_, host_recv_;
};
}  // namespace broken_gpu

template <typename GridT, class = AssertGridHasFields<GridT>>
class UpdateBrokenInterfacesGPU : public OperatorNode {
  ADD_SLOT(MPI_Comm, mpi, INPUT, MPI_COMM_WORLD);
  ADD_SLOT(GridT, grid, INPUT, REQUIRED);
  ADD_SLOT(CellVertexField, cvf, INPUT, REQUIRED, DocString{"Store vertex positions for every polyhedron"});
  ADD_SLOT(shapes, shapes_collection, INPUT, REQUIRED, DocString{"Collection of shapes"});
  ADD_SLOT(double, rcut_inc, INPUT, REQUIRED,
           DocString{"value added to the search distance to update neighbor list less frequently. in physical space"});
  ADD_SLOT(MultiMatParamsT<ContactParams>, multimat_cp, INPUT, REQUIRED,
           DocString{"List of contact parameters for simulations with multiple materials"});
  ADD_SLOT(Classifier, ic, INPUT_OUTPUT, DocString{"Interaction lists classified according to their types"});
  ADD_SLOT(InterfaceManager, im, INPUT_OUTPUT, DocString{"Interfaces built from the InnerBond interactions"});
  ADD_SLOT(bool, enable_broken_interface_contacts, INPUT, true,
           DocString{"If false, every broken interface requests a neighbor-list rebuild (previous behavior)."});
  ADD_SLOT(bool, broken_interface, INPUT_OUTPUT, REQUIRED,
           DocString{"In: an interface is broken. Out: a neighbor-list rebuild is required."});
  ADD_SLOT(broken_gpu::Scratch, scratch, PRIVATE, DocString{"Persistent device buffers"});

 public:
  inline std::string documentation() const final {
    return R"EOF(
        GPU version of broken_interface_friction_updater + add_broken_interface_ghost_contacts +
        add_broken_interface_contacts. Must run after apply_interface_fracture_criterion.

        Broken interfaces are compacted on the device; only the counts and the bonds of the broken
        OwnerGhost interfaces (48 bytes per bond) go through the host for the MPI exchange.

        YAML example:

          - update_broken_interfaces_gpu
      )EOF";
  }

  inline void execute() final {
    using namespace broken_gpu;
    if (!(*broken_interface)) {
      return;  // globally consistent
    }
    if (global_cuda_ctx() == nullptr || !global_cuda_ctx()->has_devices()) {
      color_log::error("update_broken_interfaces_gpu", "This operator only works on GPU.");
    }

    ONIKA_CU_DEVICE_SYNCHRONIZE();
    // Clear a non-sticky error left by an unchecked call (e.g. cudaMemPrefetchAsync unsupported on WSL).
    if (cudaError_t pending = cudaGetLastError(); pending != cudaSuccess) {
      ldbg << "update_broken_interfaces_gpu: clearing pending CUDA error: " << cudaGetErrorString(pending)
           << std::endl;
    }
    onikaStream_t st = global_cuda_ctx()->getThreadStream(0);
    Scratch& s = *scratch;
    InterfaceManager& interfaces = *im;
    const size_t n_interfaces = interfaces.size();
    auto ib_view = ic->get_sticked_interaction_wrapper();
    BondFields bonds = make_bond_fields(ib_view);
    auto [dn, cp_unused, fn, ft] = ic->contact_state(InteractionTypeId::InnerBondId);
    const auto cells = grid->cells();
    const Interface* itf = interfaces.data_.data();

    // 1. Compaction + friction update
    const size_t k = s.compactor_.run(n_interfaces, IsBroken{interfaces.break_interface_.data()}, s.broken_, st);
    if (k > 0) {
      const auto cp = multimat_cp->get_multimat_accessor();
      friction_update_kernel<<<nblocks(k), kThreads, 0, st>>>(k, s.broken_.data(), itf, bonds, cells, cp, dn, fn, ft);
    }

    *broken_interface = !(*enable_broken_interface_contacts);
    if (*broken_interface) {
      ONIKA_CU_STREAM_SYNCHRONIZE(st);
      return;  // rebuild
    }

    // 2. Pack the bonds of the broken OwnerGhost interfaces
    const size_t m = s.compactor_.run(k, IsOwnerGhost{s.broken_.data(), itf, bonds.ghost}, s.ghost_sel_, st);
    size_t n_send = 0;
    if (m > 0) {
      ghost_sizes_kernel<<<nblocks(m), kThreads, 0, st>>>(m, s.ghost_sel_.data(), s.broken_.data(), itf,
                                                           s.sizes_.reserve(m));
      ONIKA_CU_CHECK_ERRORS(cudaGetLastError());
      exclusive_scan_device(s.sizes_.data(), s.offsets_.reserve(m), m, st);
      n_send = read_device(s.offsets_.data() + m - 1, st) + read_device(s.sizes_.data() + m - 1, st);
      pack_ghost_kernel<<<nblocks(m), kThreads, 0, st>>>(m, s.ghost_sel_.data(), s.broken_.data(), itf,
                                                          s.offsets_.data(), bonds, s.send_.reserve(n_send));
      s.host_send_.resize(n_send);
      ONIKA_CU_CHECK_ERRORS(cudaMemcpyAsync(s.host_send_.data(), s.send_.data(), n_send * sizeof(BrokenGhostBond),
                                            cudaMemcpyDeviceToHost, st));
    }
    ONIKA_CU_STREAM_SYNCHRONIZE(st);

    // 3. MPI exchange (host staging)
    int nprocs = 1;
    MPI_Comm_size(*mpi, &nprocs);
    std::vector<int> counts(nprocs), displs(nprocs);
    int local_bytes = n_send * sizeof(BrokenGhostBond);
    MPI_Allgather(&local_bytes, 1, MPI_INT, counts.data(), 1, MPI_INT, *mpi);
    std::exclusive_scan(counts.begin(), counts.end(), displs.begin(), 0);
    const size_t n_recv = (displs.back() + counts.back()) / sizeof(BrokenGhostBond);
    size_t q = 0;
    const size_t n_partner = interfaces.partner_ghost_data_.size();
    if (n_recv > 0) {
      s.host_recv_.resize(n_recv);
      MPI_Allgatherv(s.host_send_.data(), local_bytes, MPI_BYTE, s.host_recv_.data(), counts.data(), displs.data(),
                     MPI_BYTE, *mpi);
      std::stable_sort(s.host_recv_.begin(), s.host_recv_.end(), [](const BrokenGhostBond& a, const BrokenGhostBond& b) {
        return key_less(a, b.id_i, b.id_j);
      });

      // 4. Match the PartnerGhost copies
      if (n_partner > 0) {
        ONIKA_CU_CHECK_ERRORS(cudaMemcpyAsync(s.recv_.reserve(n_recv), s.host_recv_.data(),
                                              n_recv * sizeof(BrokenGhostBond), cudaMemcpyHostToDevice, st));
        match_partner_kernel<<<nblocks(n_partner), kThreads, 0, st>>>(
            n_partner, interfaces.partner_ghost_data_.data(), bonds, s.recv_.data(), n_recv,
            s.first_.reserve(n_partner), s.count_.reserve(n_partner));
        q = s.compactor_.run(n_partner, IsMatched{s.first_.data()}, s.matched_, st);
      }
    }

    check_broken_ghost_balance(*mpi, m, q, "update_broken_interfaces_gpu");

    // 5. Contacts of the broken pairs (local first, then PartnerGhost)
    const size_t n_pairs = k + q;
    size_t n_contacts = 0;
    if (n_pairs > 0) {
      PairArrays pairs = {s.cell_i_.reserve(n_pairs), s.cell_j_.reserve(n_pairs), s.p_i_.reserve(n_pairs),
                          s.p_j_.reserve(n_pairs),    s.ghost_.reserve(n_pairs),  s.hist_.reserve(n_pairs)};
      if (k > 0) {
        local_pairs_kernel<<<nblocks(k), kThreads, 0, st>>>(k, s.broken_.data(), itf, bonds, pairs);
      }
      if (q > 0) {
        partner_pairs_kernel<<<nblocks(q), kThreads, 0, st>>>(q, k, s.matched_.data(),
                                                               interfaces.partner_ghost_data_.data(), s.first_.data(),
                                                               s.count_.data(), bonds, pairs);
      }

      auto* vertex_fields = cvf->data();
      const shape* shps = shapes_collection->data();
      dim3 block(kBlockX, kBlockY, 1);
      CountInteractionsPPKernel<kBlockX, kBlockY><<<n_pairs, block, 0, st>>>(
          cells, vertex_fields, shps, *rcut_inc, pairs.cell_i, pairs.cell_j, pairs.p_i, pairs.p_j,
          s.counts_.reserve(n_pairs), n_pairs);

      int* tc[InteractionTypeId::NTypesPP];
      int* tp[InteractionTypeId::NTypesPP];
      for (int t = 0; t < InteractionTypeId::NTypesPP; t++) {
        tc[t] = s.type_counts_[t].reserve(n_pairs);
        tp[t] = s.type_prefix_[t].reserve(n_pairs);
      }
      split_counts_kernel<<<nblocks(n_pairs), kThreads, 0, st>>>(n_pairs, s.counts_.data(), tc[0], tc[1], tc[2],
                                                                   tc[3]);
      ONIKA_CU_CHECK_ERRORS(cudaGetLastError());
      TypeBase base;
      InteractionParticleAccessor accessor;
      for (int t = 0; t < InteractionTypeId::NTypesPP; t++) {
        exclusive_scan_device(tc[t], tp[t], n_pairs, st);
        const size_t total = read_device(tp[t] + n_pairs - 1, st) + read_device(tc[t] + n_pairs - 1, st);
        const int typeID = InteractionTypeId::FirstIdParticle + t;
        base.v[t] = ic->extend_pp_tail(typeID, total);
        n_contacts += total;
      }
      for (int t = 0; t < InteractionTypeId::NTypesPP; t++) {
        accessor[InteractionTypeId::FirstIdParticle + t] =
            ic->get_data<InteractionType::ParticleParticle>(InteractionTypeId::FirstIdParticle + t).view();
      }
      pack_prefix_kernel<<<nblocks(n_pairs), kThreads, 0, st>>>(n_pairs, s.prefix_.reserve(n_pairs), tp[0], tp[1],
                                                                  tp[2], tp[3], base);
      BrokenPairHistory history = {pairs.hist, bonds, s.recv_.data()};
      FillInteractionsPPKernel<kBlockX, kBlockY><<<n_pairs, block, 0, st>>>(
          cells, vertex_fields, shps, *rcut_inc, pairs.cell_i, pairs.cell_j, pairs.p_i, pairs.p_j, pairs.ghost,
          s.prefix_.data(), accessor, n_pairs, history);
    }

    // 6. The friction now lives in the tail; drop the broken local interfaces.
    if (k > 0) {
      clear_friction_kernel<<<nblocks(k), kThreads, 0, st>>>(k, s.broken_.data(), itf, bonds.friction);
      const size_t n_kept = s.compactor_.run(n_interfaces, IsKept{interfaces.break_interface_.data()}, s.kept_, st);
      assert(n_kept == n_interfaces - k);
      if (n_kept > 0) {
        gather_interfaces_kernel<<<nblocks(n_kept), kThreads, 0, st>>>(n_kept, s.kept_.data(), itf,
                                                                        s.interfaces_.reserve(n_kept));
        ONIKA_CU_CHECK_ERRORS(cudaMemcpyAsync(interfaces.data_.data(), s.interfaces_.data(),
                                              n_kept * sizeof(Interface), cudaMemcpyDeviceToDevice, st));
      }
      ONIKA_CU_STREAM_SYNCHRONIZE(st);
      interfaces.data_.resize(n_kept);
      interfaces.break_interface_.resize(n_kept);
      if (n_kept > 0) {
        ONIKA_CU_CHECK_ERRORS(cudaMemsetAsync(interfaces.break_interface_.data(), 0, n_kept, st));
      }
    }
    ONIKA_CU_STREAM_SYNCHRONIZE(st);
    ONIKA_CU_CHECK_ERRORS(cudaGetLastError());

    ldbg << "update_broken_interfaces_gpu: " << k << " broken interfaces (" << m << " OwnerGhost, " << n_send
         << " bonds sent), " << n_recv << " bonds received, " << q << " PartnerGhost interfaces updated, "
         << n_contacts << " contacts created" << std::endl;
  }
};

// === register factories ===
ONIKA_AUTORUN_INIT(update_broken_interfaces_gpu) {
  OperatorNodeFactory::instance()->register_factory("update_broken_interfaces_gpu",
                                                    make_grid_variant_operator<UpdateBrokenInterfacesGPU>);
}
}  // namespace exaDEM
#endif
