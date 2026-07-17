#pragma once

#include <exaDEM/drivers.hpp>
#include <exaDEM/experimental/polyhedron/nbh_gpu/nbh_storage.hpp>
#include <exaDEM/polyhedron/vertices.hpp>

namespace exaDEM {
struct CellDriverGPUAcessor {
  InteractionTypePerCellCounter* __restrict__ offset_;
  InteractionTypePerCellCounter* __restrict__ size_;
};

struct ResetCell {
  InteractionTypePerCellCounter* __restrict__ offset_;  ///< Pointer to array of offsets per interaction type
  InteractionTypePerCellCounter* __restrict__ size_;    ///< Pointer to array of sizes per interaction type

  /**
   * @brief Reset the data for a given cell index.
   * @param i Index of the cell to reset
   */
  ONIKA_HOST_DEVICE_FUNC
  inline void operator()(size_t i) const {
    for (size_t j = 0; j < InteractionTypeId::NTypes; j++) {
      size_[i][j] = 0;
      offset_[i][j] = 0;
    }
  }
};

struct CellDriverStorage {
  // Vector type alias using unified memory (GPU/CPU compatible)
  template <typename T>
  using VectorT = onika::memory::CudaMMVector<T>;
  VectorT<InteractionTypePerCellCounter> offset_;  ///< Offset for each inter
  VectorT<InteractionTypePerCellCounter> size_;    ///< Number of interaction

  // size should be the number of non empty cells
  template <typename ExecCtx>
  void resize(size_t newsize, ExecCtx& exec_ctx) {
    size_.resize(newsize);
    offset_.resize(newsize);

    // Reset members (skip flag and arrays)
    ResetCell reset_func = {offset_.data(), size_.data()};

    // Parallel execution over all cells
    onika::parallel::ParallelForOptions opts;
    opts.omp_scheduling = onika::parallel::OMP_SCHED_GUIDED;
    parallel_for(newsize, reset_func, exec_ctx(), opts);
  }
  CellDriverGPUAcessor accessor() { return {offset_.data(), size_.data()}; }
};

// CountNumberOfInteractionParticleDriverFunc = CountIPDFunc
template <typename TMPLC>
struct CountIPDFunc {
  TMPLC cells_;
  CellDriverGPUAcessor accessor_;
  const size_t* const cell_ptr_;
  const double rcut_inc_;
  const shape* const shps_;
  VertexField* const vertex_fields_;
  DriversGPUAccessor drvs_;

  ONIKA_HOST_DEVICE_FUNC inline void operator()(long idx) const {
    size_t cell_id = cell_ptr_[idx];
    auto& cell = cells_[cell_id];
    VertexField& vertex_cell = vertex_fields_[cell_id];

    struct NbhDriverCounter {
      InteractionTypePerCellCounter counter_{};
      ONIKA_HOST_DEVICE_FUNC inline void operator()(PlaceholderInteraction& item, int sub_i, int sub_j) {
        counter_[item.type()]++;
      }
    };

    size_t n_particles = cell.size();
    NbhDriverCounter func;
    PlaceholderInteraction item = {};
    item.pair_.swap_ = false;
    item.pair_.ghost_ = InteractionPair::NotGhost;
    auto& pi_c = item.i();
    auto& pd_c = item.driver();
    pi_c.cell_ = cell_id;
    pd_c.cell_ = 123456;  // Default value [debug]
    pd_c.p_ = 12345;      // Default value [debug]

    const auto* __restrict__ id = cell[field::id];
    const auto* __restrict__ h = cell[field::homothety];
    const auto* __restrict__ t = cell[field::type];
    const auto* __restrict__ rx = cell[field::rx];
    const auto* __restrict__ ry = cell[field::ry];
    const auto* __restrict__ rz = cell[field::rz];
    const auto* __restrict__ quat = cell[field::orient];
    for (size_t drvs_idx = 0; drvs_idx < drvs_.nb_drivers_; drvs_idx++) {
      DRIVER_TYPE drv_type = drvs_.type_index_[drvs_idx].type_;
      if (drv_type == DRIVER_TYPE::CYLINDER) {
        item.pair_.type_ = InteractionTypeId::VertexCylinder;
        Cylinder& driver = drvs_.get_typed_driver<Cylinder>(drvs_idx);
        add_driver_interaction(driver, func, item, n_particles, rcut_inc_, t, id, vertex_cell, h, shps_);
      } else if (drv_type == DRIVER_TYPE::SURFACE) {
        item.pair_.type_ = InteractionTypeId::VertexSurface;
        Surface& driver = drvs_.get_typed_driver<Surface>(drvs_idx);
        add_driver_interaction(driver, func, item, n_particles, rcut_inc_, t, id, vertex_cell, h, shps_);
      } else if (drv_type == DRIVER_TYPE::BALL) {
        item.pair_.type_ = InteractionTypeId::VertexBall;
        Ball& driver = drvs_.get_typed_driver<Ball>(drvs_idx);
        add_driver_interaction(driver, func, item, n_particles, rcut_inc_, t, id, vertex_cell, h, shps_);
      } else if (drv_type == DRIVER_TYPE::RSHAPE) {
        RShapeDriver& driver = drvs_.get_typed_driver<RShapeDriver>(drvs_idx);
        add_driver_interaction(driver, cell_id, func, item, n_particles, rcut_inc_, t, id, rx, ry, rz, vertex_cell, h,
                               quat, shps_);
      }
    }
    auto& res = accessor_.size_[idx];
    for (int typeID = get_first_id<InteractionType::ParticleDriver>();
         typeID <= get_last_id<InteractionType::ParticleDriver>(); typeID++) {
      if (func.counter_[typeID] > 0) {
        // accessor_.skip_driver_[idx] = false;
        ONIKA_CU_ATOMIC_ADD(res[typeID], func.counter_[typeID]);
      }
    }
  }
};

template <typename TMPLC>
struct ClassifyIPDFunc {
  TMPLC cells_;
  CellDriverGPUAcessor accessor_;
  const size_t* const cell_ptr_;
  const double rcut_inc_;
  const shape* const shps_;
  VertexField* const vertex_fields_;
  DriversGPUAccessor drvs_;
  const InteractionWrapperAccessor interactions_;

  static constexpr InteractionType IT = InteractionType::ParticleDriver;

  ONIKA_HOST_DEVICE_FUNC inline void operator()(long idx) const {
    struct AddInteractionFunc {
      const InteractionWrapperAccessor& data_;
      InteractionTypePerCellCounter prefix_;
      ONIKA_HOST_DEVICE_FUNC inline void operator()(PlaceholderInteraction& item, int sub_i, int sub_j) {
        item.pair_.pi_.sub_ = sub_i;
        item.pair_.pj_.sub_ = sub_j;
        auto& container = data_.get_typed_accessor<IT>(item.type());
        container.set(prefix_[item.type()]++, item);
      }
    };
    AddInteractionFunc func = {interactions_, accessor_.offset_[idx]};

    size_t cell_id = cell_ptr_[idx];
    auto& cell = cells_[cell_id];
    VertexField& vertex_cell = vertex_fields_[cell_id];
    size_t n_particles = cell.size();

    PlaceholderInteraction item = {};
    item.pair_.swap_ = false;
    item.pair_.ghost_ = InteractionPair::NotGhost;
    auto& pi = item.i();
    auto& pd = item.driver();
    pi.cell_ = cell_id;
    pd.cell_ = 123456;  // Default value [debug]
    pd.p_ = 12345;      // Default value [debug]

    // By default,  if the interaction is between a particle and a driver
    // Data about the particle j is set to -1
    // Except for id_j that contains the driver id
    const auto* __restrict__ id = cell[field::id];
    const auto* __restrict__ h = cell[field::homothety];
    const auto* __restrict__ t = cell[field::type];
    const auto* __restrict__ rx = cell[field::rx];
    const auto* __restrict__ ry = cell[field::ry];
    const auto* __restrict__ rz = cell[field::rz];
    const auto* __restrict__ quat = cell[field::orient];
    for (size_t drvs_idx = 0; drvs_idx < drvs_.nb_drivers_; drvs_idx++) {
      DRIVER_TYPE drv_type = drvs_.type_index_[drvs_idx].type_;
      pd.id_ = drvs_idx;  // we store the driver idx
      if (drv_type == DRIVER_TYPE::CYLINDER) {
        item.pair_.type_ = InteractionTypeId::VertexCylinder;
        Cylinder& driver = drvs_.get_typed_driver<Cylinder>(drvs_idx);
        add_driver_interaction(driver, func, item, n_particles, rcut_inc_, t, id, vertex_cell, h, shps_);
      } else if (drv_type == DRIVER_TYPE::SURFACE) {
        item.pair_.type_ = InteractionTypeId::VertexSurface;
        Surface& driver = drvs_.get_typed_driver<Surface>(drvs_idx);
        add_driver_interaction(driver, func, item, n_particles, rcut_inc_, t, id, vertex_cell, h, shps_);
      } else if (drv_type == DRIVER_TYPE::BALL) {
        item.pair_.type_ = InteractionTypeId::VertexBall;
        Ball& driver = drvs_.get_typed_driver<Ball>(drvs_idx);
        add_driver_interaction(driver, func, item, n_particles, rcut_inc_, t, id, vertex_cell, h, shps_);
      } else if (drv_type == DRIVER_TYPE::RSHAPE) {
        RShapeDriver& driver = drvs_.get_typed_driver<RShapeDriver>(drvs_idx);
        add_driver_interaction(driver, cell_id, func, item, n_particles, rcut_inc_, t, id, rx, ry, rz, vertex_cell, h,
                               quat, shps_);
      }
    }
  }
};
}  // namespace exaDEM

namespace onika {
namespace parallel {
template <typename TMPLC>
struct ParallelForFunctorTraits<exaDEM::CountIPDFunc<TMPLC>> {
  static inline constexpr bool RequiresBlockSynchronousCall = false;
  static inline constexpr bool CudaCompatible = true;
};
template <typename TMPLC>
struct ParallelForFunctorTraits<exaDEM::ClassifyIPDFunc<TMPLC>> {
  static inline constexpr bool RequiresBlockSynchronousCall = false;
  static inline constexpr bool CudaCompatible = true;
};
}  // namespace parallel
}  // namespace onika
