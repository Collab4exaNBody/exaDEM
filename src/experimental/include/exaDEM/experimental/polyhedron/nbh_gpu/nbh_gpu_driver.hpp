#pragma once

#include <exaDEM/drivers.hpp>
#include <exaDEM/experimental/polyhedron/nbh_gpu/nbh_storage.hpp>
#include <exaDEM/polyhedron/vertices.hpp>

namespace exaDEM {
template <typename CellsT>
struct CountDriverInteractionsFunc {
  CellsT cells_;
  // WARNING (TEMPORARY): mutable workaround for onika::cuda::span's const
  mutable CellStorage::View cell_storage_accessor_;
  const size_t* const cell_ptr_;
  const double rcut_inc_;
  onika::cuda::span<shape> shps_;
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
        add_driver_interaction(driver, func, item, n_particles, rcut_inc_, t, id, vertex_cell, h, shps_.data());
      } else if (drv_type == DRIVER_TYPE::SURFACE) {
        item.pair_.type_ = InteractionTypeId::VertexSurface;
        Surface& driver = drvs_.get_typed_driver<Surface>(drvs_idx);
        add_driver_interaction(driver, func, item, n_particles, rcut_inc_, t, id, vertex_cell, h, shps_.data());
      } else if (drv_type == DRIVER_TYPE::BALL) {
        item.pair_.type_ = InteractionTypeId::VertexBall;
        Ball& driver = drvs_.get_typed_driver<Ball>(drvs_idx);
        add_driver_interaction(driver, func, item, n_particles, rcut_inc_, t, id, vertex_cell, h, shps_.data());
      } else if (drv_type == DRIVER_TYPE::RSHAPE) {
        RShapeDriver& driver = drvs_.get_typed_driver<RShapeDriver>(drvs_idx);
        add_driver_interaction(driver, cell_id, func, item, n_particles, rcut_inc_, t, id, rx, ry, rz, vertex_cell, h,
                               quat, shps_.data());
      }
    }
    auto& res = cell_storage_accessor_.size_[idx];
    for (int typeID = get_first_id<InteractionType::ParticleDriver>();
         typeID <= get_last_id<InteractionType::ParticleDriver>(); typeID++) {
      if (func.counter_[typeID] > 0) {
        ONIKA_CU_ATOMIC_ADD(res[typeID], func.counter_[typeID]);
      }
    }
  }
};

template <typename CellsT>
struct ClassifyDriverInteractionsFunc {
  CellsT cells_;
  CellStorage::View cell_storage_accessor_;
  const size_t* const cell_ptr_;
  const double rcut_inc_;
  onika::cuda::span<shape> shps_;
  VertexField* const vertex_fields_;
  DriversGPUAccessor drvs_;
  const ClassifierViewAccessor interaction_classifier_accessor_;

  static constexpr InteractionType IT = InteractionType::ParticleDriver;

  ONIKA_HOST_DEVICE_FUNC inline void operator()(long idx) const {
    struct AddInteractionFunc {
      const ClassifierViewAccessor& interaction_classifier_accessor_;
      InteractionTypePerCellCounter prefix_;
      ONIKA_HOST_DEVICE_FUNC inline void operator()(PlaceholderInteraction& item, int sub_i, int sub_j) {
        item.pair_.pi_.sub_ = sub_i;
        item.pair_.pj_.sub_ = sub_j;
        auto& container = interaction_classifier_accessor_.get_typed_accessor<IT>(item.type());
        container.set(prefix_[item.type()]++, item);
      }
    };
    AddInteractionFunc func = {interaction_classifier_accessor_, cell_storage_accessor_.offset_[idx]};

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
        add_driver_interaction(driver, func, item, n_particles, rcut_inc_, t, id, vertex_cell, h, shps_.data());
      } else if (drv_type == DRIVER_TYPE::SURFACE) {
        item.pair_.type_ = InteractionTypeId::VertexSurface;
        Surface& driver = drvs_.get_typed_driver<Surface>(drvs_idx);
        add_driver_interaction(driver, func, item, n_particles, rcut_inc_, t, id, vertex_cell, h, shps_.data());
      } else if (drv_type == DRIVER_TYPE::BALL) {
        item.pair_.type_ = InteractionTypeId::VertexBall;
        Ball& driver = drvs_.get_typed_driver<Ball>(drvs_idx);
        add_driver_interaction(driver, func, item, n_particles, rcut_inc_, t, id, vertex_cell, h, shps_.data());
      } else if (drv_type == DRIVER_TYPE::RSHAPE) {
        RShapeDriver& driver = drvs_.get_typed_driver<RShapeDriver>(drvs_idx);
        add_driver_interaction(driver, cell_id, func, item, n_particles, rcut_inc_, t, id, rx, ry, rz, vertex_cell, h,
                               quat, shps_.data());
      }
    }
  }
};
}  // namespace exaDEM

namespace onika {
namespace parallel {
template <typename CellsT>
struct ParallelForFunctorTraits<exaDEM::CountDriverInteractionsFunc<CellsT>> {
  static inline constexpr bool RequiresBlockSynchronousCall = false;
  static inline constexpr bool CudaCompatible = true;
};
template <typename CellsT>
struct ParallelForFunctorTraits<exaDEM::ClassifyDriverInteractionsFunc<CellsT>> {
  static inline constexpr bool RequiresBlockSynchronousCall = false;
  static inline constexpr bool CudaCompatible = true;
};
}  // namespace parallel
}  // namespace onika
