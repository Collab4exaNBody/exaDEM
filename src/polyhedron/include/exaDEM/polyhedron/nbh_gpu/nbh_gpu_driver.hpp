#pragma once

#include <cub/block/block_scan.cuh>

namespace exaDEM {

template<typename TMPLC>
struct ApplyNbhDriverFunc {
  TMPLC cells;
  NbhCellAccessor accessor;
  size_t* cell_ptr;
  const double rcut_inc;
  const shape* const shps;
  VertexField* const vertex_fields;
  DriversGPUAccessor drvs;

  ONIKA_HOST_DEVICE_FUNC inline void operator()(long idx) const {
    size_t cell_id = cell_ptr[idx];
    auto& cell = cells[cell_id_a];
    VertexField& vertex_cell = vertex_fields[cell_id];

    struct NbhDriverCounter {
      InteractionTypePerCellCounter counter;
      ONIKA_HOST_DEVICE_FUNC inline 
          void operator()(PlaceholderInteraction& item, int sub_i, int sub_j) {
            counter[item.type]++;
          }
    };

    size_t n_particles = cell.size();
    NbhDriverCounter func;
    PlaceholderInteraction item;
    auto& pi = item.i();       // particle i (id, cell id, particle position, sub vertex)
    auto& pd = item.driver();  // particle driver (id, cell id, particle position, sub vertex)
    pi.cell = cell_idx;
    // By default, if the interaction is between a particle and a driver
    // Data about the particle j is set to -1
    // Except for id_j that contains the driver id
    pd.id = -1;
    pd.cell = -1;
    pd.p = -1;
    const auto* __restrict__ id = cell[field::id];
    const auto* __restrict__ h = cell[field::homothety];
    const auto* __restrict__ t = cell[field::type];
    const auto* __restrict__ rx = cell[field::rx];
    const auto* __restrict__ ry = cell[field::ry];
    const auto* __restrict__ rz = cell[field::rz];
    const auto* __restrict__ quat = cell[field::orient];
    for (size_t drvs_idx = 0; drvs_idx < drvs.get_size(); drvs_idx++) {
      pd.id = drvs_idx;  // we store the driver idx
      if (drvs.type(drvs_idx) == DRIVER_TYPE::CYLINDER) {
        item.pair.type = InteractionTypeId::VertexCylinder;
        Cylinder& driver = drvs.get_typed_driver<Cylinder>(drvs_idx);
        add_driver_interaction(driver, func, item, n_particles, rcut_inc, t, id, vertex_cell, h,
                               shps);
      } else if (drvs.type(drvs_idx) == DRIVER_TYPE::SURFACE) {
        item.pair.type = InteractionTypeId::VertexSurface;
        Surface& driver = drvs.get_typed_driver<Surface>(drvs_idx);
        add_driver_interaction(driver, func, item, n_particles, rcut_inc, t, id, vertex_cell, h,
                               shps);
      } else if (drvs.type(drvs_idx) == DRIVER_TYPE::BALL) {
        item.pair.type = InteractionTypeId::VertexBall;
        Ball& driver = drvs.get_typed_driver<Ball>(drvs_idx);
        add_driver_interaction(driver, func, item, n_particles, rcut_inc, t, id, vertex_cell, h,
                               shps);
      } else if (drvs.type(drvs_idx) == DRIVER_TYPE::RSHAPE) {
        RShapeDriver& driver = drvs.get_typed_driver<RShapeDriver>(drvs_idx);
        // driver.grid_indexes_summary();
        add_driver_interaction(driver, cell_id, func, item, n_particles, rcut_inc, t, id, rx, ry,
                               rz, vertex_cell, h, orient, shps);
      }
    }
    auto& res = accessor.size[idx];
    for (int typeID = get_first_id<InteractionType::ParticleDriver>() ;
         typeID <= get_last_id<InteractionType::ParticleDriver>() ; typeID++) {
      if (func.counter[typeID]>0) {
        //accessor.skip_driver[idx] = false;
        ONIKA_CU_ATOMIC_ADD(res[typeID], func.counter[typeID]);
      }
    }
  } 
};
}  // namespace exaDEM

template<typename TMPLC>
struct ApplyNbhClassifierDriverFunc {
  TMPLC cells;
  NbhCellAccessor accessor;
  size_t* cell_ptr;
  const double rcut_inc;
  const shape* const shps;
  VertexField* const vertex_fields;
  DriversGPUAccessor drvs;

  ONIKA_HOST_DEVICE_FUNC inline void operator()(long idx) const {
    struct AddInteractionFunc {
      const InteractionParticleAccessor& data;
      InteractionTypePerCellCounter& prefix;
      ONIKA_HOST_DEVICE_FUNC inline
          void operator()(PlaceholderInteraction& item, int sub_i, int sub_j) {
            item.pair.pi.sub = sub_i;
            item.pair.pj.sub = sub_j;
            data[item.type()].set(prefix[item.type()]++, item);
          }
    };
    AddInteractionFunc func = {interactions, accessor.offset[idx]};

    size_t cell_id = cell_ptr[idx];
    auto& cell = cells[cell_id_a];
    VertexField& vertex_cell = vertex_fields[cell_id];
    size_t n_particles = cell.size();
    PlaceholderInteraction item;
    auto& pi = item.i();       // particle i (id, cell id, particle position, sub vertex)
    auto& pd = item.driver();  // particle driver (id, cell id, particle position, sub vertex)
    pi.cell = cell_idx;
    // By default, if the interaction is between a particle and a driver
    // Data about the particle j is set to -1
    // Except for id_j that contains the driver id
    pd.id = -1;
    pd.cell = -1;
    pd.p = -1;
    const auto* __restrict__ id = cell[field::id];
    const auto* __restrict__ h = cell[field::homothety];
    const auto* __restrict__ t = cell[field::type];
    const auto* __restrict__ rx = cell[field::rx];
    const auto* __restrict__ ry = cell[field::ry];
    const auto* __restrict__ rz = cell[field::rz];
    const auto* __restrict__ quat = cell[field::orient];
    for (size_t drvs_idx = 0; drvs_idx < drvs.get_size(); drvs_idx++) {
      pd.id = drvs_idx;  // we store the driver idx
      if (drvs.type(drvs_idx) == DRIVER_TYPE::CYLINDER) {
        item.pair.type = InteractionTypeId::VertexCylinder;
        Cylinder& driver = drvs.get_typed_driver<Cylinder>(drvs_idx);
        add_driver_interaction(driver, func, item, n_particles, rcut_inc, t, id, vertex_cell, h,
                               shps);
      } else if (drvs.type(drvs_idx) == DRIVER_TYPE::SURFACE) {
        item.pair.type = InteractionTypeId::VertexSurface;
        Surface& driver = drvs.get_typed_driver<Surface>(drvs_idx);
        add_driver_interaction(driver, func, item, n_particles, rcut_inc, t, id, vertex_cell, h,
                               shps);
      } else if (drvs.type(drvs_idx) == DRIVER_TYPE::BALL) {
        item.pair.type = InteractionTypeId::VertexBall;
        Ball& driver = drvs.get_typed_driver<Ball>(drvs_idx);
        add_driver_interaction(driver, func, item, n_particles, rcut_inc, t, id, vertex_cell, h,
                               shps);
      } else if (drvs.type(drvs_idx) == DRIVER_TYPE::RSHAPE) {
        RShapeDriver& driver = drvs.get_typed_driver<RShapeDriver>(drvs_idx);
        // driver.grid_indexes_summary();
        add_driver_interaction(driver, cell_id, func, item, n_particles, rcut_inc, t, id, rx, ry,
                               rz, vertex_cell, h, orient, shps);
      }
    }

  }
};


namespace onika {
namespace parallel {
template<typename TMPLC>
struct ParallelForFunctorTraits<exaDEM::ApplyNbhDriverFunc<TMPLC>> {
  static inline constexpr bool RequiresBlockSynchronousCall = false;
  static inline constexpr bool CudaCompatible = true;
};
}  // namespace parallel
}  // namespace onika
