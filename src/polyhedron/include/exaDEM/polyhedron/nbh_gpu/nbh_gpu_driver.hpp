#pragma once

#include <exaDEM/drivers.hpp>
#include <exaDEM/polyhedron/vertices.hpp>
#include <exaDEM/polyhedron/nbh_gpu/nbh_storage.hpp>

namespace exaDEM {
struct CellDriverGPUAcessor {
  InteractionTypePerCellCounter* __restrict__ offset;
  InteractionTypePerCellCounter* __restrict__ size;
};

struct ResetCell {
  InteractionTypePerCellCounter* __restrict__ _offset; ///< Pointer to array of offsets per interaction type
  InteractionTypePerCellCounter* __restrict__ _size;   ///< Pointer to array of sizes per interaction type

  /**
   * @brief Reset the data for a given cell index.
   * @param i Index of the cell to reset
   */
  ONIKA_HOST_DEVICE_FUNC
      inline void operator()(size_t i) const {
        for (size_t j = 0; j < InteractionTypeId::NTypes; j++) {
          _size[i][j] = 0;
          _offset[i][j] = 0;
        }
      }
};


struct CellDriverStorage {
  // Vector type alias using unified memory (GPU/CPU compatible)
  template<typename T> using VectorT = onika::memory::CudaMMVector<T>;
  VectorT<InteractionTypePerCellCounter> offset;    ///< Offset for each inter
  VectorT<InteractionTypePerCellCounter> size;      ///< Number of interaction

  // size should be the number of non empty cells
  template<typename ExecCtx>
  void resize(size_t newsize, ExecCtx& exec_ctx) {
    size.resize(newsize);
    offset.resize(newsize);

    // Reset members (skip flag and arrays)
    ResetCell reset_func = {offset.data(), size.data()};

    // Parallel execution over all cells
    onika::parallel::ParallelForOptions opts;
    opts.omp_scheduling = onika::parallel::OMP_SCHED_GUIDED;
    parallel_for(newsize, reset_func, exec_ctx(), opts);
  }
  CellDriverGPUAcessor accessor() {
    return {offset.data(), size.data()};
  }
};

// CountNumberOfInteractionParticleDriverFunc = CountIPDFunc
template<typename TMPLC>
struct CountIPDFunc {
  TMPLC cells;
  CellDriverGPUAcessor accessor;
  const size_t* const cell_ptr;
  const double rcut_inc;
  const shape* const shps;
  VertexField* const vertex_fields;
  DriversGPUAccessor drvs;

  ONIKA_HOST_DEVICE_FUNC inline void operator()(long idx) const {
    size_t cell_id = cell_ptr[idx];
    auto& cell = cells[cell_id];
    VertexField& vertex_cell = vertex_fields[cell_id];

    struct NbhDriverCounter {
      InteractionTypePerCellCounter counter{};
      ONIKA_HOST_DEVICE_FUNC inline 
          void operator()(PlaceholderInteraction& item, int sub_i, int sub_j) {
            counter[item.type()]++;
          }
    };

    size_t n_particles = cell.size();
    NbhDriverCounter func;
    PlaceholderInteraction item;  // not used
                                  // By default, if the interaction is between a particle and a driver
                                  // Data about the particle j is set to -1
                                  // Except for id_j that contains the driver id
    const auto* __restrict__ id = cell[field::id];
    const auto* __restrict__ h = cell[field::homothety];
    const auto* __restrict__ t = cell[field::type];
    const auto* __restrict__ rx = cell[field::rx];
    const auto* __restrict__ ry = cell[field::ry];
    const auto* __restrict__ rz = cell[field::rz];
    const auto* __restrict__ quat = cell[field::orient];
    for (size_t drvs_idx = 0; drvs_idx < drvs.m_nb_drivers; drvs_idx++) {
      DRIVER_TYPE drv_type = drvs.m_type_index[drvs_idx].m_type;
      if (drv_type == DRIVER_TYPE::CYLINDER) {
        item.pair.type = InteractionTypeId::VertexCylinder;
        Cylinder& driver = drvs.get_typed_driver<Cylinder>(drvs_idx);
        add_driver_interaction(driver, func, item, n_particles,
                               rcut_inc, t, id, vertex_cell, h, shps);
      } else if (drv_type == DRIVER_TYPE::SURFACE) {
        item.pair.type = InteractionTypeId::VertexSurface;
        Surface& driver = drvs.get_typed_driver<Surface>(drvs_idx);
        add_driver_interaction(driver, func, item, n_particles, rcut_inc, t, id, vertex_cell, h,
                               shps);
      } else if (drv_type == DRIVER_TYPE::BALL) {
        item.pair.type = InteractionTypeId::VertexBall;
        Ball& driver = drvs.get_typed_driver<Ball>(drvs_idx);
        add_driver_interaction(driver, func, item, n_particles, rcut_inc, t, id, vertex_cell, h,
                               shps);
      } else if (drv_type == DRIVER_TYPE::RSHAPE) {
        RShapeDriver& driver = drvs.get_typed_driver<RShapeDriver>(drvs_idx);
        add_driver_interaction(driver, cell_id, func, item, n_particles, rcut_inc, t, id, rx, ry,
                               rz, vertex_cell, h, quat, shps);
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

template<typename TMPLC>
struct ClassifyIPDFunc {
  TMPLC cells;
  CellDriverGPUAcessor accessor;
  const size_t* const cell_ptr;
  const double rcut_inc;
  const shape* const shps;
  VertexField* const vertex_fields;
  DriversGPUAccessor drvs;
  const InteractionWrapperAccessor interactions;

  static constexpr InteractionType IT = InteractionType::ParticleDriver;

  ONIKA_HOST_DEVICE_FUNC inline void operator()(long idx) const {
    struct AddInteractionFunc {
      const InteractionWrapperAccessor& data;
      InteractionTypePerCellCounter prefix;
      ONIKA_HOST_DEVICE_FUNC inline
          void operator()(PlaceholderInteraction& item, int sub_i, int sub_j) {
            item.pair.pi.sub = sub_i;
            item.pair.pj.sub = sub_j;
            auto& container = data.get_typed_accessor<IT>(item.type());
            container.set(prefix[item.type()]++, item);
          }
    };
    AddInteractionFunc func = { interactions, accessor.offset[idx]};

    size_t cell_id = cell_ptr[idx];
    auto& cell = cells[cell_id];
    VertexField& vertex_cell = vertex_fields[cell_id];
    size_t n_particles = cell.size();
    PlaceholderInteraction item;
    auto& pi = item.i();       // particle i (id, cell id, particle position, sub vertex)
    auto& pd = item.driver();  // particle driver (id, cell id, particle position, sub vertex)
    pi.cell = cell_id;
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
    for (size_t drvs_idx = 0; drvs_idx < drvs.m_nb_drivers; drvs_idx++) {
      DRIVER_TYPE drv_type = drvs.m_type_index[drvs_idx].m_type;
      pd.id = drvs_idx;  // we store the driver idx
      if (drv_type == DRIVER_TYPE::CYLINDER) {
        item.pair.type = InteractionTypeId::VertexCylinder;
        Cylinder& driver = drvs.get_typed_driver<Cylinder>(drvs_idx);
        add_driver_interaction(driver, func, item, n_particles, rcut_inc, t, id, vertex_cell, h,
                               shps);
      } else if (drv_type == DRIVER_TYPE::SURFACE) {
        item.pair.type = InteractionTypeId::VertexSurface;
        Surface& driver = drvs.get_typed_driver<Surface>(drvs_idx);
        add_driver_interaction(driver, func, item, n_particles, rcut_inc, t, id, vertex_cell, h,
                               shps);
      } else if (drv_type == DRIVER_TYPE::BALL) {
        item.pair.type = InteractionTypeId::VertexBall;
        Ball& driver = drvs.get_typed_driver<Ball>(drvs_idx);
        add_driver_interaction(driver, func, item, n_particles, rcut_inc, t, id, vertex_cell, h,
                               shps);
      } else if (drv_type == DRIVER_TYPE::RSHAPE) {
        RShapeDriver& driver = drvs.get_typed_driver<RShapeDriver>(drvs_idx);
        add_driver_interaction(driver, cell_id, func, item, n_particles, rcut_inc, t, id, rx, ry,
                               rz, vertex_cell, h, quat, shps);
      }
    }
  }
};
}  // namespace exaDEM


namespace onika {
namespace parallel {
template<typename TMPLC>
struct ParallelForFunctorTraits<exaDEM::CountIPDFunc<TMPLC>> {
  static inline constexpr bool RequiresBlockSynchronousCall = false;
  static inline constexpr bool CudaCompatible = true;
};
template<typename TMPLC>
struct ParallelForFunctorTraits<exaDEM::ClassifyIPDFunc<TMPLC>> {
  static inline constexpr bool RequiresBlockSynchronousCall = false;
  static inline constexpr bool CudaCompatible = true;
};
}  // namespace parallel
}  // namespace onika
