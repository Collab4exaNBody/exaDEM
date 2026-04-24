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

#include <mpi.h>
#include <onika/scg/operator.h>
#include <onika/scg/operator_factory.h>
#include <onika/scg/operator_slot.h>

#include <exaDEM/drivers.hpp>

namespace exaDEM {
using namespace onika::scg;

struct force_to_accel {
  const double dt;
  const double mass;

  template <class T>
  inline void operator()(T& arg) {
    static_assert(get_type<T>() != DRIVER_TYPE::UNDEFINED);
    if constexpr (std::is_same_v<std::remove_cv_t<T>, Ball>) {
      arg.motion.weigth = mass;
      arg.f_ra(dt);
    }
    if constexpr (std::is_same_v<std::remove_cv_t<T>, Surface>) {
      arg.motion.weigth = mass;
    }
    arg.force_to_accel();
  }
};

struct gather_forces_moment {
  inline std::tuple<bool, exanb::Vec3d, exanb::Vec3d> operator()(Ball& arg) {
    if (arg.motion.is_compressive() || arg.motion.is_force_motion()) {
      return {true, arg.motion.forces, {0, 0, 0}};
    } else {
      return {false, {0, 0, 0}, {0, 0, 0}};
    }
  }

  inline std::tuple<bool, exanb::Vec3d, exanb::Vec3d> operator()(Surface& arg) {
    if (arg.motion.is_compressive() || arg.motion.is_force_motion()) {
      return {true, arg.motion.forces, {0, 0, 0}};
    } else {
      return {false, {0, 0, 0}, {0, 0, 0}};
    }
  }

  inline std::tuple<bool, exanb::Vec3d, exanb::Vec3d> operator()(Cylinder& arg) {
    return {false, {0, 0, 0}, {0, 0, 0}};
  }

  inline std::tuple<bool, exanb::Vec3d, exanb::Vec3d> operator()(RShapeDriver& arg) {
    if (arg.motion.need_forces() || arg.need_moment()) {
      return {true, arg.motion.forces, arg.fields.mom};
    }
    return {false, {0, 0, 0}, {0, 0, 0}};
  }
};

struct set_forces_moment {
  exanb::Vec3d forces;
  exanb::Vec3d moment;

  template <typename DriverT>
  inline void operator()(DriverT& arg) {
    arg.motion.forces = forces;
    if constexpr (std::is_same_v<std::decay_t<DriverT>, RShapeDriver>) {
      arg.fields.mom = moment;
    }
  }
};

class ForceToAccelDriverFunctor : public OperatorNode {
  ADD_SLOT(MPI_Comm, mpi, INPUT, MPI_COMM_WORLD, DocString{"MPI communicator for parallel processing."});
  ADD_SLOT(Drivers, drivers, INPUT_OUTPUT, REQUIRED, DocString{"List of Drivers"});
  ADD_SLOT(double, dt, INPUT, REQUIRED, DocString{"dt is the time increment of the timeloop"});
  ADD_SLOT(double, system_mass, INPUT, REQUIRED);

 public:
  inline std::string documentation() const final {
    return R"EOF(
          This operator updates driver centers using their velocities. Not that accelerations are not used.
        )EOF";
  }

  inline void execute() final {
    // we need to update forces if required
    std::vector<int> ids;      // id, forces
    std::vector<double> pack;  // id, forces
    gather_forces_moment reduce;
    for (size_t id = 0; id < drivers->get_size(); id++) {
      auto [update, forces, moment] = drivers->apply(id, reduce);
      if (update) {
        ids.push_back(id);
        pack.push_back(forces.x);
        pack.push_back(forces.y);
        pack.push_back(forces.z);
        pack.push_back(moment.x);
        pack.push_back(moment.y);
        pack.push_back(moment.z);
      }
    }

    if (pack.size() > 0) {
      std::vector<double> unpack(pack.size());
      MPI_Allreduce(pack.data(), unpack.data(), pack.size(), MPI_DOUBLE, MPI_SUM, *mpi);
      for (size_t i = 0; i < ids.size(); i++) {
        int id = ids[i];
        set_forces_moment func;
        func.forces = exanb::Vec3d{unpack[i * 6], unpack[i * 6 + 1], unpack[i * 6 + 2]};
        func.moment = exanb::Vec3d{unpack[i * 6 + 3], unpack[i * 6 + 4], unpack[i * 6 + 5]};
        drivers->apply(id, func);
      }
    }
    force_to_accel func = {*dt, *system_mass};
    for (size_t id = 0; id < drivers->get_size(); id++) {
      drivers->apply(id, func);
    }
  }
};

// === register factories ===
ONIKA_AUTORUN_INIT(force_to_accel_driver) {
  OperatorNodeFactory::instance()->register_factory("force_to_accel_driver",
                                                    make_simple_operator<ForceToAccelDriverFunctor>);
}
}  // namespace exaDEM
