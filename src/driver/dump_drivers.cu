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
#include <onika/string_utils.h>
#include <exanb/core/parallel_grid_algorithm.h>

#include <exaDEM/drivers.hpp>
#include <memory>

namespace exaDEM {
using namespace exanb;

// Writes each driver's dump_data() (plain "drivers:" storage format, read back by read_drivers)
// rather than dump_op()'s register_*: operator-invocation format (see write_op_drivers).
struct DumpDriverDataFunc {
  int* const id_ptr_ = nullptr;
  const std::string directory_ = "";
  std::stringstream* const stream_ptr_ = nullptr;
  const Driver_params& motion_;

  inline void operator()(exaDEM::Surface& surface) const { surface.dump_data(motion_, (*id_ptr_)++, *stream_ptr_); }

  inline void operator()(exaDEM::Ball& ball) const { ball.dump_data(motion_, (*id_ptr_)++, *stream_ptr_); }

  inline void operator()(exaDEM::Cylinder& cylinder) const { cylinder.dump_data(motion_, (*id_ptr_)++, *stream_ptr_); }

  inline void operator()(exaDEM::RShapeDriver& rshape_param) const {
    rshape_param.dump_data(motion_, (*id_ptr_)++, directory_, *stream_ptr_);
  }
};

class DumpDrivers : public OperatorNode {
  ADD_SLOT(Drivers, drivers, INPUT_OUTPUT, REQUIRED, DocString{"List of Drivers"});
  ADD_SLOT(long, timestep, INPUT, REQUIRED, DocString{"Iteration number"});
  ADD_SLOT(std::string, dir_name, INPUT, REQUIRED, DocString{"Main output directory."});

 public:
  inline std::string documentation() const final {
    return R"EOF(
      Writes driver information to <dir_name>/CheckpointFiles/drivers_%010d.msp, in a plain
      "drivers:" storage format (one entry per driver: type, id, state, params, and for RSHAPE
      drivers a filename/minkowski) meant to be read back directly by read_drivers -- unlike
      write_op_drivers's setup_drivers:/register_*: operator-invocation format, this one is not
      meant to be pasted into a top-level includes: list.

      YAML example [no option]:

        - dump_drivers
     )EOF";
  }

  inline void execute() final {
    auto& drvs = *drivers;
    size_t n_drivers = drivers->get_size();

    if (n_drivers == 0) {
      return;
    }

    std::string path = *dir_name + "/CheckpointFiles/";
    std::stringstream data_stream;
    std::string filename = path + "drivers_%010d.msp";
    filename = onika::format_string(filename, *timestep);
    data_stream << "drivers:" << std::endl;
    data_stream << std::setprecision(16);

    int id_count = 0;
    for (size_t i = 0; i < n_drivers; i++) {
      DumpDriverDataFunc func = {&id_count, path, &data_stream, drvs.get_motion(i)};
      drvs.apply(i, func);
    }
    ldbg << id_count << " drivers have been dumped" << std::endl;

    std::ofstream file(filename.c_str());
    file << std::setprecision(16);
    file << data_stream.rdbuf();
  }
};

// === register factories ===
ONIKA_AUTORUN_INIT(dump_drivers) {
  OperatorNodeFactory::instance()->register_factory("dump_drivers", make_simple_operator<DumpDrivers>);
}
}  // namespace exaDEM
