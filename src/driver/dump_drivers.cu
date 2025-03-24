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

#include <onika/scg/operator.h>
#include <onika/scg/operator_slot.h>
#include <onika/scg/operator_factory.h>
#include <exanb/core/parallel_grid_algorithm.h>
#include <mpi.h>
#include <memory>
#include <exaDEM/stl_mesh.h>
#include <exaDEM/drivers.h>
#include <onika/string_utils.h>

namespace exaDEM
{

  struct DumpDriverFunc
  {
    int * const id_ptr = nullptr;
    const std::string directory = "";
    std::stringstream * const stream_ptr = nullptr;

    inline void operator()(exaDEM::Surface &surface) const { surface.dump_driver( (*id_ptr) ++ , *stream_ptr ); }

    inline void operator()(exaDEM::Ball &ball) const { ball.dump_driver( (*id_ptr) ++ , *stream_ptr ); }

    inline void operator()(exaDEM::Cylinder &cylinder) const { cylinder.dump_driver( (*id_ptr) ++ , *stream_ptr ); }

    inline void operator()(exaDEM::Stl_mesh &stl_param) const { stl_param.dump_driver( (*id_ptr) ++ , directory, *stream_ptr ); }
  };

  using namespace exanb;
  class DumpDriver : public OperatorNode
  {
    static constexpr Vec3d null = {0.0, 0.0, 0.0};

    ADD_SLOT(Drivers, drivers, INPUT_OUTPUT, REQUIRED, DocString{"List of Drivers"});
    ADD_SLOT(long, timestep, INPUT, DocString{"Iteration number"});
    ADD_SLOT(std::string, dir_name, INPUT, REQUIRED, DocString{"Main output directory."});

  public:
    inline std::string documentation() const override final { return R"EOF( This operator outputs driver information. )EOF"; }

    inline void execute() override final
    {
      auto &drvs = *drivers;
      size_t n_drivers = drivers->get_size();

      if (n_drivers == 0)
        return;

      std::string path = *dir_name + "/CheckpointFiles/";
      std::stringstream data_stream;
      std::string filename = path + "driver_%010d.msp";
      filename = onika::format_string(filename, *timestep);
      data_stream << "setup_drivers:" << std::endl;
      data_stream << std::setprecision(16);

      int id_count = 0;
      DumpDriverFunc func = {&id_count, path, &data_stream};
      for (size_t i = 0; i < n_drivers; i++)
      {
        drvs.apply( i , func );
      }
      ldbg << id_count << " drivers have been dumped" << std::endl;
      
      std::ofstream file(filename.c_str());
      file << std::setprecision(16);
      file << data_stream.rdbuf();
    }
  };

  // === register factories ===
  ONIKA_AUTORUN_INIT(dump_drivers) { OperatorNodeFactory::instance()->register_factory("dump_driver", make_simple_operator<DumpDriver>); }
} // namespace exaDEM
