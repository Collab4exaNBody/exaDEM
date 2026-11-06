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
#include <filesystem>
#include <fstream>

namespace exaDEM {
using namespace onika::scg;

// Above this number of drivers, the per-driver table no longer fits readably on screen,
// so the detailed listing is written to a file instead.
static constexpr size_t PRINT_DRIVERS_TABLE_MAX = 10;

class PrintDrivers : public OperatorNode {
  ADD_SLOT(Drivers, drivers, INPUT, REQUIRED, DocString{"List of Drivers"});
  ADD_SLOT(std::string, dir_name, INPUT, "ExaDEMOutputDir", DocString{"Main output directory."});
  ADD_SLOT(MPI_Comm, mpi, INPUT, MPI_COMM_WORLD);

 public:
  inline std::string documentation() const final {
    return R"EOF(
        This operator prints drivers. If there are more than 10 drivers, the detailed
        listing is written to <dir_name>/drivers.txt instead of being printed as a table.
        )EOF";
  }

  inline void execute() final {
    using exanb::lout;
    auto& drvs = *drivers;
    lout << std::endl;
    lout << "==================== Driver Configuraions =======================" << std::endl;
    lout << "===== Summary" << std::endl;
    drvs.stats_drivers();
    lout << "===== List Of Drivers" << std::endl;
    if (drvs.get_size() > PRINT_DRIVERS_TABLE_MAX) {
      int rank = 0;
      MPI_Comm_rank(*mpi, &rank);
      if (rank == 0) {
        namespace fs = std::filesystem;
        fs::create_directories(*dir_name);
        std::string full_path = *dir_name + "/drivers.txt";
        std::ofstream file(full_path);
        drvs.print_drivers_detailed(file);
        lout << "More than " << PRINT_DRIVERS_TABLE_MAX << " drivers: full driver list written to " << full_path
             << std::endl;
      }
    } else {
      drvs.print_drivers_table();
    }
    lout << "=================================================================" << std::endl;
  }
};

// === register factories ===
ONIKA_AUTORUN_INIT(print_drivers) {
  OperatorNodeFactory::instance()->register_factory("print_drivers", make_simple_operator<PrintDrivers>);
}
}  // namespace exaDEM
