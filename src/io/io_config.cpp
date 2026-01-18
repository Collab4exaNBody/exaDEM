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

namespace exaDEM {
using namespace exanb;

class IOConfigNode : public OperatorNode {
  // save file
  ADD_SLOT(std::string, dir_name, INPUT_OUTPUT, "ExaDEMOutputDir", DocString{"Main output directory."});
  ADD_SLOT(std::string, log_name, INPUT_OUTPUT, "log.txt", DocString{"Write an Output file containing log lines."});
  ADD_SLOT(std::string, avg_stress_tensor_name, INPUT_OUTPUT, "AvgStressTensor.txt",
           DocString{"Write an Output file containing stress tensors."});
  ADD_SLOT(std::string, interaction_basename, INPUT_OUTPUT, "InteractionOutputDir-",
           DocString{"Write an Output file containing interactions."});

 public:
  inline std::string documentation() const final {
    return R"EOF(
      This operator defines the tree structure of output files.

      YAML example:

        - config_io:
           dir_name: "YourExaDEMOutputDirName"
           log_name: "ExaDEMLogs.txt"
      )EOF";
  }

  inline bool is_sink() const final {
    return true;
  }

  inline void execute() final {
    std::string dirName = *dir_name;
    std::string logName = dirName + "/" + (*log_name);
    std::string avgStressTensorName = dirName + "/" + (*avg_stress_tensor_name);
    std::string interactionBasename = dirName + "/ExaDEMAnalyses/" + (*interaction_basename);
    lout << std::endl;
    lout << "==================== IO Directory Configuration =================" << std::endl;
    lout << "Directory Name:             " << dirName << std::endl;
    lout << "Log Filename:               " << logName << std::endl;
    lout << "Avg Stress Tensor Filename: " << avgStressTensorName << std::endl;
    lout << "Interaction Basename Dir:   " << interactionBasename << std::endl;
    lout << "Paraview Files Directory:   " << dirName + "/ParaviewOutputs/" << std::endl;
    lout << "Checkpoint Files Directory: " << dirName + "/CheckpointFiles/" << std::endl;
    lout << "=================================================================" << std::endl;
  }
};

// === register factories ===
ONIKA_AUTORUN_INIT(io_config) {
  OperatorNodeFactory::instance()->register_factory("io_config", make_simple_operator<IOConfigNode>);
}

}  // namespace exaDEM
