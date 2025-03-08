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
#include <onika/scg/operator_factory.h>
#include <onika/scg/operator_slot.h>
#include <onika/log.h>
#include <onika/string_utils.h>

#include <memory>

namespace exanb
{

  class TimeStepParaviewFileNameOperator : public OperatorNode
  {
    ADD_SLOT(long, timestep, INPUT, REQUIRED);
    ADD_SLOT(std::string, format, INPUT, REQUIRED);
    ADD_SLOT(std::string, dir_name, INPUT, REQUIRED, DocString{"Output directory name."});
    ADD_SLOT(std::string, filename, OUTPUT);

  public:
    inline void execute() override final
    {
      std::string paraview_filename = (*dir_name) + "/" + (*format);
      *filename = onika::format_string(paraview_filename, *timestep);
    }

    inline void yaml_initialize(const YAML::Node &node) override final
    {
      YAML::Node tmp;
      if (node.IsScalar())
      {
        tmp["format"] = node;
      }
      else
      {
        tmp = node;
      }
      this->OperatorNode::yaml_initialize(tmp);
    }
  };

  // === register factories ===
  ONIKA_AUTORUN_INIT(paraview_file_name) { OperatorNodeFactory::instance()->register_factory("timestep_paraview_file", make_compatible_operator<TimeStepParaviewFileNameOperator>); }

} // namespace exanb
