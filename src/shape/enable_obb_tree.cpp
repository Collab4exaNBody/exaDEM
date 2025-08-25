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
#include <vector>
#include <iomanip>
#include <onika/scg/operator.h>
#include <onika/scg/operator_slot.h>
#include <onika/scg/operator_factory.h>
#include <onika/math/basic_types.h>
#include <onika/math/basic_types_operators.h>
#include <onika/math/basic_types_stream.h>
#include <onika/memory/allocator.h> // for ONIKA_ASSUME_ALIGNED macro
#include <exanb/core/particle_type_id.h>
#include <exaDEM/shapes.hpp>
#include <exaDEM/shape_reader.hpp>

namespace exaDEM
{
  using namespace exanb;
  class EnableOBBTree : public OperatorNode
  {
    ADD_SLOT(shapes, shapes_collection, INPUT_OUTPUT, REQUIRED, DocString{"Collection of shapes"});
    ADD_SLOT(bool, verbosity, INPUT, false, PRIVATE );
    ADD_SLOT(bool, enable_obb_tree, OUTPUT);

  public:
    inline std::string documentation() const override final
    {
      return R"EOF( 
        This operator initializes for each shape its OBBtree.

        YAML example:

          - enable_obb_tree:
             verbosity: false
    	)EOF";
    }

    inline void execute() override final
    {
      auto& shapes = *shapes_collection;
      shapes.enable_obb_tree();
      for(size_t sid = 0; sid < shapes.size() ; sid++)
      {
        shape* shp = shapes[sid];
        shp->buildOBBtree();
        if(*verbosity)
        {
          lout << "Build OBBTree for the shape " << shp->m_name << "." << std::endl;
        }
      }
      *enable_obb_tree = true;
    };
  };

  // === register factories ===
  ONIKA_AUTORUN_INIT(enable_obb_tree) 
  { 
    OperatorNodeFactory::instance()->register_factory("enable_obb_tree", make_simple_operator<EnableOBBTree>); 
  }
} // namespace exaDEM
