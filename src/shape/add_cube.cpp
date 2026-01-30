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
#include <exanb/core/particle_type_id.h>
#include <exaDEM/shapes.hpp>
#include <exaDEM/shape/basic_shape/cube.hpp>

namespace exaDEM {
class AddShapeCubeOperator : public OperatorNode {
  ADD_SLOT(shapes, shapes_collection, INPUT_OUTPUT, DocString{"Collection of shapes"});
  ADD_SLOT(ParticleTypeMap, particle_type_map, INPUT_OUTPUT);
  ADD_SLOT(double, length, INPUT, 1.0, DocString{"Define cube length. "});
  ADD_SLOT(std::string, name, INPUT, "cube", DocString{"Set Shape name."});
  ADD_SLOT(double, minskowski, INPUT, 0.01, DocString{"Set Minskowski value."});
  ADD_SLOT(bool, verbosity, INPUT, true);

 public:
  inline std::string documentation() const override final {
    return R"EOF( 
        This operator initialize shapes data structure from a shape input file.

        YAML example:

					- add_cube:
             name: MyCube
             minskowski: 0.0001
             length: 0.25
    )EOF";
  }

  inline void execute() override final {
    auto& ptm = *particle_type_map;
    lout << "Add Shape: " << *name << " length: " << *length << " minskowski: " << *minskowski << std::endl;
    shape shp = basic_shape::create_cube(*name, *length, *minskowski);
    register_shape(ptm, *shapes_collection, shp);
  };
};

// === register factories ===
ONIKA_AUTORUN_INIT(add_cube_file) {
  OperatorNodeFactory::instance()->register_factory("add_cube", make_simple_operator<AddShapeCubeOperator>);
}
}  // namespace exaDEM
