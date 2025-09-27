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
  class ReadShapeFileOperator : public OperatorNode
  {
    ADD_SLOT(std::string, filename, INPUT, REQUIRED, DocString{"Input filename"});
    ADD_SLOT(shapes, shapes_collection, INPUT_OUTPUT, DocString{"Collection of shapes"});
    ADD_SLOT(ParticleTypeMap, particle_type_map, INPUT_OUTPUT );
    ADD_SLOT(std::vector<double>, scale_factor, INPUT, OPTIONAL, DocString{"This option 'scale_factor' the input shapes. OBB, volume, vertices, and intertia are recomputed. Note that a vector of double should be provided. Example: scale_factor: [1.2,1,5.2]"});
    ADD_SLOT(std::vector<std::string>, rename, INPUT, OPTIONAL, DocString{"This option renames the input shapes. Note that a vector of string should be provided. Example: rename: [Shape1, Shape2, Shape3]"});
    ADD_SLOT(bool, rescale_minskowski, INPUT, true, DocString{"This option disable the rescaling of the minskowski radius."});
    ADD_SLOT(bool, verbosity, INPUT, true );

  public:
    inline std::string documentation() const override final
    {
      return R"EOF( 
        This operator initialize shapes data structure from a shape input file.

        YAML example:

					- read_shape_file:
						 filename: shapes.shp

					- read_shape_file:
						 filename: shapes.shp
						 rename: [PolyR, Octahedron]

					- read_shape_file:
						 filename: shapes.shp
						 rename:       [ PolyRSize2, OctahedronSize2]
						 scale_facton: [        2.0,             2.0]
    	    			)EOF";
    }

    inline void execute() override final
    {
      auto& ptm = *particle_type_map;
      lout << "Read file= " << *filename << std::endl;
      const bool BigShape = false; // do not remove it
      std::vector<shape> list_of_shapes = exaDEM::read_shps(*filename, BigShape);
      if(rename.has_value())
      {
        std::vector<std::string> names = *rename;
        if(list_of_shapes.size() != names.size()) 
        {
          color_log::error("read_shape_file", "The vector size 'rename' should have " + std::to_string(list_of_shapes.size()) + " + elements and not " + std::to_string(names.size()) + "elements"); 
        }

        for(size_t sid = 0 ; sid < list_of_shapes.size() ; sid++)
        {
          list_of_shapes[sid].m_name = names[sid];
        }
      }

      if(scale_factor.has_value())
      {
        if( !(*rescale_minskowski) ) 
        { 
          color_log::warning("read_shape_file", "You are disabling the minskowski radius rescaling, note that the volume and other properties could be wrong.");
        }

        std::vector<double> scales = *scale_factor;

        if(list_of_shapes.size() != scales.size())
        {
          color_log::error("read_shape_file", "The vector size 'scale_factor' should have " + std::to_string(list_of_shapes.size()) + " + elements and not " + std::to_string(scales.size()) + "elements"); 
        }

        const bool rescale_minskowki_radius = *rescale_minskowski;
        for(size_t sid = 0 ; sid < list_of_shapes.size() ; sid++)
        {
          list_of_shapes[sid].rescale(scales[sid], rescale_minskowki_radius);
        }
      }

      exaDEM::register_shapes(ptm, *shapes_collection, list_of_shapes);

      if( *verbosity )
      {
        for(const auto& [ name, type ] : ptm)
        {
          lout << "Shape[" << type <<"] is " << name << std::endl;
        }
      }
    };
  };

  // === register factories ===
  ONIKA_AUTORUN_INIT(read_shape_file) { OperatorNodeFactory::instance()->register_factory("read_shape_file", make_simple_operator<ReadShapeFileOperator>); }
} // namespace exaDEM
