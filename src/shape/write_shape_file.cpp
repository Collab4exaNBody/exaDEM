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
#include <exanb/core/operator.h>
#include <exanb/core/operator_slot.h>
#include <exanb/core/operator_factory.h>
#include <exanb/core/basic_types.h>
#include <exanb/core/basic_types_operators.h>
#include <exanb/core/basic_types_stream.h>
#include <onika/memory/allocator.h> // for ONIKA_ASSUME_ALIGNED macro
#include <vector>
#include <iomanip>
#include <mpi.h>
#include <filesystem>
#include <exaDEM/shapes.hpp>
#include <exaDEM/shape_reader.hpp>

namespace exaDEM
{
  using namespace exanb;
  class WriteShapeFileOperator : public OperatorNode
  {
    ADD_SLOT(MPI_Comm, mpi, INPUT, MPI_COMM_WORLD);
    ADD_SLOT(std::string, filename, INPUT, "RestartShapeFile.shp", DocString{"Input filename"});
    ADD_SLOT(shapes, shapes_collection, INPUT, DocString{"Collection of shapes"});
    ADD_SLOT(std::string, dir_name, INPUT, REQUIRED, DocString{"Main output directory."});

  public:
    inline std::string documentation() const override final { return R"EOF( This operator writes shapes data structure into a "shp" file. )EOF"; }

    inline void execute() override final
    {
      // get shapes
      auto &shps = *shapes_collection;
      // this operator does not writes data file if there is not any shape.
      size_t size = shps.get_size();
      if (size == 0)
        return;
      int rank;
      MPI_Comm_rank(*mpi, &rank);
      // same data for all mpi processes
      if (rank == 0)
      {
        // define paths
        std::stringstream stream;
        std::string dir = *dir_name + "/CheckpointFiles/";
        std::string filepath = dir + *filename;
        lout << "Write shapes into: " << filepath << std::endl;
        // creating directory if it does not already exist
        const std::filesystem::path fspath(dir);
        std::filesystem::create_directories(fspath);
        // open output file
        std::ofstream outFile(filepath);
        if (!outFile)
        {
          std::cerr << "Error: impossible to create the output file: " << filepath << std::endl;
          return;
        }
        // fill stream with shape data
        for (size_t i = 0; i < size; i++)
        {
          const shape *shp = shps[i];
          exaDEM::write_shp(*shp, stream);
        }
        // fill output file
        outFile << stream.rdbuf();
      }
    };
  };

  // === register factories ===
  CONSTRUCTOR_FUNCTION { OperatorNodeFactory::instance()->register_factory("write_shape_file", make_simple_operator<WriteShapeFileOperator>); }
} // namespace exaDEM
