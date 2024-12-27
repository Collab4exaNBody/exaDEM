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
#pragma once
#include <exanb/core/basic_types.h>
#include <filesystem>
#include <fstream>

namespace exaDEM
{
  using namespace exanb;

  struct info_ball { int id; Vec3d center; double radius; Vec3d vel; };

  void write_balls_paraview(std::vector<info_ball> balls, std::filesystem::path path, std::string filename)
  {
    std::string full_path = path.string() + "/" + filename;
    if( !std::filesystem::exists(full_path))
		{
			std::filesystem::create_directories(path);
		}  
		int rank;
		MPI_Comm_rank(MPI_COMM_WORLD, &rank);
		if(rank != 0) return;

    std::ofstream file;
		file.open(full_path);
		if (!file)
		{
			std::cerr << "Error: impossible to create the output file: " << full_path << std::endl;
			return;
		}
    std::stringstream stream;

    stream << "# vtk DataFile Version 3.0" << std::endl;
    stream << "Spheres" << std::endl;
    stream << "ASCII" << std::endl;
    stream << "DATASET POLYDATA" << std::endl;

    std::stringstream ids;
    std::stringstream centers;
    std::stringstream radii;
    std::stringstream vels;

    centers << "POINTS " << balls.size() << " float" << std::endl;
    ids << "SCALARS Driver_Index int 1" << std::endl;
    ids << "LOOKUP_TABLE Driver_Index" << std::endl;
    radii << "SCALARS Radius float 1" << std::endl;
    radii << "LOOKUP_TABLE Radius" << std::endl;

    vels << "VECTORS dataName float" << std::endl;
    
    for( auto [id, center, rad, vel] : balls )
    {
      ids     << id << std::endl;
      centers << center.x << " " << center.y << " " << center.z << std::endl;
      radii   << rad << std::endl;
      vels    << vel.x << " " << vel.y << " " << vel.z << std::endl;
    }

    file << stream.rdbuf() << std::endl;
    file << centers.rdbuf() << std::endl;
    file << "POINT_DATA " << balls.size() << std::endl;
    file << radii.rdbuf() << std::endl;
    file << ids.rdbuf() << std::endl;
    file << vels.rdbuf() << std::endl;

	}
}
