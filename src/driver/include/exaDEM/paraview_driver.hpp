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
#include <onika/math/basic_types.h>
#include <filesystem>
#include <fstream>

namespace exaDEM
{
  using namespace exanb;

  struct info_ball { int id; Vec3d center; double radius; Vec3d vel; };
  struct info_surface { int id; Vec3d normal; double offset; double vel; };

  std::tuple<bool, bool, Vec3d> intersect(Vec3d& n, double offset, Vec3d& corner1, Vec3d& corner2)
  {
    Vec3d c1 = corner1 - offset * n;
    Vec3d c2 = corner2 - offset * n;
    Vec3d dist = c2 - c1;
    double norm_proj_dist = exanb::dot(n, dist);
    if ( norm_proj_dist == 0) 
    {
      return {false, false, Vec3d{0,0,0}};
      //return {false, true, Vec3d{0,0,0}};
    }
    double t = -(exanb::dot(n, c1)) / norm_proj_dist;
    if (t < 0 || t > 1) 
    {
      return {false, false,Vec3d{0,0,0}};
    }
    Vec3d point = c1 + t * dist + offset * n;
    return {true, false, point};
  }

  std::vector<Vec3d> compute_intersections(Domain& domain, Vec3d& normal, double offset)
  {
    std::vector<Vec3d> points;
    auto [inf, sup] = domain.bounds();
    std::array<Vec3d, 8> corners = {
      Vec3d{inf.x, inf.y, inf.z}, // Corner 0
      Vec3d{sup.x, inf.y, inf.z}, // Corner 1
      Vec3d{sup.x, sup.y, inf.z}, // Corner 2
      Vec3d{inf.x, sup.y, inf.z}, // Corner 3
      Vec3d{inf.x, inf.y, sup.z}, // Corner 4
      Vec3d{sup.x, inf.y, sup.z}, // Corner 5
      Vec3d{sup.x, sup.y, sup.z},  // Corner 6
      Vec3d{inf.x, sup.y, sup.z} // Corner 7
    };

    std::vector<std::pair<int, int>> edges = {
        {0, 1}, {1, 2}, {2, 3}, {3, 0}, // Base inférieure
        {4, 5}, {5, 6}, {6, 7}, {7, 4}, // Base supérieure
        {0, 4}, {1, 5}, {2, 6}, {3, 7}  // Arêtes verticales
    };

    
		for(size_t idx = 0; idx < edges.size() ; idx++)
		{
			auto [i,j] = edges[idx];
			auto [one_point, two_points, point] = intersect(normal, offset, corners[i], corners[j]);
			if( one_point ) points.push_back(point);
			if( two_points) 
			{ 
				points.push_back(corners[i]); 
				points.push_back(corners[j]); 
			}
		}

		sort(points.begin(), points.end());
		auto it = unique(points.begin(), points.end());
		points.erase(it, points.end());
    // It's not my cup of tea, I give up. 

    int last_elem = points.size()-1;
    for(int i = 0; i < last_elem ; i++)
    {
      Vec3d& pi = points[i];
      int to_swap = -666;
      double vdist = 1e32;
      for( size_t j = i + 1; j < points.size() ; j++)
      {
        Vec3d& pj = points[j];
        Vec3d dist = pi - pj;
        int n_component_null = 0;
        if( dist.x == 0 ) n_component_null++;
        if( dist.y == 0 ) n_component_null++;
        if( dist.z == 0 ) n_component_null++;
        if(n_component_null >= 1)
        {
          if( to_swap != 666)
          {
            double new_vdist = exanb::norm(dist);
            if( new_vdist < vdist)
            {
              vdist = new_vdist;// exanb::norm(dist);
              to_swap = j;
            }
          }
          else
          {
            vdist = exanb::norm(dist);
            to_swap = j;
          }
        }
      }
      if ( to_swap == -666 ) 
      {
        // It should not append, but ... 
        lout << "Error when dumping surface drivers" << std::endl;
      }
      else
      {
        std::swap(points[i+1], points[to_swap]);
      }
    }
		return points;
	}

	void write_surfaces_paraview(Domain& domain, std::vector<info_surface>& surfaces, std::filesystem::path path, std::string filename)
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
		std::stringstream vertices;
		std::stringstream vels;
		std::stringstream polygons;

    file << std::setprecision(16);
    vertices.precision(8);
    vels.precision(16);

		ids << "SCALARS Driver_Index int 1" << std::endl;
		ids << "LOOKUP_TABLE Driver_Index" << std::endl;
		vels << "SCALARS Velocity float 1" << std::endl;
		vels << "LOOKUP_TABLE Velocity" << std::endl;

		int count = 0;
		for( auto [id, normal, offset, vel ] : surfaces )
		{
			std::vector<Vec3d> list_of_vertices = compute_intersections(domain, normal, offset);
			polygons << list_of_vertices.size() << " ";
			for(size_t i = 0 ; i < list_of_vertices.size() ; i++)
			{
				Vec3d& v = list_of_vertices[i];
				ids      << id << std::endl;
				vertices << v.x << " " << v.y << " " << v.z << std::endl;
				vels    << vel << std::endl;
				polygons << count << " ";
				count++;
			}
			polygons << std::endl;
		} 

		file << stream.rdbuf() << std::endl;
		file << "POINTS " << count << " float" << std::endl;
		file << vertices.rdbuf() << std::endl; 
		file << "POLYGONS " << surfaces.size() << " " << count + surfaces.size() << std::endl;
		file << polygons.rdbuf() << std::endl;
		file << "POINT_DATA " << count << std::endl;
		file << ids.rdbuf() << std::endl;
		file << vels.rdbuf() << std::endl;
	}

	void write_balls_paraview(std::vector<info_ball>& balls, std::filesystem::path path, std::string filename)
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
    
    file << std::setprecision(16);

		std::stringstream stream;

		stream << "# vtk DataFile Version 3.0" << std::endl;
		stream << "Spheres" << std::endl;
		stream << "ASCII" << std::endl;
		stream << "DATASET POLYDATA" << std::endl;

		std::stringstream ids;
		std::stringstream centers;
		std::stringstream radii;
		std::stringstream vels;

    centers.precision(16);
    radii.precision(16);
    vels.precision(16);

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
