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
#include <onika/math/quaternion.h>
#include <onika/math/quaternion_yaml.h>
#include <exaDEM/driver_base.h>
#include <exaDEM/shape.hpp>
#include <exaDEM/shape_reader.hpp>
//#include <exaDEM/interaction/interaction.hpp>
#include <filesystem>
#include <onika/physics/units.h>

namespace exaDEM
{
  using namespace exanb;

  template <typename T> using vector_t = onika::memory::CudaMMVector<T>;
  //template <typename T> using vector_t = std::vector<T>;

  /**
   * @brief Struct representing a list of elements( vertex, edge, or face).
   */
  struct list_of_elements
  {
    std::vector<int> vertices; /**< List of vertex indices. */
    std::vector<int> edges;    /**< List of edge indices. */
    std::vector<int> faces;    /**< List of face indices. */
    void clean() { vertices.clear(); edges.clear(); faces.clear(); }
  };

  struct Stl_params
  {
    exanb::Vec3d center = Vec3d{0,0,0}; /**< Center position of the STL mesh. */
    exanb::Vec3d vel = Vec3d{0,0,0};    /**< Velocity of the STL mesh. */
    exanb::Vec3d vrot = Vec3d{0,0,0};   /**< Angular velocity of the STL mesh. */
    exanb::Quaternion quat = {1,0,0,0}; /**< Quaternion of the STL mesh. */
    double surface = -1;                /**< Surface, used with linear_compression_motion. */
    double mass = std::numeric_limits<double>::max() / 4; /**< Mass of the ball */
  };
}

namespace YAML
{
  using exaDEM::Stl_params;
  using exaDEM::MotionType;
  using exanb::lerr;
  using onika::physics::Quantity;

  template <> struct convert<Stl_params>
  {
    static bool decode(const Node &node, Stl_params &v)
    {
      if (!node.IsMap())
      {
        return false;
      }
      if( check(node, "center") ) { v.center = node["center"].as<Vec3d>(); }
      if( check(node, "vel") )    { v.vel = node["vel"].as<Vec3d>(); }
      if( check(node, "vrot") )   { v.vrot = node["vrot"].as<Vec3d>(); }
      if( check(node, "mass") ) { v.mass = node["mass"].as<double>(); }
      if( check(node, "quat") ) { v.quat = node["quat"].as<exanb::Quaternion>(); }
      if( check(node, "surface") ) { v.surface = node["surface"].as<double>(); }
      return true;
    }
  };
}

namespace exaDEM
{
  const std::vector<MotionType> stl_valid_motion_types = {STATIONARY, LINEAR_MOTION, LINEAR_FORCE_MOTION, LINEAR_COMPRESSIVE_MOTION, TABULATED};

  using namespace exanb;
  /**
   * @brief Struct representing a STL mesh in the exaDEM simulation.
   */
  struct Stl_mesh : public Stl_params, Driver_params
  {
    shape shp;              /**< Shape of the STL mesh. */
    vector_t<Vec3d> vertices;      /**< Collection of vertices (computed from shp, quat and center). */
    std::vector<list_of_elements> grid_indexes; /**< Grid indices of the STL mesh. */
    std::vector<omp_lock_t> grid_mutexes; /**< Grid indices of the STL mesh. */
    /** We don't need to save these values */
    exanb::Vec3d acc = {0,0,0};       /**< Acceleration of the mesh */

    /**
     * @brief Get the type of the driver (in this case, STL_MESH).
     * @return The type of the driver.
     */
    constexpr DRIVER_TYPE get_type() { return DRIVER_TYPE::STL_MESH; }


    void set_shape(shape& s) {shp = s;}

   /**
     * @brief Print information about the STL mesh.
     */
    inline void print() const
    {
      lout << "Driver Type: MESH STL" << std::endl;
      lout << "Name               : " << shp.m_name << std::endl;
      lout << "Center             : " << center << std::endl;
      lout << "Velocity           : " << vel << std::endl;
      lout << "Angular Velocity   : " << vrot << std::endl;
      lout << "Orientation        : " << quat.w << " " << quat.x << " " << quat.y << " " << quat.z << std::endl;
      if(surface > 0.0) lout << "Surface            : " << surface << std::endl;
      lout << "Number of faces    : " << shp.get_number_of_faces() << std::endl;
      lout << "Number of edges    : " << shp.get_number_of_edges() << std::endl;
      lout << "Number of vertices : " << shp.get_number_of_vertices() << std::endl;
      Driver_params::print_driver_params();
    }

    /**
     * @brief Print information about the STL mesh.
     */
    inline void initialize()
    {
      // checks
      if( shp.get_number_of_faces() == 0 
          && shp.get_number_of_edges() == 0 
          && shp.get_number_of_vertices() == 0)
      {
        lout << "Your shape is not correctly defined, no vertex, no edge, and no face" << std::endl;
        std::abort();
      }

      // resize and initialize vertices
      vertices.resize(shp.get_number_of_vertices());  
#pragma omp parallel for schedule(static)
      for(int i = 0; i < shp.get_number_of_vertices() ; i++)
      {
        this->update_vertex(i);
      }

      // remove relative paths
      std::filesystem::path full_name = this->shp.m_name;
      this->shp.m_name = full_name.filename();
      // motion type
      if( !Driver_params::is_valid_motion_type(stl_valid_motion_types)) std::exit(EXIT_FAILURE);
      if( !Driver_params::check_motion_coherence()) std::exit(EXIT_FAILURE);
      if( mass <= 0.0 )
      {
        lout << "Please, define a positive mass." << std::endl;
        std::exit(EXIT_FAILURE);
      }

      if( is_compressive() )
      {
        double s = shp.compute_surface();
        if( surface <= 0 )
        {
          lout << "\033[31m[STL Error]: The surface value must be positive for LINEAR_COMPRESSIVE_FORCE. You need to specify surface: XX in the 'state' slot.\033[0m" << std::endl;
          lout << "\033[33m[STL Error]: The computed surface of all faces is: " << s << "\033[0m" << std::endl;
          std::exit(EXIT_FAILURE);
          lout << "\033[3221m[STL Warning]: The surface value must be positive for LINEAR_COMPRESSIVE_FORCE. You need to specify surface: XX in the 'state' slot.\033[0m" << std::endl;
        }
        if ( s - surface > 1e-6 )
        {
          lout << "\033[32m[STL Error]: The computed surface of all faces is: " << s << "\033[0m" << std::endl;
        }
      }
    }

    inline void force_to_accel()
    {
      if( is_compressive() )
      {
        constexpr double C = 0.5;
        if( mass != 0 )
        {
          const double s = surface;
          // compute acceleration
          Vec3d tmp = (forces - sigma * s - (damprate * vel) ) / (mass * C);
          // get acc into the motion vector axis
          acc =  exanb::dot(tmp, this->motion_vector) * this->motion_vector; 
        }
      }
      else if( is_force_motion() )
      {
        if( mass >= 1e100 ) lout << "Warning, the mass of the stl mesh is set to " << mass << std::endl;
        acc = Driver_params::sum_forces() / mass;
        //lout << "acceleration: " << acc << std::endl;
      }
      else
      {
        acc = {0,0,0};
      }
    }

		inline void push_f_v(const double dt)
		{

			if ( is_stationary() )
			{
				vel = Vec3d{0,0,0};
			}
			else
			{
				if( is_force_motion() )
				{
					vel += acc * dt;
				}

				if( motion_type == LINEAR_MOTION )
				{
					vel = motion_vector * const_vel; 
				}

				if( is_compressive() )
				{
					if( this->sigma != 0 ) vel += 0.5 * dt * acc;
				}
			}
		}

		inline void push_f_v_r(const double time, const double dt)
		{
      if( is_tabulated() ) 
      {
        center = tab_to_position(time);
        vel = tab_to_velocity(time);
      }
			else if( !is_stationary() )
			{
				if( motion_type == LINEAR_MOTION )
				{
					assert( vel = this->const_vel );
				}
				center += dt * vel + 0.5 * dt * dt * acc;
			}
		}


		ONIKA_HOST_DEVICE_FUNC 
			inline void update_vertex(int i)
			{
				vertices[i] = shp.get_vertex(i, this->center, this->quat);
			}

		/**
		 * @brief return driver velocity
		 */
		ONIKA_HOST_DEVICE_FUNC inline Vec3d &get_vel() { return vel; }

		/**
		 * @brief return driver velocity
		 */
		ONIKA_HOST_DEVICE_FUNC inline exanb::Quaternion &get_quat() { return quat; }

		inline bool stationary()
		{
			return is_stationary() && (vrot == Vec3d{0,0,0});
		}

		void dump_driver(int id, std::string path, std::stringstream &stream)
		{
			std::string filename = path + this->shp.m_name + ".shp";
			stream << "  - register_stl_mesh:" << std::endl;
			stream << "     id: " << id << std::endl;
			stream << "     filename: " << filename << std::endl;
			stream << "     minskowski: " << this->shp.m_radius << std::endl;
			stream << "     state: {";
			stream << "center: [" << this->center << "]";
			stream << ", vel: [" << this->vel << "]";
			stream << ", vrot: [" << this->vrot << "]";
			stream << ", surface: " << surface;
			stream << ", quat: [" << quat.w << "," << quat.x << "," << quat.y << "," << quat.z << "]";
			if ( is_force_motion() )
			{
				stream << ",mass: " << this->mass;
			}
			stream << "}" <<std::endl;
			Driver_params::dump_driver_params(stream);
			write_shp(this->shp, filename);
		}

		// angular velocity
		inline void push_av_to_quat(double dt)
		{
			using namespace exanb;
			// std::cout << dt << " " << vrot << std::endl;
			this->quat = this->quat + dot(this->quat, this->vrot) * dt;
			this->quat = normalize(this->quat);
			ldbg << "Quat[stl mesh]: " << this->quat.w << " " << this->quat.x << " " << this->quat.y << " " << this->quat.z << std::endl;
		}

		/**
		 * @brief Prints a summary of grid indices for the STL mesh.
		 * @details This function prints the number of elements in the grid indexes for vertices, edges, and faces.
		 */
		inline void grid_indexes_summary()
		{
			const size_t size = grid_indexes.size();
			size_t nb_fill_cells(0), nb_v(0), nb_e(0), nb_f(0), max_v(0), max_e(0), max_f(0);

#     pragma omp parallel for reduction(+: nb_fill_cells, nb_v, nb_e, nb_f) reduction(max: max_v, max_e, max_f)
			for (size_t i = 0; i < size; i++)
			{
				auto &list = grid_indexes[i];
				if (list.vertices.size() == 0 && list.edges.size() == 0 && list.faces.size())
					continue;
				nb_fill_cells++;
				nb_v += list.vertices.size();
				nb_e += list.edges.size();
				nb_f += list.faces.size();
				max_v = std::max(max_v, list.vertices.size());
				max_e = std::max(max_e, list.edges.size());
				max_f = std::max(max_f, list.faces.size());
			}

			lout << "========= STL Grid summary ======" << std::endl;
			lout << "Number of emplty cells = " << nb_fill_cells << " / " << size << std::endl;
			lout << "Vertices (Total/Max)   = " << nb_v << " / " << max_v << std::endl;
			lout << "Edges    (Total/Max)   = " << nb_e << " / " << max_e << std::endl;
			lout << "Faces    (Total/Max)   = " << nb_f << " / " << max_f << std::endl;
			lout << "=================================" << std::endl;
		}
	};
} // namespace exaDEM


namespace onika 
{ 
  namespace memory
  {
    template<> struct MemoryUsage< exaDEM::Stl_mesh >
    {
      static inline size_t memory_bytes(const exaDEM::Stl_mesh& obj)
      {
        const exaDEM::Stl_params * cparms = &obj;
        const exaDEM::Driver_params * dparms = &obj;
        return onika::memory::memory_bytes( *cparms , *dparms );
      }
    };
  } 
}


