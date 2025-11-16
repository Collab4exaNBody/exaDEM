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

#include <fstream>
#include <vector>
#include <cassert>
#include <cmath>
#include <onika/log.h>
#include <exaDEM/color_log.hpp>
#include <exaDEM/basic_types.hpp>
#include <onika/cuda/cuda.h>
#include <onika/memory/allocator.h>
#include <onika/cuda/stl_adaptors.h>
#include <onika/math/basic_types.h>

//#include <exaDEM/shape_printer.hpp>

namespace exaDEM
{
  using namespace onika;
  struct subBox { size_t isub; int nbPoints;} ;
  OBB build_OBB(const std::span<vec3r> vec, double radius);

  /**
   * @brief Structure representing a polyhedral shape for DEM simulations.
   */
  struct shape
  {
    template <typename T> using VectorT = onika::memory::CudaMMVector<T>;

    VectorT<exanb::Vec3d> m_vertices;   ///< List of vertices of the shape
    exanb::Vec3d m_inertia_on_mass;     ///< Inertia vector divided by mass
    VectorT<OBB> m_obb_vertices;        ///< Oriented bounding boxes for each vertex (only for STL meshes)
    VectorT<OBB> m_obb_edges;           ///< OBBs for edges (only for STL meshes)
    VectorT<OBB> m_obb_faces;           ///< OBBs for faces (only for STL meshes)
    OBB obb;                            ///< Global OBB of the shape
    VectorT<int> m_edges;               ///< List of edges, stored as pairs of vertex indices
    VectorT<int> m_faces;               ///< List of faces, stored as sequences of vertex indices
    VectorT<int> m_offset_faces;        ///< Offsets for indexing faces in m_faces
    VectorT<double> m_face_area;           ///< Face area
    double m_radius;                    ///< Radius used for contact detection
    double m_volume;                    ///< Volume of the shape
    std::string m_name = "undefined";   ///< Name of the shape
    OBBtree<subBox> obbtree;            ///< Optional OBB tree for accelerated collision detection

    /**
     * @brief Default constructor.
     */
    shape()
    {
      m_faces.push_back(0); // init
    }

    /**
     * @brief Clear vertices, edges, faces and reset the name.
     */
    void clear()
    {
      m_vertices.clear();
      m_faces.clear();
      m_faces.resize(1);
      m_faces[0] = 0;
      m_edges.clear();
      m_name = "undefined";
    }

    // #######  pre compute functions ######## //

    /**
     * @brief Precompute OBBs for edges based on particle center and orientation.
     * @param particle_center Center of the particle.
     * @param particle_quat Orientation of the particle.
     */
    inline void pre_compute_obb_edges
      (const exanb::Vec3d &particle_center,
       const exanb::Quaternion &particle_quat);

    /**
     * @brief Precompute OBBs for faces based on particle center and orientation.
     * @param particle_center Center of the particle.
     * @param particle_quat Orientation of the particle.
     */
    inline void pre_compute_obb_faces(
        const exanb::Vec3d &particle_center,
        const exanb::Quaternion &particle_quat);

    /**
     * @brief Precompute OBBs for vertices using scratch memory.
     * @param scratch Temporary array of Vec3d.
     */
    inline void pre_compute_obb_vertices(const exanb::Vec3d *scratch);

    /**
     * @brief Precompute OBBs for edges using scratch memory.
     * @param scratch Temporary array of Vec3d.
     */
    inline void pre_compute_obb_edges(const exanb::Vec3d *scratch);

    /**
     * @brief Precompute OBBs for faces using scratch memory.
     * @param scratch Temporary array of Vec3d.
     */
    inline void pre_compute_obb_faces(const exanb::Vec3d *scratch);

    /*
     * @brief Enlarge all OBBs by a given value.
     * @param value Amount to enlarge.
     */
    inline void increase_obb(const double value);

    /**
     * @brief Compute all OBBs with scratch memory, particle center, and orientation.
     * @param scratch Temporary array of Vec3d.
     * @param particle_center Center of the particle.
     * @param particle_quat Orientation of the particle.
     */
    void compute_prepro_obb(exanb::Vec3d *scratch,
        const exanb::Vec3d &particle_center,
        const exanb::Quaternion &particle_quat);

    /**
     * @brief Get the volume of the shape.
     * @return Volume (asserts if not initialized)
     */
    ONIKA_HOST_DEVICE_FUNC
      inline double get_volume() const
      {
        assert(m_volume != 0 && "wrong initialisation");
        return m_volume;
      }

    /**
     * @brief Get the volume of the shape.
     * @param h homothety
     * @return Volume (asserts if not initialized)
     */
    ONIKA_HOST_DEVICE_FUNC
      inline double get_volume(double h) const
      {
        assert(m_volume != 0 && "wrong initialisation");
        return h * h * h * m_volume;
      }

    /**
     * @brief Get the inertia on mass vector.
     * @return Reference to inertia vector
     */
    ONIKA_HOST_DEVICE_FUNC
      inline const exanb::Vec3d &get_Im() { return m_inertia_on_mass; }

    /**
     * @brief Get the inertia on mass vector.
     * @param h homothety
     * @return Reference to inertia vector
     */
    ONIKA_HOST_DEVICE_FUNC
      inline const exanb::Vec3d get_Im(double h) { return h * h * m_inertia_on_mass; }

    /**
     * @brief Get the inertia on mass vector (const version).
     * @return Reference to inertia vector
     */
    ONIKA_HOST_DEVICE_FUNC
      inline const exanb::Vec3d &get_Im() const { return m_inertia_on_mass; }

    /**
     * @brief Get the inertia on mass vector (const version).
     * @param h homothety
     * @return Reference to inertia vector
     */
    ONIKA_HOST_DEVICE_FUNC
      inline const exanb::Vec3d get_Im(double h) const { return h * h * m_inertia_on_mass; }

    /**
     * @brief Get number of vertices.
     * @return Number of vertices
     */
    ONIKA_HOST_DEVICE_FUNC
      inline int get_number_of_vertices() { return onika::cuda::vector_size(m_vertices); }

    /**
     * @brief Get number of vertices (const version).
     * @return Number of vertices
     */
    ONIKA_HOST_DEVICE_FUNC
      inline int get_number_of_vertices() const { return onika::cuda::vector_size(m_vertices); }

    /**
     * @brief Get number of edges.
     * @return Number of edges
     */
    ONIKA_HOST_DEVICE_FUNC
      inline int get_number_of_edges() { return onika::cuda::vector_size(m_edges) / 2; }

    /**
     * @brief Get number of edges (const version).
     * @return Number of edges
     */
    ONIKA_HOST_DEVICE_FUNC
      inline int get_number_of_edges() const { return onika::cuda::vector_size(m_edges) / 2; }

    /**
     * @brief Get number of faces.
     * @return Number of faces
     */
    ONIKA_HOST_DEVICE_FUNC
      inline int get_number_of_faces()
      {
        const int *__restrict__ faces = onika::cuda::vector_data(m_faces);
        return faces[0];
      }

    /**
     * @brief Get number of faces (const version).
     * @return Number of faces
     */
    ONIKA_HOST_DEVICE_FUNC
      inline int get_number_of_faces() const
      {
        const int *__restrict__ faces = onika::cuda::vector_data(m_faces);
        return faces[0];
      }

    /**
     * @brief Access a vertex by index (non-const).
     * @param i Vertex index
     * @return Reference to the vertex
     */
    ONIKA_HOST_DEVICE_FUNC
      inline exanb::Vec3d &get_vertex(int i)
      {
        Vec3d *__restrict__ vertices = onika::cuda::vector_data(m_vertices);
        return vertices[i];
      }

    /**
     * @brief Access a vertex by index (const version).
     * @param i Vertex index
     * @return Const reference to the vertex
     */
    ONIKA_HOST_DEVICE_FUNC
      inline const exanb::Vec3d &get_vertex(int i) const
      {
        const Vec3d *__restrict__ vertices = onika::cuda::vector_data(m_vertices);
        return vertices[i];
      }

    /**
     * @brief Get the transformed vertex position given particle center and orientation (non-const).
     * @param i Vertex index
     * @param p Particle center
     * @param orient Particle orientation
     * @return Transformed vertex position
     */
    ONIKA_HOST_DEVICE_FUNC
      inline exanb::Vec3d get_vertex(
          int i, 
          const exanb::Vec3d &p, 
          const exanb::Quaternion &orient)
      {
        const Vec3d *__restrict__ vertices = onika::cuda::vector_data(m_vertices);
        return p + orient * vertices[i];
      }

    /**
     * @brief Get the transformed vertex position given particle center and orientation (const version).
     * @param i Vertex index
     * @param p Particle center
     * @param orient Particle orientation
     * @return Transformed vertex position
     */
    ONIKA_HOST_DEVICE_FUNC
      inline exanb::Vec3d get_vertex(int i, 
          const exanb::Vec3d &p, 
          const exanb::Quaternion &orient) const
      {
        const Vec3d *__restrict__ vertices = onika::cuda::vector_data(m_vertices);
        return p + orient * vertices[i];
      }

    /**
     * @brief Get an edge by index (non-const).
     * @param i Edge index
     * @return Pair of vertex indices defining the edge
     */
    ONIKA_HOST_DEVICE_FUNC
      inline std::pair<int, int> get_edge(int i)
      {
        const int *__restrict__ edges = onika::cuda::vector_data(m_edges);
        return {edges[2 * i], edges[2 * i + 1]};
      }

    /**
     * @brief Get an edge by index (const version).
     * @param i Edge index
     * @return Pair of vertex indices defining the edge
     */
    ONIKA_HOST_DEVICE_FUNC
      inline std::pair<int, int> get_edge(int i) const
      {
        const int *__restrict__ edges = onika::cuda::vector_data(m_edges);
        return {edges[2 * i], edges[2 * i + 1]};
      }

    /**
     * @brief Get pointer to the faces array (non-const).
     * @return Pointer to integer array of faces
     */
    ONIKA_HOST_DEVICE_FUNC
      inline int *get_faces()
      {
        int *faces = onika::cuda::vector_data(m_faces);
        return faces;
      }

    /**
     * @brief Get pointer to the faces array (const version).
     * @return Pointer to integer array of faces
     */
    ONIKA_HOST_DEVICE_FUNC
      inline int *get_faces() const
      {
        const int *faces = onika::cuda::vector_data(m_faces);
        return (int *)faces;
      }

    /**
     * @brief Compute offsets for each face in the flat faces array.
     */
    inline void compute_offset_faces()
    {
      int n = this->get_number_of_faces();
      m_offset_faces.resize(n);
      int *start = this->get_faces() ;
#     pragma omp parallel for
      for(int i = 0 ; i < n ; i++)
      {
        int * ptr = start + 1; // the first element is the total number of face
        int acc = 1;
        for (int it = i; it > 0; it--)
        {
          acc += ptr[0] + 1; // ptr[0] contains the number of vertices of this face
          ptr += ptr[0] + 1;
        }
        m_offset_faces[i] = acc; 
      } 
    }


    /**
     * @brief Get a face by index.
     * @param i Face index
     * @return Pair {pointer to vertex indices, number of vertices}
     */
    ONIKA_HOST_DEVICE_FUNC
      const std::pair<int *, int> get_face(int i)
      {
        auto * __restrict__ data =  onika::cuda::vector_data(m_offset_faces);
        int *ptr = this->get_faces();
        int index = data[i];
        return {ptr + index + 1, ptr[index]};
      }

    /**
     * @brief Get a face by index.
     * @param i Face index
     * @return Pair {pointer to vertex indices, number of vertices}
     */
    ONIKA_HOST_DEVICE_FUNC
      const std::pair<int *, int> get_face(int i) const
      {
        auto * __restrict__ data =  onika::cuda::vector_data(m_offset_faces);
        int *ptr = this->get_faces();
        int index = data[i];
        return {ptr + index + 1, ptr[index]};
      }

    /**
     * @brief Get a face by index.
     * @param i Face index
     * @return Pair {pointer to vertex indices, number of vertices}
     */
    ONIKA_HOST_DEVICE_FUNC
      double get_face_area(int i) const
      {
        const auto * ptr =  onika::cuda::vector_data(m_face_area);
        return ptr[i];
      }

    /**
     * @brief Get the oriented and translated OBB of an edge.
     * @param position Particle center position
     * @param index Edge index
     * @param orientation Particle orientation
     * @return Transformed OBB of the edge
     */
    ONIKA_HOST_DEVICE_FUNC
      inline OBB get_obb_edge(
          const exanb::Vec3d &position, 
          const size_t index, 
          const exanb::Quaternion& orientation) const
      {
        OBB res = m_obb_edges[index];
        res.rotate(conv_to_quat(orientation));
        res.translate(conv_to_vec3r(position));
        return res;
      }

    /**
     * @brief Get the oriented and translated OBB of an edge.
     * @param position Particle center position
     * @param index Edge index
     * @param orientation Particle orientation
     * @return Transformed OBB of the edge
     */
    ONIKA_HOST_DEVICE_FUNC
      inline OBB get_obb_face(
          const exanb::Vec3d &position, 
          const size_t index, 
          const exanb::Quaternion& orientation) const
      {
        OBB res = m_obb_faces[index];
        res.rotate(conv_to_quat(orientation));
        res.translate(conv_to_vec3r(position));
        return res;
      }

    /**
     * @brief Add a vertex to the shape.
     * @param vertex 3D position of the vertex
     */
    void add_vertex(const exanb::Vec3d &vertex) 
    { 
      m_vertices.push_back(vertex); 
    }

    /**
     * @brief Add an edge to the shape.
     * @param i Index of the first vertex (>= 0)
     * @param j Index of the second vertex (>= 0)
     */  
    void add_edge(
        int i, 
        int j)
    {
      assert(i >= 0 && "add negatif vertex");
      assert(j >= 0 && "add negatif vertex");
      m_edges.push_back(i);
      m_edges.push_back(j);
    }

    /**
     * @brief Add a face to the shape.
     * @param n Number of vertices in the face (must be > 0)
     * @param data Pointer to an array of vertex indices
     */
    void add_face(size_t num_vertices, int * const vertex_indices)
    {
      assert(num_vertices != 0);
      m_faces[0]++;
      const size_t old_size = m_faces.size();
      m_faces.resize(old_size + num_vertices + 1); // number of vertex + 1 storage to this number
      m_faces[old_size] = num_vertices;
      for (size_t it = 0; it < num_vertices; it++)
      {
        m_faces[old_size + 1 + it] = vertex_indices[it];
      }
    }

    /**
     * @brief retur, the minkowski radius used for detection.
     * @param radius Minkowsku radius
     */
    double minskowski() { return m_radius; }
    double minskowski() const { return m_radius; }

    /**
     * @brief Set the minkowski radius used for detection.
     * @param radius Minkowsku radius
     */
    void add_radius(const double radius) { m_radius = radius; }

    /**
     * @brief Compute the maximum cutoff radius (distance from origin + Minkowski radius)
     * @return Maximum cutoff radius
     */
    double compute_max_rcut() const
    {
      const size_t n_vertices = this->get_number_of_vertices();
      double rcut = 0;
      for (size_t vertex_idx = 0; vertex_idx < n_vertices; vertex_idx++)
      {
        const auto &vertex = this->get_vertex(vertex_idx);
        const double d = exanb::norm(vertex) + m_radius;
        rcut = std::max(rcut, d);
      }
      assert(rcut != 0);
      return rcut;
    }
    /**
     * @brief Apply a function to all vertices of the shape (non-const version).
     * 
     * @tparam Func Type of the function to apply. Should take a Vec3d reference as first argument.
     * @tparam Args Additional arguments passed to the function
     * @param func Function to apply on each vertex
     * @param args Additional arguments forwarded to func
     */
    template <typename Func, typename... Args> void for_all_vertices(Func &func, Args &&...args)
    {
      const size_t n = this->get_number_of_vertices();
      for (size_t it = 0; it < n; it++)
      {
        auto &vertex = this->get_vertex(it);
        func(vertex, std::forward<Args>(args)...);
      }
    }

    /**
     * @brief Apply a function to all vertices of the shape (const version).
     * 
     * @tparam Func Type of the function to apply. Should take a Vec3d reference as first argument.
     * @tparam Args Additional arguments passed to the function
     * @param func Function to apply on each vertex
     * @param args Additional arguments forwarded to func
     */
    template <typename Func, typename... Args> void for_all_vertices(Func &func, Args &&...args) const
    {
      const size_t n = this->get_number_of_vertices();
      for (size_t it = 0; it < n; it++)
      {
        auto &vertex = this->get_vertex(it);
        func(vertex, std::forward<Args>(args)...);
      }
    }

    /**
     * @brief Apply a function to all edges of the shape (non-const version).
     * 
     * @tparam Func Type of the function to apply. Should take two vertex indices as first arguments.
     * @tparam Args Additional arguments passed to the function
     * @param func Function to apply on each edge
     * @param args Additional arguments forwarded to func
     */
    template <typename Func, typename... Args>
      void for_all_edges(Func &func, Args &&...args)
      {
        const size_t n_edges = this->get_number_of_edges();
        for (size_t edge_idx = 0; edge_idx < n_edges; edge_idx++)
        {
          auto [v0, v1] = this->get_edge(edge_idx);
          func(v0, v1, std::forward<Args>(args)...);
        }
      }

    /**
     * @brief Apply a function to all edges of the shape (const version).
     *
     * @tparam Func Type of the function to apply. Should take two vertex indices as first arguments.
     * @tparam Args Additional arguments passed to the function
     * @param func Function to apply on each edge
     * @param args Additional arguments forwarded to func
     */
    template <typename Func, typename... Args>
      void for_all_edges(Func &func, Args &&...args) const
      {
        const size_t n_edges = this->get_number_of_edges();
        for (size_t edge_idx = 0; edge_idx < n_edges; edge_idx++)
        {
          auto [v0, v1] = this->get_edge(edge_idx);
          func(v0, v1, std::forward<Args>(args)...);
        }
      }

    /**
     * @brief Apply a function to all faces of the shape (non-const version).
     * 
     * @tparam Func Type of the function to apply. Should take face size and pointer to vertex indices as first arguments.
     * @tparam Args Additional arguments passed to the function
     * @param func Function to apply on each face
     * @param args Additional arguments forwarded to func
     */
    template <typename Func, typename... Args>
      void for_all_faces(Func &func, Args &&...args)
      {
        const size_t n_faces = this->get_number_of_faces();
        for (size_t face_idx = 0; face_idx < n_faces; face_idx++)
        {
          auto [vertices_ptr, face_size] = this->get_face(face_idx);
          func(face_size, vertices_ptr, std::forward<Args>(args)...);
        }
      }

    /**
     * @brief Apply a function to all faces of the shape (const version).
     *
     * @tparam Func Type of the function to apply. Should take face size and pointer to vertex indices as first arguments.
     * @tparam Args Additional arguments passed to the function
     * @param func Function to apply on each face
     * @param args Additional arguments forwarded to func
     */
    template <typename Func, typename... Args>
      void for_all_faces(Func &func, Args &&...args) const
      {
        const size_t n_faces = this->get_number_of_faces();
        for (size_t face_idx = 0; face_idx < n_faces; face_idx++)
        {
          auto [vertices_ptr, face_size] = this->get_face(face_idx);
          func(face_size, vertices_ptr, std::forward<Args>(args)...);
        }
      }

    /**
     * @brief Rescale the shape by a given factor.
     * @param scale Scaling factor
     * @param enable_minskowski_rescaling Enable to rescale the minskowski radius
     */
    void rescale(const double scale, const bool enable_minskowski_rescaling)
    {
      if(enable_minskowski_rescaling) 
      {
        m_radius *= scale;
      }

      auto scale_vertices = [] (exanb::Vec3d& v, double s) { v = s * v; };
      for_all_vertices(scale_vertices, scale);
      m_volume = this->get_volume(scale);
      m_inertia_on_mass = this->get_Im(scale);
      std::vector<vec3r> vertices;
      vertices.resize(m_vertices.size());
      for(size_t vid = 0; vid < m_vertices.size() ; vid++) vertices[vid] = conv_to_vec3r(get_vertex(vid));
      obb = build_OBB(vertices, m_radius); 
    }

    /**
     * @brief Compute the surface area of the shape.
     * @return Total surface area
     */
    double compute_surface() const
    {
      double surface = 0.0;
      const size_t n_faces = this->get_number_of_faces();

#pragma omp parallel for reduction(+:surface)
      for (size_t face_idx = 0; face_idx < n_faces; face_idx++)
      {
        auto [vertices_ptr, face_size] = this->get_face(face_idx);
        const Vec3d& v0 = m_vertices[vertices_ptr[0]];

        if (face_size == 3)
        {
          for (int j = 1; j < face_size - 1; j++)
          {
            const size_t k = j + 1;
            const Vec3d v1 = m_vertices[vertices_ptr[j]] - v0;
            const Vec3d v2 = m_vertices[vertices_ptr[k]] - v0;
            surface += 0.5 * exanb::norm(exanb::cross(v1, v2));
          }
        }
      }

      return surface;
    }

    /**
     * @brief Shifts all mesh vertices by a given vector.
     *
     * @param shift The shift vector applied to each vertex.
     */
    void shift_vertices(const Vec3d& shift)
    {
      auto shift_vertex = [] (Vec3d& vertex, const Vec3d& shift) {
        vertex -= shift;
      };
      for_all_vertices(shift_vertex, shift);
    }


    // CPU
		// could be optimized without sort ...
		/**
		 * @brief Finds the face matching a given set of vertex IDs.
		 * @param vertices Vertex IDs to identify (modified by sort).
		 * @return Matching face ID.
		 */
		uint16_t identify_face(std::span<int> vertices) const
		{
			std::vector<int> input(vertices.begin(), vertices.end());
			std::sort(input.begin(), input.end());

      int n_vertices = vertices.size();
			for (uint16_t fid = 0; fid < get_number_of_faces(); ++fid)
			{
				auto [data, size] = get_face(fid);
				if (size != n_vertices)
					continue;

				std::vector<int> face(data, data + size);
				std::sort(face.begin(), face.end());

				if (input == face)
					return fid;
			}

			std::string msg = "Impossible to identify a face in shape: " + m_name +".\n";
			msg += "Vertices ID are: [ ";
			for (size_t i=0 ; i<vertices.size() ; i++) msg += std::to_string(vertices[i]) + " ";
			msg += "]";
			color_log::error("shape::identify_face", msg);
		}

		/// OBBTree Section

		/**
		 * @brief Build an OBB tree for the shape (faces, edges, vertices).
		 *
		 * Each face, edge, and vertex is wrapped in an OBBbundle, and then the tree
		 * is recursively built. The OBBs are enlarged by the particle radius.
		 */
		void buildOBBtree()
		{
			obbtree.reset(obbtree.root);
			std::vector<OBBbundle<subBox>> obb_bundles;

			// Build OBBs for faces
			for (int face_idx = 0; face_idx < this->get_number_of_faces(); ++face_idx)
			{
				auto [vertex_ids, n_vertices] = this->get_face(face_idx);
				OBBbundle<subBox> bundle;
				bundle.data.isub = face_idx;
				bundle.data.nbPoints = n_vertices;

				for (int vi = 0; vi < n_vertices; vi++)
					bundle.points.push_back(conv_to_vec3r(m_vertices[vertex_ids[vi]]));

				std::vector<OBBbundle<subBox>> single_bundle{bundle};
				bundle.obb = OBBtree<subBox>::fitOBB(single_bundle, m_radius);
				obb_bundles.push_back(bundle);
			}

			// Build OBBs for edges
			for (int edge_idx = 0; edge_idx < this->get_number_of_edges(); ++edge_idx)
			{
				auto [v0, v1] = this->get_edge(edge_idx);
				OBBbundle<subBox> bundle;
				bundle.data.isub = edge_idx;
				bundle.data.nbPoints = 2;
				bundle.points.push_back(conv_to_vec3r(m_vertices[v0]));
				bundle.points.push_back(conv_to_vec3r(m_vertices[v1]));

				std::vector<OBBbundle<subBox>> single_bundle{bundle};
				bundle.obb = OBBtree<subBox>::fitOBB(single_bundle, m_radius);
				obb_bundles.push_back(bundle);
			}

			// Build OBBs for vertices
			for (int vert_idx = 0; vert_idx < this->get_number_of_vertices(); ++vert_idx)
			{
				OBBbundle<subBox> bundle;
				bundle.data.isub = vert_idx;
				bundle.data.nbPoints = 1;
				bundle.points.push_back(conv_to_vec3r(m_vertices[vert_idx]));

				std::vector<OBBbundle<subBox>> single_bundle{bundle};
				bundle.obb = OBBtree<subBox>::fitOBB(single_bundle, m_radius);
				obb_bundles.push_back(bundle);
			}

			// Recursively build the OBB tree
			obbtree.root = OBBtree<subBox>::recursiveBuild(obbtree.root, obb_bundles, m_radius);
		}

		/// IO Section
		/**
		 * @brief Print all vertices of the shape to the logging output.
		 */
		void print_vertices()
		{
			int vertex_idx = 0;
			auto printer = [&vertex_idx](exanb::Vec3d &v)
			{
				lout << "Vertex[" << vertex_idx++ << "]: [" << v.x << "," << v.y << "," << v.z << "]" << std::endl;
			};

			lout << "Number of vertices = " << this->get_number_of_vertices() << std::endl;
			for_all_vertices(printer);
		}

		/**
		 * @brief Print all edges of the shape to the logging output.
		 */
		void print_edges()
		{
			int edge_idx = 0;
			auto printer = [&edge_idx](int v0, int v1)
			{
				lout << "Edge[" << edge_idx++ << "]: [" << v0 << "," << v1 << "]" << std::endl;
			};

			if (this->get_number_of_edges() == 0)
			{
				lout << "No edges" << std::endl;
			}
			else
			{
				lout << "Number of edges = " << this->get_number_of_edges() << std::endl;
				for_all_edges(printer);
			}
		}

		/**
		 * @brief Print all faces of the shape to the logging output.
		 */
		void print_faces()
		{
			int face_idx = 0;
			auto printer = [&face_idx](int n_vertices, int *vertex_indices)
			{
				lout << "Face[" << face_idx++ << "]: ";
				for (int i = 0; i < n_vertices; i++)
				{
					lout << vertex_indices[i];
					if (i < n_vertices - 1) lout << ", ";
				}
				lout << std::endl;
			};

			if (this->get_number_of_faces() == 0)
			{
				lout << "No faces" << std::endl;
			}
			else
			{
				lout << "Number of faces = " << this->get_number_of_faces() << std::endl;
				for_all_faces(printer);
			}
		}

		/**
		 * @brief Print full shape information (name, radius, inertia, volume) and all vertices, edges, and faces.
		 */
		inline void print()
		{
			lout << std::endl;
			lout << "======= Shape Configuration =====" << std::endl;
			lout << "Shape Name        = " << this->m_name << std::endl;
			lout << "Shape Radius      = " << this->m_radius << std::endl;
			lout << "Shape I/m         = [" << this->m_inertia_on_mass << "]" << std::endl;
			lout << "Shape Volume      = " << this->m_volume << std::endl;
			print_vertices();
			print_edges();
			print_faces();
			lout << "=================================" << std::endl << std::endl;
		}

		/**
		 * @brief Export the shape to a Paraview-compatible VTK file.
		 */
		inline void write_paraview()
		{
			ldbg << " writting paraview for shape " << this->m_name << std::endl;
			std::string name = m_name + ".vtk";
			std::ofstream outFile(name);
			if (!outFile)
			{
				color_log::error("Shape::write_paraview", "Impossible to create an output file!", false);
				color_log::error("Shape::write_paraview", "Impossible to open the file: " + name, false);
				return;
			}
			outFile << "# vtk DataFile Version 3.0" << std::endl;
			outFile << "Spheres" << std::endl;
			outFile << "ASCII" << std::endl;
			outFile << "DATASET POLYDATA" << std::endl;
			outFile << "POINTS " << this->get_number_of_vertices() << " float" << std::endl;
			auto writer_v = [](exanb::Vec3d &v, std::ofstream &out) { out << v.x << " " << v.y << " " << v.z << std::endl; };

			for_all_vertices(writer_v, outFile);

			outFile << std::endl;

			outFile << "LINES " << this->get_number_of_edges() << " " << 3*this->get_number_of_edges() << std::endl;

			auto writer_e = [] (int a, int b, std::ofstream &out)
			{
				out << "2 " << a << " " << b << std::endl;
			};

			for_all_edges(writer_e, outFile);

			int count_polygon_size = this->get_number_of_faces();
			int count_polygon_table_size = 0;
			int *ptr = this->m_faces.data() + 1;
			for (int it = 0; it < count_polygon_size; it++)
			{
				count_polygon_table_size += ptr[0] + 1; // number of vertices + vertex idexes
				ptr += ptr[0] + 1;                      // -> next face
			}
			outFile << std::endl;

			outFile << "POLYGONS " << count_polygon_size << " " << count_polygon_table_size << std::endl;
			auto writer_f = [](const size_t size, const int *data, std::ofstream &out)
			{
				out << size;
				for (size_t it = 0; it < size; it++)
					out << " " << data[it];
				out << std::endl;
			};
			for_all_faces(writer_f, outFile);
		}

		/**
		 * @brief Export the shape at a given timestep (position + orientation) to Paraview.
		 * @param path Output directory.
		 * @param timestep Current timestep (used in filename).
		 * @param center Translation vector of the shape.
		 * @param quat Orientation quaternion of the shape.
		 */
		inline void write_move_paraview(
				std::string path, 
				int timestep, 
				Vec3d &center, 
				Quaternion &quat)
		{
			std::string time = std::to_string(timestep);
			ldbg << " writting paraview for shape " << this->m_name << " timestep: " << time << std::endl;
			std::string name = path + m_name + "_" + time + ".vtk";
			std::ofstream outFile(name);
			if (!outFile)
			{
				color_log::error("Shape::write_move_paraview", "Impossible to create the output file: " + name, false);
				return;
			}
			outFile << "# vtk DataFile Version 3.0" << std::endl;
			outFile << "Spheres" << std::endl;
			outFile << "ASCII" << std::endl;
			outFile << "DATASET POLYDATA" << std::endl;
			outFile << "POINTS " << this->get_number_of_vertices() << " float" << std::endl;
			auto writer_v = [](const exanb::Vec3d &v, const exanb::Vec3d &center, const exanb::Quaternion &Q, std::ofstream &out)
			{
				exanb::Vec3d Vertex = center + Q * v;
				out << Vertex.x << " " << Vertex.y << " " << Vertex.z << std::endl;
			};

			for_all_vertices(writer_v, center, quat, outFile);

			outFile << std::endl;
			int count_polygon_size = this->get_number_of_faces();
			int count_polygon_table_size = 0;
			int *ptr = this->m_faces.data() + 1;
			for (int it = 0; it < count_polygon_size; it++)
			{
				count_polygon_table_size += ptr[0] + 1; // number of vertices + vertex idexes
				ptr += ptr[0] + 1;                      // -> next face
			}

			outFile << "POLYGONS " << count_polygon_size << " " << count_polygon_table_size << std::endl;
			auto writer_f = [](const size_t size, const int *data, std::ofstream &out)
			{
				out << size;
				for (size_t it = 0; it < size; it++)
					out << " " << data[it];
				out << std::endl;
			};
			for_all_faces(writer_f, outFile);
		}
	};
}; // namespace exaDEM

#include <exaDEM/shape_prepro.hpp>
