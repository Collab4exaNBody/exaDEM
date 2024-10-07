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

#include <exaDEM/shape/shape.hpp>
#include <exaDEM/shape/shape_detection.hpp>

namespace exaDEM
{
	using namespace exanb;

	inline OBB build_OBB ( const std::vector<vec3r>& vec, double radius )
	{
		OBB obb;
		vec3r mu;
		mat9r C;
		for (size_t i = 0; i < vec.size(); i++) {
			mu += vec[i];
		}
		mu /= (double)vec.size();

		// loop over the points again to build the
		// covariance matrix.  Note that we only have
		// to build terms for the upper trianglular
		// portion since the matrix is symmetric
		double cxx = 0.0, cxy = 0.0, cxz = 0.0, cyy = 0.0, cyz = 0.0, czz = 0.0;
		for (size_t i = 0; i < vec.size(); i++) {
			vec3r p = vec[i];
			cxx += p.x * p.x - mu.x * mu.x;
			cxy += p.x * p.y - mu.x * mu.y;
			cxz += p.x * p.z - mu.x * mu.z;
			cyy += p.y * p.y - mu.y * mu.y;
			cyz += p.y * p.z - mu.y * mu.z;
			czz += p.z * p.z - mu.z * mu.z;
		}


		// now build the covariance matrix
		C.xx = cxx;
		C.xy = cxy;
		C.xz = cxz;
		C.yx = cxy;
		C.yy = cyy;
		C.yz = cyz;
		C.zx = cxz;
		C.zy = cyz;
		C.zz = czz;

		// ==== set the OBB parameters from the covariance matrix
		// extract the eigenvalues and eigenvectors from C
		mat9r eigvec;
		vec3r eigval;
		C.sym_eigen(eigvec, eigval);

		// find the right, up and forward vectors from the eigenvectors
		vec3r r(eigvec.xx, eigvec.yx, eigvec.zx);
		vec3r u(eigvec.xy, eigvec.yy, eigvec.zy);
		vec3r f(eigvec.xz, eigvec.yz, eigvec.zz);
		r.normalize();
		u.normalize(), f.normalize();

		// now build the bounding box extents in the rotated frame
		vec3r minim(1e20, 1e20, 1e20), maxim(-1e20, -1e20, -1e20);
		for (size_t i = 0; i < vec.size(); i++) {
			vec3r p_prime(r * vec[i], u * vec[i], f * vec[i]);
			if (minim.x > p_prime.x) minim.x = p_prime.x;
			if (minim.y > p_prime.y) minim.y = p_prime.y;
			if (minim.z > p_prime.z) minim.z = p_prime.z;
			if (maxim.x < p_prime.x) maxim.x = p_prime.x;
			if (maxim.y < p_prime.y) maxim.y = p_prime.y;
			if (maxim.z < p_prime.z) maxim.z = p_prime.z;
		}

		// set the center of the OBB to be the average of the
		// minimum and maximum, and the extents be half of the
		// difference between the minimum and maximum
		obb.center = eigvec * (0.5 * (maxim + minim));
		obb.e1 = r;
		obb.e2 = u;
		obb.e3 = f;
		obb.extent = 0.5 * (maxim - minim);

		obb.enlarge(radius);  // Add the Minskowski radius
		return obb;
	}

//#define OLD_VERSION

	// general functon;
	inline OBB build_obb_vertex(const Vec3d& position, const int index, const shape* shp, const exanb::Quaternion& orientation)
	{
		const double ext = shp->m_radius;
		auto vertex = shp->get_vertex(index, position, orientation);
		std::vector<vec3r> v = {conv_to_vec3r(vertex)};
		OBB res = build_OBB (v, ext);
		return res;
	}

	inline OBB build_obb_edge(const Vec3d& position, const int index, const shape* shp, const exanb::Quaternion& orientation)
	{
		const double ext = shp->m_radius;
		auto [first, second] = shp->get_edge(index);
		const Vec3d vf = shp->get_vertex (first,  position, orientation);
		const Vec3d vs = shp->get_vertex (second, position, orientation); 
		std::vector<vec3r> v = {conv_to_vec3r(vf), conv_to_vec3r(vs)};
		OBB res = build_OBB (v, ext);
		return res;
	}

	inline OBB build_obb_face(const Vec3d& position, const int index, const shape* shp, const exanb::Quaternion& orientation)
	{
		const double ext = shp->m_radius;
		const auto [data, nf] = shp->get_face(index);
		std::vector<vec3r> v;
		for(int i = 0 ; i < nf ; i++) 
		{ 
			v.push_back(conv_to_vec3r(shp->get_vertex(data[i], position, orientation))); 
		}
		OBB res = build_OBB (v, ext);
		return res;
	}

	inline void shape::pre_compute_obb_vertices(const Vec3d& particle_center, const exanb::Quaternion& particle_quat)
	{
		// This function could be optimized by avoiding to use `position` and `orientation` in `build_obb_face`
		const size_t size = this->get_number_of_vertices();
		m_obb_vertices.resize(size);
		const exanb::Vec3d center = conv_to_Vec3d (this->obb.center) + particle_center;

		ldbg << "obb [vertices] = " << size << std::endl;

#pragma omp parallel for schedule (static)
		for ( size_t i = 0 ; i < size ; i++)
		{
			m_obb_vertices[i] = build_obb_vertex (center, i, this, particle_quat);
		}
	}

	inline void shape::pre_compute_obb_edges(const Vec3d& particle_center, const exanb::Quaternion& particle_quat)
	{
		// This function could be optimized by avoiding to use `position` and `orientation` in `build_obb_edge`
		const size_t size = this->get_number_of_edges();
		m_obb_edges.resize(size);

		//const exanb::Vec3d vnull      = {0,0,0};
		const exanb::Vec3d center = conv_to_Vec3d (this->obb.center) + particle_center;
		ldbg << "obb [edges]    = " << size << std::endl;
#pragma omp parallel for schedule(static)
		for ( size_t i = 0 ; i < size ; i++)
		{
			m_obb_edges[i] = build_obb_edge (center, i, this, particle_quat);
		} 
	}

	inline void shape::pre_compute_obb_faces(const Vec3d& particle_center, const exanb::Quaternion& particle_quat)
	{
		// This function could be optimized by avoiding to use `position` and `orientation` in `build_obb_face`
		const size_t size = this->get_number_of_faces();
		m_obb_faces.resize(size);
		const exanb::Vec3d center = conv_to_Vec3d (this->obb.center) + particle_center;
		ldbg << "obb [faces]    = " << size << std::endl;

#pragma omp parallel for schedule(static)
		for ( size_t i = 0 ; i < size ; i++)
		{
			m_obb_faces[i] = build_obb_face (center, i, this, particle_quat);
		}
	}

	inline void shape::increase_obb ( const double value )
	{
#pragma omp parallel
		{
#pragma omp for schedule(static) nowait
			for (size_t i = 0 ; i < m_obb_vertices.size() ; i++)
			{
				m_obb_vertices[i].enlarge(value);
			}
#pragma omp for schedule(static) nowait
			for (size_t i = 0 ; i < m_obb_edges.size() ; i++)
			{
				m_obb_edges[i].enlarge(value);
			}
#pragma omp for schedule(static)
			for (size_t i = 0 ; i < m_obb_faces.size() ; i++)
			{
				m_obb_faces[i].enlarge(value);
			}
		}
	}
}
