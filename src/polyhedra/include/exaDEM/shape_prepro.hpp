#pragma once

#include <exaDEM/shape.hpp>
#include <exaDEM/shape_detection.hpp>

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
		obb.e[0] = r;
		obb.e[1] = u;
		obb.e[2] = f;
		obb.extent = 0.5 * (maxim - minim);

		obb.enlarge(radius);  // Add the Minskowski radius
		return obb;
	}

//#define OLD_VERSION

	// general functon;
	inline OBB build_obb_edge(const Vec3d& position, const int index, const shape* shp, const exanb::Quaternion& orientation)
	{
		const double ext = shp->m_radius;
		auto [first, second] = shp->get_edge(index);
		const Vec3d vf = shp->get_vertex (first,  position, orientation);
		const Vec3d vs = shp->get_vertex (second, position, orientation); 
		std::vector<vec3r> v = {conv_to_vec3r(vf), conv_to_vec3r(vs)};
		OBB res = build_OBB (v, ext);
		return res;
#ifdef OLD_VERSION
		// TODO use orientation
		OBB res;
		const Vec3d center = (vf + vs) / 2;
		res.center = { center.x , center.y, center.z };

		constexpr int DIM = 3;
		// TODO:  add obb orientation HERE
		// Use distance between vf and vs and apply rotation

		const auto v = conv_to_vec3r (vf);

		for ( int dim = 0 ; dim < DIM ; dim++ )
		{
			res.extent[dim] = std::abs ( res.center[dim] - v [dim] );
		}

		res.enlarge( ext );
		return res;
#endif
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
#ifdef OLD_VERSION
		OBB obb_face;
		assert ( nf > 0 );

		constexpr int DIM = 3;
		for ( int dim = 0 ; dim < DIM ; dim++ )
		{
			double acc = 0.0;
			for (int f = 0 ; f < nf ; f++)
			{
				vec3r v = conv_to_vec3r (shp->get_vertex(data[f], position, orientation));
				acc += v[dim];
			}
			obb_face.center[dim] = acc / nf;
		}

		// TODO:  add obb orientation HERE
		// Use distance between vf and vs and apply rotation
		// TODO : we could do it with only one loop
		for ( int dim = 0 ; dim < DIM ; dim++ )
		{
			double acc = 0.0;
			for (int f = 0 ; f < nf ; f++)
			{
				vec3r v = conv_to_vec3r (shp->get_vertex(data[f], position, orientation));
				acc = std::max ( acc , std::abs( v[dim] - obb_face.center[dim] ));
			}
			obb_face.extent[dim] = acc;
		}

		obb_face.enlarge( ext );
		return obb_face;
#endif
	}

	inline void shape::pre_compute_obb_edges()
	{
		// This function could be optimized by avoiding to use `position` and `orientation` in `build_obb_edge`
		const size_t size = this->get_number_of_edges();
		m_obb_edges.resize(size);

		//const exanb::Vec3d vnull      = {0,0,0};
		const exanb::Quaternion onull = {1,0,0,0};
		const exanb::Vec3d center = conv_to_Vec3d (this->obb.center);
#pragma omp parallel for
		for ( size_t i = 0 ; i < size ; i++)
		{
			m_obb_edges[i] = build_obb_edge (center, i, this, onull);
		} 
	}

	inline void shape::pre_compute_obb_faces()
	{
		// This function could be optimized by avoiding to use `position` and `orientation` in `build_obb_face`
		const size_t size = this->get_number_of_faces();
		m_obb_faces.resize(size);
		//const exanb::Vec3d vnull      = {0,0,0};
		const exanb::Quaternion onull = {1,0,0,0};
		const exanb::Vec3d center = conv_to_Vec3d (this->obb.center);

#pragma omp parallel for
		for ( size_t i = 0 ; i < size ; i++)
		{
			m_obb_faces[i] = build_obb_face (center, i, this, onull);
		}
	}
}