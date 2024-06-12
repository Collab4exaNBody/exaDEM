#pragma once

#include <exaDEM/face.h>
#include <sstream>
#include <exanb/core/basic_types.h>

#include <onika/cuda/cuda.h> // mots cles specifiques
#include <onika/memory/allocator.h> // cudaMMVector

#include <cfloat>
#include "OBB.hpp"

namespace exaDEM
{
	using namespace exanb;
	/**
	* @brief Represents a collection of 3D meshes consisting of faces
	*
	* The 'stl_meshes' struct represents a collection of 3D meshes composed of faces. Is an optimization of the 'stl_mesh' data layout to fit GPU computing.
	* It provides methods fro adding faces to the mesh, accessing the mesh data, and retrieving individual faces.
	
	*/			

	/**
	 * @brief Represents a 3D mesh consisting of faces.
	 *
	 * The `stl_mesh` struct represents a 3D mesh composed of faces. It provides methods for adding faces to the mesh,
	 * accessing the mesh data, and retrieving individual faces.
	 	 
	 */
	 
	
	
	struct stl_meshes
	{
		int meshes= 0;
		int start_mesh_box_construct= 0;
		std::vector< double > vx;/**< x coordinate of the faces's vertices.*/
		std::vector< double > vy;/**< y coordinate of the faces's vertices.*/
		std::vector< double > vz;/**< z coordinate of the faces's vertices.*/
		std::vector< double > nx;/**< x coordinate of the faces's normal vector.*/
		std::vector< double > ny;/**< y coordinate of the faces's normal vector.*/
		std::vector< double > nz;/**< z coordiante of the faces's normal vector.*/
		std::vector< double > offsets;/**< Faces's offset.*/
		std::vector< int > start;/**< Vector used to keep track of the vertices's indexes*/
		std::vector< int > end;/**< Vector used to keep track of the vertices's indexes.*/
		std::vector< int > nb_vertices;
		std::vector< Box > m_boxes;/**< Faces's boxes.*/
		std::vector< OBB > m_obbs;
		std::vector< std::vector < int >> indexes;
		std::vector< std::vector < int >> indexes2;
		
		
		//float* elements;
		
		/**
		 * @brief Reads mesh data from an STL file and populates the mesh.
		 *
		 * The `read_stl` function reads mesh data from an STL file specified by `file_name` and populates the mesh with
		 * vertices and faces. It also calculates the number of vertices and faces in the mesh and provides information about
		 * the mesh's characteristics.
		 *
		 * @param file_name The name of the STL file to read.
		 */
		void read_stl(std::string file_name)
		{
			start_mesh_box_construct= start.size();
			std::ifstream input( file_name.c_str() );
			std::string first;
			std::vector<Vec3d> vertices;
			Vec3d vertex;
			int nv;
			int nv2=0;
			if(meshes==0){ nv = 0;} else{ nv = end[end.size() - 1];}
			int nf = 0;
			
			for( std::string line; getline( input, line ); )
			{
				input >> first;
				if(first == "outer")
				{
					int nv_2=0;
					start.push_back(nv);
					bool build_face = true;
					while(build_face)
					{
						
						getline(input, line);
						input >> first;
						if(first == "vertex")
						{
							input >> vertex.x >> vertex.y >> vertex.z;
							vertices.push_back(vertex);	
							nv++;
							nv_2++;
							nv2++;
						}
						else if (first != "endloop")
						{
							std::cout << "error when reading stl file, it should be endloop and not " << first << std::endl;
							build_face = false;
						}
						else { 
							build_face = false;
						}
					}
					auto [_normal, _offset, _exist] = compute_normal_and_offset(vertices);
					offsets.push_back(_offset);
					nx.push_back(_normal.x);
					ny.push_back(_normal.y);
					nz.push_back(_normal.z);
					for(auto v: vertices){
						vx.push_back(v.x);
						vy.push_back(v.y);
						vz.push_back(v.z);
					}
					end.push_back(nv);
					vertices.clear();
					nf++;
					nb_vertices.push_back(nv_2);
				}
			}
			meshes++;
			std::cout << "Mesh: " << file_name << " - number of vertices: " << nv2 << " - number of faces: " << nf << std::endl;
		}
		
		std::tuple<Vec3d, double, bool> compute_normal_and_offset(std::vector<Vec3d> vertices) 
		{
			Vec3d _normal;
			double dist = 0;
			if (vertices.size() < 3) {
				// need three vertices at least
				return std::make_tuple(_normal, dist, false);
			}

			Vec3d v1 = vertices[1] - vertices[0];
			Vec3d v2 = vertices[2] - vertices[0];
			_normal = cross(v1, v2);
			_normal = _normal / exanb::norm(_normal);
			dist = dot(_normal, vertices[0]);
			return std::make_tuple(_normal, dist, true);
		}

		/**
		 * @brief Builds bounding boxes for all faces in the mesh.
		 *
		 * The `build_boxes` function calculates and stores bounding boxes for all faces in the mesh. It ensures that the mesh
		 * has at least one face and performs this computation in parallel when possible.
		 */
		void build_boxes()
		{
			const int size = start.size();
			assert(size > 0);
			m_boxes.resize(size);
#pragma omp parallel for
			for(int i = 0 + start_mesh_box_construct ; i < size ; i++)
			{
				std::vector<Vec3d> vertices;
				for(int j=start[i]; j<end[i]; j++){
					Vec3d v= {vx[j], vy[j], vz[j]};
					vertices.push_back(v);
				}	
				m_boxes[i]= create_box(vertices);
				vertices.clear();
			}
		}
		
		void build_obbs()
		{
			const int size = start.size();
			assert(size > 0);
			m_obbs.resize(size);
#pragma omp parallel for
			for(int i = 0; i < size ; i++)
			{
				std::vector<Vec3d> vertices;
				for(int j = start[i]; j < end[i]; j++){
					Vec3d v = {vx[j], vy[j], vz[j]};
					vertices.push_back(v);
				}
				m_obbs[i] = build_OBB(vertices, 0);
				vertices.clear();
			}
			
		}	
		
		Box create_box(std::vector<Vec3d> vertices)
		{
			Vec3d inf = vertices[0];
			Vec3d sup = inf;
			for(auto vertex : vertices)
			{
				inf.x = std::min(inf.x,vertex.x);
				inf.y = std::min(inf.y,vertex.y);
				inf.z = std::min(inf.z,vertex.z);
				sup.x = std::max(sup.x,vertex.x);
				sup.y = std::max(sup.y,vertex.y);
				sup.z = std::max(sup.z,vertex.z);
			}
			Box res = {inf,sup};
			return res;
		}
		
		

		/**
		 * @brief Updates the indexes based on the bounding box of a face.
		 *
		 * The `update_indexes` function updates the indexes based on the bounding box of a face specified by `idBox` and `b1`.
		 * It iterates through the mesh's bounding boxes, compares them with `b1`, and adds relevant indexes to the specified `id`.
		 *
		 * @param id The index to update with relevant indexes.
		 * @param b1 The bounding box of the face for comparison.
		 */
		void update_indexes(const int id, Box& b1)
		{
			int b = false;
			for( size_t idBox = 0 ; idBox < m_boxes.size() ; idBox++ )
			{
				auto& b2 = m_boxes[idBox];
				if (( b1.sup.x >= b2.inf.x && b1.inf.x <= b2.sup.x) &&
						(b1.sup.y >= b2.inf.y && b1.inf.y <= b2.sup.y) &&
						(b1.sup.z >= b2.inf.z && b1.inf.z <= b2.sup.z))
				{
					indexes[id].push_back(idBox);
					b = true;
				}
			}
			
		}
		
		void update_indexes2(const int id, Box& b1)
		{
			Vec3d sup = b1.sup;
			Vec3d inf = b1.inf;
			Vec3d ct = b1.center();
			Vec3d vc = sup - ct;
			
			double supx = sup.x;
			double supy = sup.y;
			double supz = sup.z;
			double infx = inf.x;
			double infy = inf.y;
			double infz = inf.z;
			std::vector<Vec3d> vertices;
			vertices.push_back({supx, supy, supz});
			vertices.push_back({supx, infy, supz});
			vertices.push_back({supx, supy, infz});
			vertices.push_back({supx, infy, infz});
			vertices.push_back({infx, supy, supz});
			vertices.push_back({infx, supy, infz});
			vertices.push_back({infx, infy, supz});
			vertices.push_back({infx, infy, infz});
			
			OBB obb = build_OBB(vertices, 0);
			
			//if(obb.center.x != ct.x || obb.center.y != ct.y || obb.center.z != ct.z)
			//printf("AABB(%f, %f, %f)\n", ct.x, ct.y, ct.z);
			//printf("OBB(%f, %f, %f)\n", obb.center.x, obb.center.y, obb.center.z);
			//Vec3d size = b1.sup - b1.inf;
			//printf("VRAI CENTRE(%f, %f, %f) OBB(%f, %f, %f)\n", vc.x, vc.y, vc.z, obb.center.x, obb.center.y, obb.center.z); 
			//Vec3d half = 0.5 * size;
			//obb.extent = {half.x, half.y, half.z};
			//obb.extent = {size/2, size/2, size/2};
			//printf("CELL: %d SIZE: %f\n", id, size);
			for(size_t idBox = 0; idBox < m_obbs.size(); idBox++)
			{
				auto obb2 = m_obbs[idBox];
				if(obb2.intersect(obb)) indexes2[id].push_back(idBox);
			}
			
			
		}

		inline vec3r conv_to_vec3r (const Vec3d& v)
		{
			return vec3r {v.x, v.y, v.z};
		}
		
		inline std::vector<vec3r> conv_to_vec3r (std::vector<Vec3d> vector)
		{
			std::vector<vec3r> res;
			for(auto v: vector){
				res.push_back(conv_to_vec3r(v));
			}
			return res;
		}		
		
		inline OBB build_OBB ( std::vector<Vec3d>& vertices, double radius)
		{
		std::vector<vec3r> vec = conv_to_vec3r(vertices);
		
		//double radius = 0.1;	
		
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
		//printf("EXTENT: (%f, %f, %f)\n", obb.extent.x, obb.extent.y, obb.extent.y);

		obb.enlarge(radius);  // Add the Minskowski radius
		return obb;
		}
		
	};
}
