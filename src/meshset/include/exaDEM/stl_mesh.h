#pragma once

#include <exaDEM/face.h>
//#include <exaDEM/face2.h>
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
	struct stl_mesh
	{
		//std::vector <Face> m_data; /**< A collection of Face objects representing the mesh. */
		onika::memory::CudaMMVector< Face > m_data;/**< A collection of Face objects representing the mesh. */
		onika::memory::CudaMMVector< Box > m_boxes;/**< A collection of boxes.*/
		onika::memory::CudaMMVector <onika::memory::CudaMMVector <int> > indexes;/**< Indexes for mesh data. */
		onika::memory::CudaMMVector <onika::memory::CudaMMVector <int> > indexes2;/**< Indexes for mesh data. */
		std::vector<OBB> m_obbs;
		
		/**
		 * @brief Adds a Face to the mesh.
		 * @param face The Face to be added to the mesh.
		 */
		void add_face(Face& face)
		{
			m_data.push_back(face);
		}

		/**
		 * @brief Gets a reference to the mesh data.
		 * @return A reference to the vector of Face objects representing the mesh.
		 */
		onika::memory::CudaMMVector <Face>& get_data()
		{
			return m_data;
		}

		/**
		 * @brief Gets a reference to a specific Face in the mesh.
		 * @param idx The index of the Face to retrieve.
		 * @return A reference to the specified Face.
		 */
		Face& get_data(const int idx)
		{
			return m_data[idx];
		}
		
		
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
			std::ifstream input( file_name.c_str() );
			std::string first;
			std::vector<Vec3d> vertices;
			Vec3d vertex;
			int nv = 0;
			int nf = 0;
			
			for( std::string line; getline( input, line ); )
			{
				input >> first;
				if(first == "outer")
				{
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
					Face tmp (vertices);
					this->add_face(tmp);
					vertices.clear();
					nf++;
				}
			}
			std::cout << "Mesh: " << file_name << " - number of vertices: " << nv << " - number of faces: " << nf << std::endl;
		}

		/**
		 * @brief Builds bounding boxes for all faces in the mesh.
		 *
		 * The `build_boxes` function calculates and stores bounding boxes for all faces in the mesh. It ensures that the mesh
		 * has at least one face and performs this computation in parallel when possible.
		 */
		void build_boxes()
		{
			const int size = m_data.size();
			//const int size = m_data_size;
			assert(size > 0);
			m_boxes.resize(size);
#pragma omp parallel for
			for(int i = 0 ; i < size ; i++)
			{
				m_boxes[i] = m_data[i].create_box();
				/**if(i == 0){
				printf("BOX\n");
				printf("FACE: %d, OBBSUP: (%f, %f, %f), OBBINF (%f, %f, %f), CENTER (%f, %f, %f)\n", i, m_boxes[i].sup.x, m_boxes[i].sup.y, m_boxes[i].sup.z, m_boxes[i].inf.x, m_boxes[i].inf.y, m_boxes[i].inf.z, m_boxes[i].center().x, m_boxes[i].center().y, m_boxes[i].center().z);
				printf("VERTICES\n");
				auto v = m_data[i].vertices;
				for(int j= 0; j < v.size(); j++){
					printf("V%d(%f, %f, %f) ", j, v[i].x, v[i].y, v[i].z); 
				}
				printf("\n");
				}*/
			}
		}
		
		void build_obbs()
		{
			const int size = m_data.size();
			assert(size > 0);
			m_obbs.resize(size);
#pragma omp parallel for
			for(int i = 0; i < size ; i++)
			{
				
				//printf("VREAL ! %f\n", size);
				m_obbs[i] = build_OBB(m_data[i].vertices, 0);
				//if( i == 0){
				//printf("FACE: %d, E[0] (%f, %f, %f), E[1] (%f, %f, %f), E[2] (%f, %f, %f), EXTENT (%f, %f, %f), CENTER (%f, %f, %f)\n", i, m_obbs[i].e[0].x, m_obbs[i].e[0].y, m_obbs[i].e[0].z, m_obbs[i].e[1].x, m_obbs[i].e[1].y, m_obbs[i].e[1].z, m_obbs[i].e[2].x, m_obbs[i].e[2].y, m_obbs[i].e[2].z,  m_obbs[i].extent.x, m_obbs[i].extent.y, m_obbs[i].extent.z, m_obbs[i].center.x, m_obbs[i].center.y, m_obbs[i].center.z);
				//}
				//0.09357288
				//0.09357288
			}
			
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
			for( size_t idBox = 0 ; idBox < m_boxes.size() ; idBox++ )
			{
				auto& b2 = m_boxes[idBox];
				if (( b1.sup.x >= b2.inf.x && b1.inf.x <= b2.sup.x) &&
						(b1.sup.y >= b2.inf.y && b1.inf.y <= b2.sup.y) &&
						(b1.sup.z >= b2.inf.z && b1.inf.z <= b2.sup.z))
				{
					indexes[id].push_back(idBox);
					
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
			onika::memory::CudaMMVector<Vec3d> vertices;
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
		
		inline onika::memory::CudaMMVector<vec3r> conv_to_vec3r (onika::memory::CudaMMVector<Vec3d> vector)
		{
			onika::memory::CudaMMVector<vec3r> res;
			for(auto v: vector){
				res.push_back(conv_to_vec3r(v));
			}
			return res;
		}		
		
		inline OBB build_OBB ( onika::memory::CudaMMVector<Vec3d>& vertices, double radius)
		{
		onika::memory::CudaMMVector<vec3r> vec = conv_to_vec3r(vertices);
		
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
