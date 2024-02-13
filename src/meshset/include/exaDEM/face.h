#pragma once

#include <exanb/core/basic_types.h>

namespace exaDEM
{
	using namespace exanb;
	/**
	 * @brief Struct representing a 3D box.
	 */
	struct Box
	{
		Vec3d inf; /**< The lower corner of the box. */
		Vec3d sup; /**< The upper corner of the box. */

		/**
		 * @brief Calculate the center of the box.
		 * @return The center of the box as a Vec3d.
		 */
		Vec3d center()
		{
			Vec3d res = (sup - inf)/2;
			return res;
		};
	};


	/**
	 * @brief Calculate the length of a 3D vector.
	 * @param v The input vector.
	 * @return The length of the vector.
	 */
	inline double length(Vec3d& v) {return std::sqrt(v.x*v.x+v.y*v.y+v.z*v.z);}

	/**
	 * @brief Calculate the length of a const 3D vector.
	 * @param v The input vector.
	 * @return The length of the vector.
	 */
	inline double length(const Vec3d& v) {return std::sqrt(v.x*v.x+v.y*v.y+v.z*v.z);}


	/**
	 * @brief Normalize a 3D vector.
	 * @param v The input vector to be normalized.
	 */
	inline void normalize(Vec3d& v) { v = v / exanb::norm(v); }


	/**
	 * @brief Struct representing a 3D face.
	 */
	struct Face 
	{
		std::vector<Vec3d> vertices; /**< The vertices of the face. */
		Vec3d normal;  /**< The normal vector of the face. */
		double offset; /**< The offset of the face. */

		/**
		 * @brief Constructor for the Face struct.
		 * @param in A vector of Vec3d representing the vertices of the face.
		 */
		Face(std::vector<Vec3d>& in) {
			vertices = in;
			auto [_normal, _offset, _exist] = compute_normal_and_offset();
			normal = _normal;
			offset = _offset;
			if(!_exist) std::cout << " error when filling this Face " << std::endl;
		}

		/**
		 * @brief Determines if a sphere and a face potentially intersect and calculates the contact position.
		 *
		 * This function checks whether a sphere with the given radius and center intersects with a face defined by its vertices,
		 * normal vector, and offset. It returns information about the potential intersection and the contact position.
		 *
		 * @param rx The x-coordinate of the sphere's center.
		 * @param ry The y-coordinate of the sphere's center.
		 * @param rz The z-coordinate of the sphere's center.
		 * @param rad The radius of the sphere.
		 * @return A tuple containing three values:
		 *         - `bool` face_contact: Indicates whether there is an intersection with the face.
		 *         - `bool` potential_contact: Indicates potential intersection with the face for further testing.
		 *         - `Vec3d` contact_position: The contact position if an intersection occurs (otherwise, it is {0,0,0}).
		 */
		std::tuple<bool, bool, Vec3d> contact_face_sphere(const double rx, const double ry, const double rz, const double rad) const 
		{
			//printf("CONTACT FACE SPHERE\n");
			const Vec3d center = {rx,ry,rz};
			const Vec3d default_contact_point = {0,0,0}; // won't be used
			bool potential_contact = false;
			bool face_contact = false;
			Vec3d contact_position = default_contact_point;

			double p = exanb::dot(center,normal) - offset;
			if( std::abs(p) > rad)
			{
				return std::make_tuple(face_contact, potential_contact, contact_position);
			}

			potential_contact = true; // This face will be tested versus edges (second pass)

			const int nb_vertices = vertices.size();
			const Vec3d& pa = vertices[0];
			const Vec3d& pb = vertices[1];
			const Vec3d& pc = vertices[nb_vertices-1];
			Vec3d v1 = pb - pa;
			Vec3d v2 = pc - pa;
			normalize(v1);
			Vec3d n = exanb::cross(v1,v2);
			normalize(n);
			Vec3d iv = center;// - pa;
			double dist = exanb::dot(iv,n);
			if(dist < 0.0)
			{
				dist = -dist;
				n = -n;
			}

			// test if the sphere intersects the surface 
			int intersections = 0;

			// from rockable
			Vec3d P = iv - dist * n;
			v2 = exanb::cross(n, v1);
			double ori1 = exanb::dot(P,v1);
			double ori2 = exanb::dot(P,v2);

			for (int iva = 0; iva < nb_vertices ; ++iva) {
				int ivb = iva + 1;
				if (ivb == nb_vertices) ivb = 0;
				const Vec3d& posNodeA_jv = vertices[iva];
				const Vec3d& posNodeB_jv = vertices[ivb];
				double pa1 = exanb::dot(posNodeA_jv , v1);
				double pb1 = exanb::dot(posNodeB_jv , v1);
				double pa2 = exanb::dot(posNodeA_jv , v2);
				double pb2 = exanb::dot(posNodeB_jv , v2);

				// @see http://local.wasp.uwa.edu.au/~pbourke/geometry/insidepoly/
				// @see http://alienryderflex.com/polygon/
				if ((pa2 < ori2 && pb2 >= ori2) || (pb2 < ori2 && pa2 >= ori2)) {
					if (pa1 + (ori2 - pa2) / (pb2 - pa2) * (pb1 - pa1) < ori1) {
						intersections = 1 - intersections;
					}
				}
			}

			if(intersections == 1) // ODD 
			{
				contact_position = normal*offset; // we need dot(conatct_position, normal)
				face_contact = true;
			}

			return  std::make_tuple(face_contact, potential_contact, contact_position);
		}


		/**
		 * @brief Determines if a sphere intersects with an edge and calculates the contact position.
		 *
		 * This function checks whether a sphere with the given radius and center intersects with any edge of a polygon defined
		 * by its vertices. It returns information about the intersection and the contact position if applicable.
		 *
		 * @param rx The x-coordinate of the sphere's center.
		 * @param ry The y-coordinate of the sphere's center.
		 * @param rz The z-coordinate of the sphere's center.
		 * @param rad The radius of the sphere.
		 * @return A tuple containing two values:
		 *         - `bool` intersects: Indicates whether there is an intersection with an edge.
		 *         - `Vec3d` contact_position: The contact position if an intersection occurs (otherwise, it is {0,0,0}).
		 */	
		std::tuple<bool, Vec3d> contact_edge_sphere(const double rx, const double ry, const double rz, const double rad) const 
		{
			//printf("CONTACT EDGE SPHERE\n");
			// already tested if  exanb::dot(center,normal) - offset < rad
			// test if the sphere intersects an edge 
			const Vec3d center = {rx,ry,rz};
			const Vec3d default_contact_point = {0,0,0}; // won't be used
			for (size_t i = 0; i < vertices.size(); ++i) {
				Vec3d p1 = vertices[i];
				Vec3d p2 = vertices[(i + 1) % vertices.size()];
				Vec3d edge = p2 - p1;
				Vec3d sphereToEdge = center - p1;

				double distanceToEdge = length(exanb::cross(edge, sphereToEdge)) / length(edge);

				if (distanceToEdge <= rad && exanb::dot(sphereToEdge, edge) > 0 && exanb::dot(sphereToEdge - edge, edge) < 0) {
					auto n_edge = edge / exanb::norm(edge);
					Vec3d contact_position = p1 + n_edge * dot(sphereToEdge, n_edge);
					return std::make_tuple(true, contact_position); 
				}
			}
			return std::make_tuple(false, default_contact_point);
		}

		std::tuple<bool, Vec3d, int> intersect_sphere(const double rx, const double ry, const double rz, const double rad) const 
		{
			Vec3d center = {rx,ry,rz};
			const Vec3d default_norm = {0,0,0};

			double p = exanb::dot(center,normal) - offset;
			if( std::abs(p) >= rad) 
			{
				return std::make_tuple(false, default_norm,-1);
			}

			const int nb_vertices = vertices.size();
			Vec3d v1 = vertices[1] - vertices[0];
			Vec3d v2 = vertices[nb_vertices-1] - vertices[0];
			v1 = v1 / exanb::norm(v1);
			Vec3d n = exanb::cross(v1,v2);
			n = n / exanb::norm(n);
			Vec3d iv = center - vertices[0];
			double dist = exanb::dot(iv,n);
			if(dist < 0.0)
			{
				dist = -dist;
				n = -n;
			}

			// test if the sphere intersects the surface 
			int intersections = 0;

			// from rockable
			Vec3d	P = iv - dist * n;
			v2 = exanb::cross(n, v1);
			double ori1 = exanb::dot(P, v1);
			double ori2 = exanb::dot(P,v2);

			for (int iva = 0; iva < nb_vertices ; ++iva) {
				int ivb = iva + 1;
				if (ivb == nb_vertices) ivb = 0;
				const Vec3d& posNodeA_jv = vertices[iva];
				const Vec3d& posNodeB_jv = vertices[ivb];
				double pa1 = exanb::dot(posNodeA_jv , v1);
				double pb1 = exanb::dot(posNodeB_jv , v1);
				double pa2 = exanb::dot(posNodeA_jv , v2);
				double pb2 = exanb::dot(posNodeB_jv , v2);

				// @see http://local.wasp.uwa.edu.au/~pbourke/geometry/insidepoly/
				// @see http://alienryderflex.com/polygon/
				if ((pa2 < ori2 && pb2 >= ori2) || (pb2 < ori2 && pa2 >= ori2)) {
					if (pa1 + (ori2 - pa2) / (pb2 - pa2) * (pb1 - pa1) < ori1) {
						intersections = 1 - intersections;
					}
				}
			}

			if(intersections == 1) // ODD 
			{
				Vec3d contact_position = normal*offset; // we need dot(conatct_position, normal)
				return std::make_tuple(true, contact_position, 0);
			}

			// test if the sphere intersects an edge 
			for (size_t i = 0; i < vertices.size(); ++i) 
			{
				Vec3d p1 = vertices[i];
				Vec3d p2 = vertices[(i + 1) % vertices.size()];
				Vec3d edge = p2 - p1;
				Vec3d sphereToEdge = center - p1;

				// Calculer la distance entre le centre de la sphère et le bord le plus proche
				double distanceToEdge = length(exanb::cross(edge, sphereToEdge)) / length(edge);

				if (distanceToEdge <= rad && exanb::dot(sphereToEdge, edge) > 0 && exanb::dot(sphereToEdge - edge, edge) < 0) {
					auto n_edge = edge / exanb::norm(edge);
					Vec3d contact_position = p1 + n_edge * dot(sphereToEdge, n_edge);
					return std::make_tuple(true, contact_position, 1); // La sphère touche un bord
				}
			}
			return std::make_tuple(false, default_norm, -1);
		}

		/**
		 * @brief Computes the normal vector and offset for a face defined by its vertices.
		 *
		 * This function calculates the normal vector and offset for a face based on its vertices. It returns the computed
		 * normal vector, offset, and a boolean indicating success or failure.
		 *
		 * @return A tuple containing three values:
		 *         - `Vec3d` normal: The computed normal vector.
		 *         - `double` offset: The computed offset.
		 *         - `bool` success: Indicates whether the calculation was successful (true) or not (false).
		 */
		std::tuple<Vec3d, double, bool> compute_normal_and_offset() 
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
		 * @brief Creates a bounding box that contains the vertices of a polygon.
		 *
		 * This function computes a bounding box that encloses the vertices of a polygon. It returns the created box.
		 *
		 * @return A Box struct representing the bounding box of the polygon.
		 */
		Box create_box()
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
	};
};
