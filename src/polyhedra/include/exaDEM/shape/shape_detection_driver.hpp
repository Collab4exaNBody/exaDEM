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
#include <exanb/core/basic_types_operators.h>
#include <exaDEM/shape/shape.hpp>
#include <math.h>
#include <exaDEM/shape/shape_prepro.hpp>
#include <exaDEM/shape/shape_detection.hpp>
#include <exaDEM/drivers.h>

namespace exaDEM
{
	using namespace exanb;
	using VertexArray = ::onika::oarray_t<::exanb::Vec3d, EXADEM_MAX_VERTICES>;

	// -> First Filters
	template<typename Driver>
		struct filter_driver
	{
		Driver& driver;
		template<typename... Args>
			inline bool operator()(Args&&... args)
			{
				return driver.filter(std::forward<Args>(args)...);
			}
	};

	// API 
	template <typename Driver> 
		inline bool filter_vertex_driver (
				Driver& driver, const double rcut,
				const Vec3d& pi, const int i, const shape* shpi, const exanb::Quaternion& oi)
		{
			filter_driver<Driver> filter = {driver};
			const Vec3d vi = shpi->get_vertex(i, pi, oi);
			return filter ( rcut + shpi->m_radius, vi );
		}

	template <typename Driver> 
		inline bool filter_vertex_driver(
				Driver& driver, const double rcut,
				const VertexArray& vertexes, const int i, const shape* shpi)
		{
			filter_driver<Driver> filter = {driver};
			return filter( rcut +  shpi->m_radius, vertexes[i]);
		}

	// -> First Filters
	template<typename Driver>
		struct detector_driver
	{
		Driver& driver;
		template<typename... Args>
			inline std::tuple<bool, double, Vec3d, Vec3d> operator()(Args&&... args)
			{
				return driver.detector(std::forward<Args>(args)...);
			}
	};

	// API 
	template <typename Driver> 
		inline std::tuple<bool, double, Vec3d, Vec3d> detector_vertex_driver (
				Driver& driver, const Vec3d& pi, const int i, const shape* shpi, const exanb::Quaternion& oi)
		{
			detector_driver<Driver> detector = {driver};
			const Vec3d vi = shpi->get_vertex(i, pi, oi);
			return detector( shpi->m_radius, vi );
		}

	template <typename Driver> 
		inline std::tuple<bool, double, Vec3d, Vec3d> detector_vertex_driver(
				Driver& driver, const VertexArray& vertexes, const int i, const shape* shpi)
		{
			detector_driver<Driver> detector = {driver};
			return detector( shpi->m_radius, vertexes[i]);
		}

	// Second Detections
	////////////////////////// CYLINDER /////////////////////////////////////////
	/**
	 * @brief Detects the intersection between a vertex of a polyhedron and a cylinder.
	 *
	 * This function checks if a vertex, represented by its position 'pi' and orientation 'oi',
	 * intersects with a cylindrical shape defined by its center projection 'center_proj', axis 'axis',
	 * and radius 'radius'.
	 *
	 * @param pi The position of the vertex.
	 * @param i Index of the vertex.
	 * @param shpi Pointer to the shape of the vertex.
	 * @param oi Orientation of the polyhedron.
	 * @param center_proj The center projection of the cylinder center on the axis.
	 * @param axis The axis of the cylindrical shape.
	 * @param radius The radius of the cylinder.
	 *
	 * @return A tuple containing:
	 *   - A boolean indicating whether there is an intersection (true) or not (false).
	 *   - The penetration depth in case of intersection.
	 *   - The contact normal at the intersection point.
	 *   - The contact point between the vertex and the cylinder.
	 */
/*
	inline std::tuple<bool, double, Vec3d, Vec3d> detection_vertex_cylinder(
			const Vec3d& pi, const int i, const shape* shpi, const exanb::Quaternion& oi, 
			const Vec3d& center_proj, const Vec3d& axis, const double radius)
	{
		// === get vertex radius and vertex position	
		const double ri = shpi->m_radius;
		const Vec3d vi = shpi->get_vertex(i, pi, oi);

		// === project the vertex in the plan as the cylinder center
		const Vec3d proj = vi * axis;

		// === direction
		const Vec3d dir = center_proj - proj;

		// === interpenetration
		const double d = exanb::norm(dir);

		// === compute interpenetration
		const double dn = radius - (ri + d);

		if ( dn > 0 )
		{
			return {false, 0.0, Vec3d(), Vec3d()};
		}
		else
		{
			// === compute contact normal 
			const Vec3d n = dir / d;

			// === compute contact point
			const Vec3d contact_position = vi - n * (ri + 0.5 * dn);

			return {true, dn, n, contact_position};
		}
	}
*/
}
