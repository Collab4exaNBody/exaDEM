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
			ONIKA_HOST_DEVICE_FUNC inline bool operator()(Args&&... args)
			{
				return driver.filter(std::forward<Args>(args)...);
			}
	};

	// API 
	template <typename Driver> 
		ONIKA_HOST_DEVICE_FUNC inline bool filter_vertex_driver (
				Driver& driver, const double rcut,
				const Vec3d& pi, const int i, const shape* shpi, const exanb::Quaternion& oi)
		{
			filter_driver<Driver> filter = {driver};
			const Vec3d vi = shpi->get_vertex(i, pi, oi);
			return filter ( rcut + shpi->m_radius, vi );
		}

	template <typename Driver> 
		ONIKA_HOST_DEVICE_FUNC inline bool filter_vertex_driver(
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
			ONIKA_HOST_DEVICE_FUNC inline std::tuple<bool, double, Vec3d, Vec3d> operator()(Args&&... args)
			{
				return driver.detector(std::forward<Args>(args)...);
			}
	};

	// API 
	template <typename Driver> 
		ONIKA_HOST_DEVICE_FUNC inline std::tuple<bool, double, Vec3d, Vec3d> detector_vertex_driver (
				Driver& driver, const Vec3d& pi, const int i, const shape* shpi, const exanb::Quaternion& oi)
		{
			detector_driver<Driver> detector = {driver};
			const Vec3d vi = shpi->get_vertex(i, pi, oi);
			return detector( shpi->m_radius, vi );
		}

	template <typename Driver> 
		ONIKA_HOST_DEVICE_FUNC inline std::tuple<bool, double, Vec3d, Vec3d> detector_vertex_driver(
				Driver& driver, const VertexArray& vertexes, const int i, const shape* shpi)
		{
			detector_driver<Driver> detector = {driver};
			return detector( shpi->m_radius, vertexes[i]);
		}

	template <typename Driver> 
		ONIKA_HOST_DEVICE_FUNC inline std::tuple<bool, double, Vec3d, Vec3d> detector_vertex_driver(
				Driver& driver, const Vec3d& position, const double radius)
		{
			detector_driver<Driver> detector = {driver};
			return detector(radius, position);
		}
}
