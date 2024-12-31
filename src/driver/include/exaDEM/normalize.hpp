#pragma once

namespace exanb
{
	/**
	 * @brief Calculate the length of a 3D vector.
	 * @param v The input vector.
	 * @return The length of the vector.
	 */
	inline double length(Vec3d &v) { return std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z); }

	/**
	 * @brief Calculate the length of a const 3D vector.
	 * @param v The input vector.
	 * @return The length of the vector.
	 */
	inline double length(const Vec3d &v) { return std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z); }

	/**
	 * @brief Normalize a 3D vector.
	 * @param v The input vector to be normalized.
	 */
	inline void _normalize(Vec3d &v) { v = v / exanb::norm(v); }
}
