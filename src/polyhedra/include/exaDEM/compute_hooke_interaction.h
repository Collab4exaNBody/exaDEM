#pragma once

#include <exaDEM/interaction.hpp>

namespace exaDEM
{
	template<class GridT>
		struct HookeForceInteractionFunctor
		{
			GridT& cells;
			mutexes& locker;
      const shapes & shps;
      const HookeParams params;
      const double time;

			inline const Vec3d get_r(const int cell_id, const int p_id)
			{
				const Vec3d res = {
					cells[cell_id][field::rx][p_id],
					cells[cell_id][field::ry][p_id],
					cells[cell_id][field::rz][p_id]};
				return res;
			};

			inline const Vec3d get_v(const int cell_id, const int p_id)
			{
				const Vec3d res = {
					cells[cell_id][field::vx][p_id],
					cells[cell_id][field::vy][p_id],
					cells[cell_id][field::vz][p_id]};
				return res;
			};

			ONIKA_HOST_DEVICE_FUNC inline void compute_force_interaction (  )
				{
				}

			ONIKA_HOST_DEVICE_FUNC inline void operator () ( Interaction * __restrict__ ptr, const size_t size) const
			{
			}

			ONIKA_HOST_DEVICE_FUNC inline void operator () ( Interaction& it) const
			{
			}
		};

}
