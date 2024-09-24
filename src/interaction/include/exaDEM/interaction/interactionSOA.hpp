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

#include <iostream>
//#include <ostream>
#include <exanb/core/basic_types.h>
#include <exanb/core/basic_types_stream.h>
#include <exaDEM/interaction/interaction.hpp>

namespace exaDEM
{
	/**
	 * @brief Structure representing an interaction in a Discrete Element Method (DEM) simulation.
	 */
	struct InteractionSOA
	{
		template <typename T> using VectorT =  onika::memory::CudaMMVector<T>;
		
		VectorT<double> ft_x;
		VectorT<double> ft_y;
		VectorT<double> ft_z;
		
		VectorT<double> mom_x;
		VectorT<double> mom_y;
		VectorT<double> mom_z;
		
		VectorT<size_t> id_i;
		VectorT<size_t> id_j;
		
		VectorT<size_t> cell_i;
		VectorT<size_t> cell_j;
		
		VectorT<int> p_i;
		VectorT<int> p_j;
		
		VectorT<size_t> sub_i;
		VectorT<size_t> sub_j;
		
		uint16_t type;          /**< Type of the interaction (e.g., contact type). */

		/**
		 * @brief Resets the Interaction structure by setting friction and moment vectors to zero.
		 */
		ONIKA_HOST_DEVICE_FUNC void reset( size_t id )
		{
			onika::cuda::vector_data(ft_x)[id] = 0;
			onika::cuda::vector_data(ft_y)[id] = 0;
			onika::cuda::vector_data(ft_z)[id] = 0;
			
			onika::cuda::vector_data(mom_x)[id] = 0;
			onika::cuda::vector_data(mom_y)[id] = 0;
			onika::cuda::vector_data(mom_z)[id] = 0;
		}
		
		void clear()
		{
			ft_x.clear();
			ft_y.clear();
			ft_z.clear();
			
			mom_x.clear();
			mom_y.clear();
			mom_z.clear();
			
			id_i.clear();
			id_j.clear();
			
			cell_i.clear();
			cell_j.clear();
			
			p_i.clear();
			p_j.clear();
			
			sub_i.clear();
			sub_j.clear();
		}
		
		const ONIKA_HOST_DEVICE_FUNC size_t size() const
		{
			return onika::cuda::vector_size(ft_x);
		}
		
		ONIKA_HOST_DEVICE_FUNC size_t size() 
		{
			return onika::cuda::vector_size(ft_x);
		}
		
		void insert( std::vector<exaDEM::Interaction> tmp )
		{
			for(auto interaction : tmp)
			{
				ft_x.push_back(interaction.friction.x);
				ft_y.push_back(interaction.friction.y);
				ft_z.push_back(interaction.friction.z);
			
				mom_x.push_back(interaction.moment.x);
				mom_y.push_back(interaction.moment.y);
				mom_z.push_back(interaction.moment.z);
			
				id_i.push_back(interaction.id_i);
				id_j.push_back(interaction.id_j);
			
				cell_i.push_back(interaction.cell_i);
				cell_j.push_back(interaction.cell_j);
			
				p_i.push_back(interaction.p_i);
				p_j.push_back(interaction.p_j);
			
				sub_i.push_back(interaction.sub_i);
				sub_j.push_back(interaction.sub_j);
			}
		}
		
		/*const void insert( std::vector<exaDEM::Interaction> tmp ) const
		{
			for(auto interaction : tmp)
			{
				ft_x.push_back(interaction.friction.x);
				ft_y.push_back(interaction.friction.y);
				ft_z.push_back(interaction.friction.z);
			
				mom_x.push_back(interaction.moment.x);
				mom_y.push_back(interaction.moment.y);
				mom_z.push_back(interaction.moment.z);
			
				id_i.push_back(interaction.id_i);
				id_j.push_back(interaction.id_j);
			
				cell_i.push_back(interaction.cell_i);
				cell_j.push_back(interaction.cell_j);
			
				p_i.push_back(interaction.p_i);
				p_j.push_back(interaction.p_j);
			
				sub_i.push_back(interaction.sub_i);
				sub_j.push_back(interaction.sub_j);
			}
		}*/
		
		ONIKA_HOST_DEVICE_FUNC exaDEM::Interaction operator[](size_t id)
		{
			return { {onika::cuda::vector_data(ft_x)[id], onika::cuda::vector_data(ft_y)[id], onika::cuda::vector_data(ft_z)[id]}, 
				{onika::cuda::vector_data(mom_x)[id], onika::cuda::vector_data(mom_y)[id], onika::cuda::vector_data(mom_z)[id]},
				 onika::cuda::vector_data(id_i)[id], onika::cuda::vector_data(id_j)[id],
				 onika::cuda::vector_data(cell_i)[id], onika::cuda::vector_data(cell_j)[id], 
				 onika::cuda::vector_data(p_i)[id], onika::cuda::vector_data(p_j)[id], 
				 onika::cuda::vector_data(sub_i)[id], onika::cuda::vector_data(sub_j)[id], 
				 type};
		}
		
		const exaDEM::Interaction operator[](size_t id) const
		{
			return { {ft_x[id], ft_y[id], ft_z[id]}, {mom_x[id], mom_y[id], mom_z[id]}, id_i[id], id_j[id], cell_i[id], cell_j[id], p_i[id], p_j[id], sub_i[id], sub_j[id], type};
		}
		
		exaDEM::Interaction get_interaction( size_t id )
		{
			return { {ft_x[id], ft_y[id], ft_z[id]}, {mom_x[id], mom_y[id], mom_z[id]}, id_i[id], id_j[id], cell_i[id], cell_j[id], p_i[id], p_j[id], sub_i[id], sub_j[id], type};
		}
		
		ONIKA_HOST_DEVICE_FUNC void update(size_t id, exaDEM::Interaction item)
		{
			onika::cuda::vector_data(ft_x)[id] = item.friction.x;
			onika::cuda::vector_data(ft_y)[id] = item.friction.y;
			onika::cuda::vector_data(ft_z)[id] = item.friction.z;
			
			onika::cuda::vector_data(mom_x)[id] = item.moment.x;
			onika::cuda::vector_data(mom_y)[id] = item.moment.y;
			onika::cuda::vector_data(mom_z)[id] = item.moment.z;
		}
		
		void print()
		{
			for(int i = 0; i < 1000; i++)
			{
				printf("INTERACTION: %d ID_I: %d ID_J: %d\n", i, id_i[i], id_j[i]);
			}
		}


	};
}
