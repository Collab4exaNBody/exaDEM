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

namespace exaDEM
{
	struct Interaction_neighbors
	{
		onika::memory::CudaMMVector<double> ft_x;
		onika::memory::CudaMMVector<double> ft_y;
		onika::memory::CudaMMVector<double> ft_z;
		
		onika::memory::CudaMMVector<double> mom_x;
		onika::memory::CudaMMVector<double> mom_y;
		onika::memory::CudaMMVector<double> mom_z;
		
		onika::memory::CudaMMVector<int> id_i;
		onika::memory::CudaMMVector<int> id_j;
		
		onika::memory::CudaMMVector<int> cell_i;
		onika::memory::CudaMMVector<int> cell_j;
		
		onika::memory::CudaMMVector<int> p_i;
		onika::memory::CudaMMVector<int> p_j;
		
		onika::memory::CudaMMVector<uint16_t> sub_i;
		onika::memory::CudaMMVector<uint16_t> sub_j;
		
		onika::memory::CudaMMVector<int> particles_start;
		onika::memory::CudaMMVector<int> particles_end;
		
		
		bool iterator = false;
		
		 
	};
}
