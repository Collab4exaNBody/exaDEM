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

#include <exaDEM/contact_force_parameters.h>
#include <exaDEM/contact_force_accessor.h>
#include <exanb/core/particle_type_id.h>

namespace exaDEM
{
	template <typename T> using vector_t = onika::memory::CudaMMVector<T>;
  using ReverseParticleTypeMap = std::map< uint64_t, std::string >;

	template<typename ContactParamsT>
		struct ContactParamsMultiMat
		{
			vector_t<ContactParamsT> multimat_cp;
			vector_t<ContactParamsT> drivers_cp;
      ParticleTypeMap type_map;
      ParticleTypeMap driver_map;
      ReverseParticleTypeMap reverse_type_map;
      ReverseParticleTypeMap reverse_driver_map;
      int number_of_materials;
      int number_of_drivers;


			int size_types()
			{
				assert(type_mpa.size() == reverse_type_map.size());
				assert(type_map.size() == multimat_cp.size());
				assert(type_map.size() == number_of_materials);
				return number_of_materials;
			}

			int size_drivers()
			{
				assert(driver_map.size() == reverse_driver_map.size());
				assert(driver_map.size() == drivers_cp.size());
				assert(driver_map.size() == number_of_drivers);
				return number_of_drivers;
			}

			MultiMatContactParamsTAccessor<ContactParamsT> get_multimat_accessor() const
			{
				MultiMatContactParamsTAccessor<ContactParamsT> res = { multimat_cp.data() , number_of_materials };
				return res;
			}

			MultiMatContactParamsTAccessor<ContactParamsT> get_drivers_accessor() const
			{
				MultiMatContactParamsTAccessor<ContactParamsT> res = { drivers_cp.data() , number_of_drivers };
				return res;
			}

			void setup_multimat(const ParticleTypeMap& input_map, ContactParamsT cp = {}) 
			{
				/** Define type map and reverse type map */
				type_map = input_map;
				for(auto it : type_map)
				{
					reverse_type_map[it.second] = it.first;
				}
				// Allocate array of Contact Force Parameters
				number_of_materials = input_map.size();
				multimat_cp.resize(number_of_materials * number_of_materials, cp);
			}

			void setup_drivers(const ParticleTypeMap& input_map, ContactParamsT cp = {}) 
			{
				driver_map = input_map;
				for(auto it : driver_map)
				{
					reverse_driver_map[it.second] = it.first;
				}
				number_of_drivers = input_map.size();
        assert(number_of_materials > 0);
				drivers_cp.resize(number_of_drivers * number_of_materials, cp);
			}

			ONIKA_HOST_DEVICE_FUNC inline int get_idx_multimat(int mat1, int mat2) 
			{ 
				assert(mat1 < size_types());
				assert(mat2 < size_types());
				return mat1*number_of_materials + mat2; 
			}

			ONIKA_HOST_DEVICE_FUNC inline int get_idx_drivers(int mat, int driver)  
			{ 
				assert(mat < number_of_materials);
				assert(driver < number_of_drivers);
				return mat*number_of_drivers + driver; 
			}

			void register_multimat(int mat1, int mat2, ContactParamsT& cp)
			{
				int id = get_idx_multimat(mat1, mat2);
				multimat_cp[id] = cp;
				id = get_idx_multimat(mat2, mat1);
				multimat_cp[id] = cp;
			}

			void register_driver(int mat, int driver, ContactParamsT& cp)
			{
				int id = get_idx_drivers(mat, driver);
				drivers_cp[id] = cp;
			}

			bool check_completeness(bool check_multimat = true, bool check_driver = true)
			{
				bool check = true;
				ContactParamsT default_cp = {};
				for(int m1 = 0 ; m1 < number_of_materials ; m1++)
				{
					if(check_multimat)
					{
						for(int m2 = 0 ; m2 <= m1 ; m2++) /** symtric definition */
						{
							if(multimat_cp[this->get_idx_multimat(m1, m2)] == default_cp)
							{
								lout << "\033[1;31mWarning: contact force parameters for the pair (mat: " 
									<< m1 << ", mat: " << m2 << ") are not defined.\033[0m" << std::endl;
								check = false;
							}
						}
					}
					if(check_driver)
					{
						for(int drv = 0 ; drv < number_of_drivers ; drv++)
						{
							if(drivers_cp[this->get_idx_drivers(m1, drv)] == default_cp)
							{
								lout << "\033[1;31mWarning: contact force parameters for the pair (mat: " 
									<< m1 << ", driver: " << drv << ") are not defined.\033[0m" << std::endl;
								check = false;
							}
						}
					}
				}
				return check;
			}

			void display(bool multimat_mode = true, bool drivers_mode = true)
			{
				display_header<ContactParamsT>();
				for(int m1 = 0 ; m1 < number_of_materials ; m1++)
				{
					if(multimat_mode)
					{
						for(int m2 = 0 ; m2 <= m1 ; m2++) /** symtric definition */
						{
							std::string type1 = reverse_type_map[m1];
							std::string type2 = reverse_type_map[m2];
							ContactParamsT& cp = multimat_cp[this->get_idx_multimat(m1, m2)];
							display_multimat(type1, type2, cp);
						}
					}
					if(drivers_mode)
					{
						for(int drv = 0 ; drv < number_of_drivers ; drv++)
						{
							display_multimat(reverse_type_map[m1], reverse_driver_map[drv], drivers_cp[this->get_idx_drivers(m1, drv)]);
						}
					}
				}
				display_end_table<ContactParamsT>();
			}


			ONIKA_HOST_DEVICE_FUNC inline ContactParamsT& get_multimat_cp(int mat1, int mat2) const
			{
				return multimat_cp[this->get_idx_multimat(mat1, mat2)];
			}

			ONIKA_HOST_DEVICE_FUNC inline ContactParamsT& get_drivers_cp(int mat, int driver) const
			{
				return drivers_cp[this->get_idx_driver(mat, driver)];
			}
		};
} // namespace exaD
