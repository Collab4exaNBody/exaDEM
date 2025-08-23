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

#include <onika/scg/operator.h>
#include <onika/scg/operator_slot.h>
#include <onika/scg/operator_factory.h>
#include <exanb/core/make_grid_variant_operator.h>
#include <exanb/core/parallel_grid_algorithm.h>
#include <exanb/core/grid.h>

namespace exaDEM
{
  using namespace exanb;

  template <typename GridT> class CheckDTCritical : public OperatorNode
  {
    ADD_SLOT(GridT, grid, INPUT_OUTPUT, REQUIRED);
    ADD_SLOT(double, mass, INPUT, OPTIONAL, DocString{"Mass of the particle"});
    ADD_SLOT(double, kn, INPUT, OPTIONAL, DocString{"Contact force parameter (normal)"});
    ADD_SLOT(double, treshold, INPUT, 1.0, DocString{"Ratio treshold."});
    ADD_SLOT(double, dt, INPUT, REQUIRED, DocString{"Time increment."});

    // -----------------------------------------------
    // ----------- Operator documentation ------------
    inline std::string documentation() const override final
    {
      return R"EOF(
        This operator checks the dt critical according to a ratio treshold (dt/dt_critical). 

        YAML example:

          - check_dt_critical:
             treshold: 0.02
        )EOF";
    }

    public:

    bool check_slots()
    {
      if(!(mass.has_value()))
      {
        lout << "\033[38;5;208m[check_dt_critical, WARNING] The input slot 'mass' is not defined. Please use the 'min_mass' operator to retrieve the smallest particle mass (use rebind to rename the output slot).\033[0m" << std::endl;
        lout << "\033[38;5;208mdt critical is not computed.\033[0m" << std::endl;
        return false;
      }
			else
			{
				if(*mass <= 0.0)
				{
					lout << "\033[31m[check_dt_critical, ERROR] Mass is not defined correctly, mass = " << *mass << "\033[0m" << std::endl;
					std::exit(EXIT_FAILURE);
				}
			}

			if(!(kn.has_value()))
			{
				lout << "\033[38;5;208m[check_dt_critical, WARNING] 'kn' is not defined. Please use the 'check_dt_critical' operator after the contact force has been applied.\033[0m" << std::endl;
				lout << "\033[38;5;208mdt critical is not computed.\033[0m" << std::endl;
				return false;
			}
			else
			{
				if(*kn <= 0.0)
				{
					lout << "\033[31m[check_dt_critical, ERROR] kn is not defined correctly, kn = " << *kn << "\033[0m" << std::endl;
					std::exit(EXIT_FAILURE);
				}
			}
			return true;
		}

		inline void execute() override final
		{
			if(!check_slots()) return;

			double _mass = *mass; 
			double _kn = *kn;
			double dt_critical = 4*std::atan(1) * std::sqrt( _mass / _kn );
			double ratio_treshold = *treshold;
			double ratio = *dt / dt_critical;
			if(ratio < ratio_treshold)
			{
				lout << "\033[32m[check_dt_critical] The time step is correctly defined, dt_critical = " << dt_critical << "\033[0m" << std::endl;
			}
			else
			{
				lout << "\033[38;5;208m[check_dt_critical, WARNING] The time step is probably too high. Please consider reducing dt to: " << dt_critical << "\033[0m" << std::endl;
			}
		}
	};

	template <class GridT> using CheckDTCriticalTmpl = CheckDTCritical<GridT>;

	// === register factories ===
	ONIKA_AUTORUN_INIT(check_dt_critical) { OperatorNodeFactory::instance()->register_factory("check_dt_critical", make_grid_variant_operator<CheckDTCritical>); }

} // namespace exaDEM
