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
#include <exanb/core/operator.h>
#include <exanb/core/operator_slot.h>
#include <exanb/core/operator_factory.h>
#include <exanb/core/make_grid_variant_operator.h>
#include <exanb/core/parallel_grid_algorithm.h>
#include <exanb/core/grid.h>

#include <memory>
#include <mpi.h>

#include <exaDEM/contact_force_parameters.h>
#include <exaDEM/compute_contact_force.h>
#include <exaDEM/interaction/interaction.hpp>
#include <exaDEM/interaction/grid_cell_interaction.hpp>
#include <exaDEM/interaction/classifier.hpp>
#include <exaDEM/shape/shapes.hpp>
#include <exaDEM/shape/shape_detection.hpp>
#include <exaDEM/shape/shape_detection_driver.hpp>
#include <exaDEM/mutexes.h>
#include <exaDEM/drivers.h>

namespace exaDEM
{
  using namespace exanb;

  template<typename GridT , class = AssertGridHasFields< GridT, field::_rx,  field::_ry,  field::_rz >>
    class StressTensor : public OperatorNode
  {
    // attributes processed during computation
    using ComputeFields = FieldSet< field::_vrot, field::_arot >;
    static constexpr ComputeFields compute_field_set {};

    ADD_SLOT( MPI_Comm                    , mpi  , INPUT , MPI_COMM_WORLD);
    ADD_SLOT( GridT                       , grid , INPUT_OUTPUT , REQUIRED );
    ADD_SLOT( GridCellParticleInteraction , ges  , INPUT , DocString{"Interaction list"} );
    ADD_SLOT( Classifier                  , ic   , INPUT_OUTPUT , DocString{"Interaction lists classified according to their types"} );
    ADD_SLOT( double                      , volume , INPUT, REQUIRED , DocString{"Volume of the domain simulation. >0 "} );
    ADD_SLOT( Mat3d                       , stress_tensor , OUTPUT , DocString{"Write an Output file containing stress tensors."} );

    public:

    inline std::string documentation() const override final
    {
      return R"EOF( This operator computes the total stress tensor and the stress tensor for each particles. )EOF";
    }

    inline void execute () override final
    {
      if( grid->number_of_cells() == 0 ) 
			{
				 return; 
			}

      // get slot data
			auto cells = grid->cells();	
			Classifier& cf = *ic;
			exanb::Mat3d& stress = *stress_tensor;
			stress = exanb::make_zero_matrix();

      // 
      if(!ic.has_value()) 
			{
				return;
			}

			size_t types = cf.number_of_waves();
      bool sym = true;

# pragma omp parallel
			{
				Mat3d TLS = exanb::make_zero_matrix(); // Thread Local Storage
				for( size_t type = 0 ; type < types ; type++ )
				{
					auto [Ip, size]           = cf.get_info(type); // get interactions
					auto [dnp, cpp, fnp, ftp] = cf.buffer_p(type); // get forces (fn, ft) and contact positions (cp) computed into the contact force operators.
#pragma omp for schedule(static)
					for(size_t i = 0 ; i < size ; i++)
					{
            // get fij and cij
						Interaction& I = Ip[i];
						auto& cell     = cells[I.cell_i];
						Vec3d fij      = fnp[i] + ftp[i];
						Vec3d pos_i    = { cell[ field::rx ][I.p_i], cell[ field::ry ][I.p_i], cell[ field::rz ][I.p_i] };
						Vec3d cij      = cpp[i] - pos_i;
						TLS += exanb::tensor( fij, cij );

            if(I.type <= 3 && sym == true) // polyhedron - polyhedron || sphere - sphere
            {
						  auto& cellj    = cells[I.cell_j];
              Vec3d fji      = -fij;
              Vec3d pos_j    = { cellj[ field::rx ][I.p_j], cellj[ field::ry ][I.p_j], cellj[ field::rz ][I.p_j] };
              Vec3d cji      = cpp[i] - pos_j;
						  TLS += exanb::tensor( fji, cji );
            }

            // compute tensor
					}
				}
#pragma omp critical
				{
					stress += TLS;
				}
			}

			// get reduction over mpi processes
			double buff[9] = {
				stress.m11, stress.m12, stress.m13,
				stress.m21, stress.m22, stress.m23,
				stress.m31, stress.m32, stress.m33 };

			MPI_Allreduce(MPI_IN_PLACE, buff, 9, MPI_DOUBLE, MPI_SUM, *mpi);
			stress = {
				buff[0], buff[1], buff[2],
				buff[3], buff[4], buff[5],
				buff[6], buff[7], buff[8] };

			assert( *volume > 0 ) ;
			stress = stress / (*volume); 
		}
	};

	template<class GridT> using StressTensorTmpl = StressTensor<GridT>;

	// === register factories ===  
	CONSTRUCTOR_FUNCTION
	{
		OperatorNodeFactory::instance()->register_factory( "stress_tensor", make_grid_variant_operator< StressTensorTmpl > );
	}
}
