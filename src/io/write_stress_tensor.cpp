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
#include <filesystem> // C++17


namespace exaDEM
{
  using namespace exanb;

  template<typename GridT , class = AssertGridHasFields< GridT>>
    class WriteStressTensor : public OperatorNode
  {
    // attributes processed during computation
    using ComputeFields = FieldSet< field::_vrot, field::_arot >;
    static constexpr ComputeFields compute_field_set {};

    ADD_SLOT( MPI_Comm                    , mpi  , INPUT , MPI_COMM_WORLD);
    ADD_SLOT( Mat3d                       , stress_tensor , INPUT , REQUIRED , DocString{"Write an Output file containing stress tensors."} );
    ADD_SLOT( std::string                 , dir_name  , INPUT , "ExaDEMOutputDir", DocString{"Write an Output file containing stress tensors."} );
    ADD_SLOT( std::string                 , file_name , INPUT , "AvgStresTensor.txt", DocString{"Write an Output file containing stress tensors."} );
    ADD_SLOT( long                        , timestep  , INPUT , REQUIRED , DocString{"Iteration number"} );
    ADD_SLOT( double                      , dt        , INPUT , REQUIRED );

    public:

    inline std::string documentation() const override final
    {
      return R"EOF( Write the average stensor tensor. )EOF";
    }

    inline void execute () override final
    {
      namespace fs = std::filesystem;
			std::string full_path = (*dir_name) + "/" + (*file_name);
      fs::path path(full_path);
			const Mat3d& stress = *stress_tensor;

      int rank;
      MPI_Comm_rank(*mpi, &rank);
      fs::create_directory(*dir_name);

      if ( rank == 0 )
      {
        std::ofstream file;
        fs::create_directory(*dir_name);
        if(! fs::exists(path) )
        {
          file.open(full_path);
          file << "Time Sxx Sxy Sxz Syx Syy Syz Szx Szy Szz" << std::endl;
        }
        else
        {
          file.open(full_path, std::ofstream::in | std::ofstream::ate); 
        }
        double t = (*dt) * (*timestep);
        file << t << " " 
            << stress.m11 <<  " " << stress.m12 << " " << stress.m13 
            << stress.m21 <<  " " << stress.m22 << " " << stress.m23 
            << stress.m31 <<  " " << stress.m32 << " " << stress.m33 << std::endl;
      }
		}
	};

	template<class GridT> using WriteStressTensorTmpl = WriteStressTensor<GridT>;

	// === register factories ===  
	CONSTRUCTOR_FUNCTION
	{
		OperatorNodeFactory::instance()->register_factory( "write_stress_tensor", make_grid_variant_operator< WriteStressTensorTmpl > );
	}
}
