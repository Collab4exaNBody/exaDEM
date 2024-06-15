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
#include <exanb/core/domain.h>
#include <exanb/core/basic_types.h>
#include <exanb/core/basic_types_operators.h>
#include <exanb/core/basic_types_stream.h>
#include <onika/memory/allocator.h> // for ONIKA_ASSUME_ALIGNED macro
#include <exanb/compute/compute_pair_optional_args.h>
#include <vector>
#include <iomanip>

#include <exanb/compute/compute_cell_particles.h>

#include <mpi.h>
#include <exaDEM/shape/shapes.hpp>
#include <exaDEM/shape/shape_reader.hpp>

namespace exaDEM
{
	using namespace exanb;
	template<	class GridT, class = AssertGridHasFields< GridT >> class ReadShapeFileOperator : public OperatorNode
	{
		ADD_SLOT( MPI_Comm , mpi      , INPUT , MPI_COMM_WORLD);
		ADD_SLOT( GridT    , grid     , INPUT_OUTPUT );
		ADD_SLOT( Domain   , domain   , INPUT , REQUIRED );
		ADD_SLOT( std::string , filename, INPUT , REQUIRED , DocString{"Input filename"});
		ADD_SLOT( shapes , shapes_collection, INPUT_OUTPUT , DocString{"Collection of shapes"});

		public:
		inline std::string documentation() const override final
		{
			return R"EOF( This operator initialize shapes data structure from a shape input file.
    	    			)EOF";
		}

		inline void execute () override final
		{
			auto& collection = *shapes_collection;
			lout << "Read file= " << *filename << std::endl;
			exaDEM::add_shape_from_file_shp(collection, *filename);
		};
	};

	// this helps older versions of gcc handle the unnamed default second template parameter
	template <class GridT> using ReadShapeFileOperatorTemplate = ReadShapeFileOperator<GridT>;

	// === register factories ===  
	CONSTRUCTOR_FUNCTION
	{
		OperatorNodeFactory::instance()->register_factory( "read_shape_file", make_grid_variant_operator< ReadShapeFileOperatorTemplate > );
	}
}
