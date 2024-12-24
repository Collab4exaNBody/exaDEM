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
#include <exanb/core/concurent_add_contributions.h>
#include <onika/parallel/parallel_for.h>


#include <memory>
#include <mpi.h>

#include <exaDEM/interaction/interaction.hpp>
#include <exaDEM/interaction/interactionSOA.hpp>
#include <exaDEM/interaction/interactionAOS.hpp>
#include <exaDEM/interaction/classifier.hpp>
#include <exaDEM/interaction/classifier_for_all.hpp>

namespace exaDEM
{
	using namespace exanb;
  using namespace onika::parallel;

	template <int type, bool sym>
		struct compute_stress_tensor
		{
			template <typename TMPLC> 
      ONIKA_HOST_DEVICE_FUNC 
      inline void operator()(
            uint64_t idx,
						Interaction& I, 
						TMPLC *const __restrict__ cells, 
						Vec3d *const __restrict__ fnp, 
						Vec3d *const __restrict__ ftp, 
						Vec3d *const __restrict__ cpp) const
				{
					assert( type == I.type );
					// get fij and cij
					auto &cell = cells[I.cell_i];
					Vec3d fij = fnp[idx] + ftp[idx];
					Vec3d pos_i = {cell[field::rx][I.p_i], cell[field::ry][I.p_i], cell[field::rz][I.p_i]};
					Vec3d cij = cpp[idx] - pos_i;
					exanb::atomic_add_contribution(cell[field::stress][I.p_i], exanb::tensor(fij, cij));

					if constexpr ( type <= 3 && sym == true) // polyhedron - polyhedron || sphere - sphere
					{
						auto &cellj = cells[I.cell_j];
						Vec3d fji = -fij;
						Vec3d pos_j = {cellj[field::rx][I.p_j], cellj[field::ry][I.p_j], cellj[field::rz][I.p_j]};
						Vec3d cji = cpp[idx] - pos_j;
						exanb::atomic_add_contribution(cell[field::stress][I.p_j], exanb::tensor(fji, cji));
					}
				}
		};

	template<int NTypes, bool Sym, typename Op>
		struct compute_stress_tensors
		{
			Op* oper;

			template <int Type, typename TMPLC>
				void iteration(Classifier<InteractionSOA>& classifier, TMPLC *const __restrict__ cells)
				{
					static_assert(Type >= 0 && Type < NTypes);
					ParallelForOptions opts;
					opts.omp_scheduling = OMP_SCHED_STATIC;
					auto [Ip, size] = classifier.get_info(Type);           // get interactions
					auto [dnp, cpp, fnp, ftp] = classifier.buffer_p(Type); // get forces (fn, ft) and contact positions (cp) computed into the contact force operators.
					InteractionWrapper<InteractionSOA> interactions(Ip);
					compute_stress_tensor<Type, Sym> func;
					WrapperForAll wrapper(interactions, func , cells, fnp, ftp, cpp);
					parallel_for(size, wrapper, oper->parallel_execution_context(), opts);
				}

			template<int Type, typename... Args> void loop(Args... args)
			{
				iteration<Type>(args...);
				if constexpr (Type - 1 >= 0) loop<Type-1> (args...); 
			}

			template<typename... Args> void operator()(Args&&... args)
			{
				static_assert(NTypes >= 1);
				loop<NTypes-1>(args...);
			}
		};


	template <typename GridT, class = AssertGridHasFields<GridT, field::_rx, field::_ry, field::_rz>> class StressTensor : public OperatorNode
	{
		// attributes processed during computation
		using ComputeFields = FieldSet<field::_vrot, field::_arot>;
		static constexpr ComputeFields compute_field_set{};

		ADD_SLOT(MPI_Comm, mpi, INPUT, MPI_COMM_WORLD);
		ADD_SLOT(GridT, grid, INPUT_OUTPUT, REQUIRED);
		ADD_SLOT(GridCellParticleInteraction, ges, INPUT, DocString{"Interaction list"});
		ADD_SLOT(Classifier<InteractionSOA>, ic, INPUT_OUTPUT, DocString{"Interaction lists classified according to their types"});
		ADD_SLOT(double, volume, INPUT, REQUIRED, DocString{"Volume of the domain simulation. >0 "});
		ADD_SLOT(Mat3d, stress_tensor, OUTPUT, DocString{"Write an Output file containing stress tensors."});

		public:
		inline std::string documentation() const override final { return R"EOF( This operator computes the total stress tensor and the stress tensor for each particles. )EOF"; }

		inline void execute() override final
		{
			if (grid->number_of_cells() == 0) { return; }

			// get slot data
			auto cells = grid->cells();
			Classifier<InteractionSOA> &cf = *ic;
			constexpr bool sym = true;
			compute_stress_tensors<13, sym, StressTensor> runner = {this}; // 13 is the number of types
			// iterate over types
			runner(cf, cells);
		}
	};

	template <class GridT> using StressTensorTmpl = StressTensor<GridT>;

	// === register factories ===
	CONSTRUCTOR_FUNCTION { OperatorNodeFactory::instance()->register_factory("stress_tensor", make_grid_variant_operator<StressTensorTmpl>); }
} // namespace exaDEM
