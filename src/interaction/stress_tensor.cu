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
#include <exanb/core/concurent_add_contributions.h>
#include <onika/parallel/parallel_for.h>


#include <memory>
#include <mpi.h>

#include <exaDEM/interaction/interaction.hpp>
#include <exaDEM/classifier/interactionSOA.hpp>
#include <exaDEM/classifier/interactionAOS.hpp>
#include <exaDEM/classifier/classifier.hpp>
#include <exaDEM/classifier/classifier_for_all.hpp>
#include <exaDEM/type/add_contribution_mat3d.hpp>

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
            const double *const __restrict__ dnp, 
            const Vec3d *const __restrict__ fnp, 
            const Vec3d *const __restrict__ ftp, 
            const Vec3d *const __restrict__ cpp) const
        {
          assert( type == I.type );
          if( dnp[idx] < 0.0)
          {
            // get fij and cij
            auto &cell = cells[I.cell_i];
            Vec3d fij = fnp[idx] + ftp[idx];
            Vec3d pos_i = {cell[field::rx][I.p_i], cell[field::ry][I.p_i], cell[field::rz][I.p_i]};
            Vec3d cij = cpp[idx] - pos_i;
            exanb::mat3d_atomic_add_contribution(cell[field::stress][I.p_i], exanb::tensor(fij, cij));

            if constexpr ( type <= 3 && sym == true) // polyhedron - polyhedron || sphere - sphere
            {
              auto &cellj = cells[I.cell_j];
              Vec3d fji = -fij;
              Vec3d pos_j = {cellj[field::rx][I.p_j], cellj[field::ry][I.p_j], cellj[field::rz][I.p_j]};
              Vec3d cji = cpp[idx] - pos_j;
              exanb::mat3d_atomic_add_contribution(cellj[field::stress][I.p_j], exanb::tensor(fji, cji));
            }
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
          auto [Ip, size] = classifier.get_info(Type); // get interactions
          if (size > 0 )
          {
            ParallelForOptions opts;
            opts.omp_scheduling = OMP_SCHED_STATIC;
            const auto [dnp, cpp, fnp, ftp] = classifier.buffer_p(Type); // get parameters: get forces (fn, ft) and contact positions (cp) computed into the contact force operators.
            InteractionWrapper<InteractionSOA> interactions(Ip);         // get data: interaction
            compute_stress_tensor<Type, Sym> func;                       // get kernel
            WrapperForAll wrapper(interactions, func , cells, dnp, fnp, ftp, cpp); // pack data, kernel, and interaction in a wrapper
            parallel_for(size, wrapper, oper->parallel_execution_context(), opts); // launch kernel
          }
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
    
  template<int NTypes, bool Sym, typename Op>
    struct compute_stress_tensors2
    {
      Op* oper;

      template <int Type, typename TMPLC>
        void iteration(Classifier2& classifier, TMPLC *const __restrict__ cells)
        {
          static_assert(Type >= 0 && Type < NTypes);
          auto [Ip, size] = classifier.get_info(Type); // get interactions
          if (size > 0 )
          {
            ParallelForOptions opts;
            opts.omp_scheduling = OMP_SCHED_STATIC;
            const auto [dnp, cpp, fnp, ftp] = classifier.buffer_p(Type); // get parameters: get forces (fn, ft) and contact positions (cp) computed into the contact force operators.
            //InteractionWrapper<InteractionSOA> interactions(Ip);         // get data: interaction
            compute_stress_tensor<Type, Sym> func;                       // get kernel
            WrapperForAll2 wrapper(Ip, func , cells, dnp, fnp, ftp, cpp); // pack data, kernel, and interaction in a wrapper
            parallel_for(size, wrapper, oper->parallel_execution_context(), opts); // launch kernel
          }
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


  template <typename GridT, class = AssertGridHasFields<GridT, field::_stress>> class StressTensor : public OperatorNode
  {
    // attributes processed during computation
    ADD_SLOT(MPI_Comm, mpi, INPUT, MPI_COMM_WORLD);
    ADD_SLOT(GridT, grid, INPUT_OUTPUT, REQUIRED);
    ADD_SLOT(Classifier<InteractionSOA>, ic, INPUT, DocString{"Interaction lists classified according to their types"});
    
    ADD_SLOT(Classifier2, ic2, INPUT);

    public:
    inline std::string documentation() const override final { return R"EOF( This operator computes the total stress tensor and the stress tensor for each particles. )EOF"; }

    inline void execute() override final
    {
      //printf("STRESS\n");
      if (grid->number_of_cells() == 0) { return; }
      if (!ic.has_value()) { return; }

      // get slot data
      auto cells = grid->cells();
      Classifier<InteractionSOA>& cf = *ic;
      //Classifier2& cf = *ic2;
      // get kernel
      constexpr bool sym = true;
      //compute_stress_tensors2<13, sym, StressTensor> runner = {this}; // 13 is the number of interaction types
                                                                     // iterate over types
      compute_stress_tensors<13, sym, StressTensor> runner = {this}; // 13 is the number of interaction types
                                                                     // iterate over types
                                                                     
      runner(cf, cells);
      //printf("STRESS2\n");
    }
  };

  template <class GridT> using StressTensorTmpl = StressTensor<GridT>;

  // === register factories ===
  ONIKA_AUTORUN_INIT(stress_tensor) { OperatorNodeFactory::instance()->register_factory("stress_tensor", make_grid_variant_operator<StressTensorTmpl>); }
} // namespace exaDEM
