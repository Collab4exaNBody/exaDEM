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
//#pragma xstamp_cuda_enable //! DO NOT REMOVE THIS LINE

#include <exanb/core/operator.h>
#include <exanb/core/operator_slot.h>
#include <exanb/core/operator_factory.h>
#include <exanb/core/make_grid_variant_operator.h>
#include <exanb/core/parallel_grid_algorithm.h>
#include <exanb/core/grid.h>

#include <memory>

#include <exaDEM/contact_force_parameters.h>
#include <exaDEM/compute_contact_force.h>

#include <exaDEM/interaction/interaction.hpp>
#include <exaDEM/classifier/interactionSOA.hpp>
#include <exaDEM/classifier/interactionAOS.hpp>
#include <exaDEM/interaction/grid_cell_interaction.hpp>
#include <exaDEM/classifier/classifier.hpp>
#include <exaDEM/classifier/classifier_for_all.hpp>
#include <exaDEM/itools/itools.hpp>
#include <exaDEM/shapes.hpp>
#include <exaDEM/shape_detection.hpp>
#include <exaDEM/shape_detection_driver.hpp>
#include <exaDEM/drivers.h>
#include <exaDEM/contact_polyhedron.h>

namespace exaDEM
{
  using namespace exanb;
  using namespace polyhedron;


  template <typename GridT, class = AssertGridHasFields<GridT, field::_radius>> class ComputeContactClassifierPolyhedronGPU : public OperatorNode
  {
    using driver_t = std::variant<exaDEM::Cylinder, exaDEM::Surface, exaDEM::Ball, exaDEM::Stl_mesh, exaDEM::UndefinedDriver>;
    ADD_SLOT(GridT, grid, INPUT_OUTPUT, REQUIRED);
    ADD_SLOT(ContactParams, config, INPUT, REQUIRED);        // can be re-used for to dump contact network
    ADD_SLOT(ContactParams, config_driver, INPUT, OPTIONAL); // can be re-used for to dump contact network
    ADD_SLOT(double, dt, INPUT, REQUIRED);
    ADD_SLOT(bool, symetric, INPUT_OUTPUT, REQUIRED, DocString{"Activate the use of symetric feature (contact law)"});
    ADD_SLOT(Drivers, drivers, INPUT, DocString{"List of Drivers {Cylinder, Surface, Ball, Mesh}"});
    ADD_SLOT(Classifier<InteractionSOA>, ic, INPUT_OUTPUT, DocString{"Interaction lists classified according to their types"});
    ADD_SLOT(shapes, shapes_collection, INPUT_OUTPUT, DocString{"Collection of shapes"});
    // analyses
    ADD_SLOT(long, timestep, INPUT, REQUIRED);
    ADD_SLOT(std::string, dir_name, INPUT, REQUIRED, DocString{"Output directory name."});
    ADD_SLOT(std::string, interaction_basename, INPUT, REQUIRED, DocString{"Write an Output file containing interactions."});
    ADD_SLOT(long, analysis_interaction_dump_frequency, INPUT, REQUIRED, DocString{"Write an interaction dump file"});

    public:
    inline std::string documentation() const override final { return R"EOF(This operator computes forces between particles and particles/drivers using the contact law.)EOF"; }


    template<int start, int end, template<int> typename FuncT, typename T, typename... Args>
      void loop_contact_force(Classifier<T>& classifier, Args &&... args)
      {
        FuncT<start> contact_law;
        run_contact_law(parallel_execution_context(), start, classifier, contact_law, args...);
        if constexpr( start + 1 <= end )
        {
          loop_contact_force<start+1, end, FuncT>(classifier, std::forward<Args>(args)...);
        }
      }

    inline void execute() override final
    {
      if (grid->number_of_cells() == 0)
      {
        return;
      }

      /** Analysis */
      const bool store_interactions = true;
      const long frequency_interaction = *analysis_interaction_dump_frequency;
      bool write_interactions = (frequency_interaction > 0 && (*timestep) % frequency_interaction == 0);

      /** Get driver and particles data */
      driver_t *drvs = drivers->data();
      const auto cells = grid->cells();

      /** Get Contact Parameters and Shape */
      const ContactParams hkp = *config;
      ContactParams hkp_drvs{};
      const shape *const shps = shapes_collection->data();

      if (drivers->get_size() > 0 && config_driver.has_value())
      {
        hkp_drvs = *config_driver;
      }

      const double time = *dt;
      auto &classifier = *ic;

      /** Contact fexaDEM/orce kernels */
      contact_law_driver<Cylinder> cyli;
      contact_law_driver<Surface> surf;
      contact_law_driver<Ball> ball;

      if (*symetric == false)
      {
        lout << "The parameter symetric in contact classifier polyhedron has to be set to true." << std::endl;
        std::abort();
      }

#     define __params__ store_interactions, cells, hkp, shps, time
#     define __params_driver__ store_interactions, cells, drvs, hkp_drvs, shps, time

      constexpr int poly_type_start = 0;
      constexpr int poly_type_end = 3;
      constexpr int stl_type_start = 7;
      constexpr int stl_type_end = 12;

      loop_contact_force<poly_type_start, poly_type_end,     contact_law>(classifier,        __params__);
      loop_contact_force <stl_type_start,  stl_type_end, contact_law_stl>(classifier, __params_driver__);
      run_contact_law(parallel_execution_context(), 4, classifier, cyli, __params_driver__);
      run_contact_law(parallel_execution_context(), 5, classifier, surf, __params_driver__);
      run_contact_law(parallel_execution_context(), 6, classifier, ball, __params_driver__);

#undef __params__
#undef __params_driver__

      if (write_interactions)
      {
        auto stream = itools::create_buffer(*grid, classifier);
        std::string ts = std::to_string(*timestep);
        itools::write_file(stream, (*dir_name), (*interaction_basename) + ts);
      }
    }
  };

  template <class GridT> using ComputeContactClassifierPolyGPUTmpl = ComputeContactClassifierPolyhedronGPU<GridT>;

  // === register factories ===
  CONSTRUCTOR_FUNCTION { OperatorNodeFactory::instance()->register_factory("contact_polyhedron", make_grid_variant_operator<ComputeContactClassifierPolyGPUTmpl>); }
} // namespace exaDEM
