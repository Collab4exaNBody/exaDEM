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
#include <exanb/core/domain.h>

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
#include <exaDEM/mutexes.h>
#include <exaDEM/drivers.h>
#include <exaDEM/contact_sphere.h>

namespace exaDEM
{
  using namespace exanb;
  using namespace sphere;

  template <typename GridT, class = AssertGridHasFields<GridT, field::_radius>> class ComputeContactClassifierSphereGPU : public OperatorNode
  {
    // attributes processed during computation
    using ComputeFields = FieldSet<field::_vrot, field::_arot>;
    static constexpr ComputeFields compute_field_set{};

    ADD_SLOT(GridT, grid, INPUT_OUTPUT, REQUIRED);
    ADD_SLOT(Domain , domain, INPUT , REQUIRED );
    ADD_SLOT(ContactParams, config, INPUT, REQUIRED, DocString{"Contact parameters for sphere interactions"});      // can be re-used for to dump contact network
    ADD_SLOT(ContactParams, config_driver, INPUT, OPTIONAL, DocString{"Contact parameters for drivers, optional"}); // can be re-used for to dump contact network
    ADD_SLOT(double, dt, INPUT, REQUIRED, DocString{"Time step value"});
    ADD_SLOT(bool, symetric, INPUT_OUTPUT, REQUIRED, DocString{"Activate the use of symetric feature (contact law)"});
    ADD_SLOT(Drivers, drivers, INPUT, REQUIRED, DocString{"List of Drivers {Cylinder, Surface, Ball, Mesh}"});
    ADD_SLOT(Classifier<InteractionSOA>, ic, INPUT_OUTPUT, DocString{"Interaction lists classified according to their types"});
    // analysis
    ADD_SLOT(long, timestep, INPUT, REQUIRED);
    ADD_SLOT(long, analysis_interaction_dump_frequency, INPUT, REQUIRED, DocString{"Write an interaction dump file"});
    ADD_SLOT(std::string, dir_name, INPUT, REQUIRED, DocString{"Output directory name."});
    ADD_SLOT(std::string, interaction_basename, INPUT, REQUIRED, DocString{"Write an Output file containing interactions."});

  public:
    inline std::string documentation() const override final { return R"EOF(This operator computes forces between particles and particles/drivers using the contact law.)EOF"; }

    template<int start, int end, template<int, bool> typename FuncT, bool def_box, typename T, typename... Args>
    void loop_contact_force(Classifier<T>& classifier, Args &&... args)
    {
      FuncT<start, def_box> contact_law;
      run_contact_law(parallel_execution_context(), start, classifier, contact_law, args...);
      if constexpr( start + 1 <= end )
      {
        loop_contact_force<start+1, end, FuncT, def_box>(classifier, std::forward<Args>(args)...);
      }
    }

    template<bool is_sym, bool def_box>
    void core()
    {
      const DriversGPUAccessor drvs = *drivers;
      auto *cells = grid->cells();
      const ContactParams hkp = *config;
      ContactParams hkp_drvs{};

      /** Def Box */
      const Mat3d& xform = domain->xform();

      if (drivers->get_size() > 0 && config_driver.has_value())
      {
        hkp_drvs = *config_driver;
      }

      const double time = *dt;
      auto &classifier = *ic;

      contact_law<is_sym, def_box> sph = {xform};
      contact_law_driver<Cylinder, def_box> cyl = {xform};
      contact_law_driver<Surface, def_box> surf = {xform};
      contact_law_driver<Ball, def_box> ball = {xform};

      run_contact_law(parallel_execution_context(), 0, classifier, sph, cells, hkp, time);
      run_contact_law(parallel_execution_context(), 4, classifier, cyl, cells, drvs, hkp_drvs, time);
      run_contact_law(parallel_execution_context(), 5, classifier, surf, cells, drvs, hkp_drvs, time);
      run_contact_law(parallel_execution_context(), 6, classifier, ball, cells, drvs, hkp_drvs, time);

      constexpr int stl_type_start = 7;
      constexpr int stl_type_end = 9;
      loop_contact_force <stl_type_start,  stl_type_end, contact_law_stl, def_box>(classifier, cells, drvs, hkp_drvs, time);
    }

    void save_results()
    {
      auto &classifier = *ic;
      auto stream = itools::create_buffer(*grid, classifier);
      std::string ts = std::to_string(*timestep);
      itools::write_file(stream, *dir_name, (*interaction_basename) + ts);
    }

    inline void execute() override final
    {
      printf("CONTACT\n");
      if (grid->number_of_cells() == 0)
      {
        return;
      }

      /** Def Box */
      bool is_def_box = !domain->xform_is_identity();

      /** Analysis */
      const long frequency_interaction = *analysis_interaction_dump_frequency;
      bool write_interactions = (frequency_interaction > 0 && (*timestep) % frequency_interaction == 0);

      if(*symetric == false && is_def_box == false) {core<false, false>();}      
      if(*symetric == true  && is_def_box == false) {core<true , false>();}      
      if(*symetric == false && is_def_box ==  true) {core<false,  true>();}      
      if(*symetric == true  && is_def_box ==  true) {core<true ,  true>();}      

      if (write_interactions)
      {
        save_results();
      }
      printf("CONTACT FIN\n");
    }
  };

  template <class GridT> using ComputeContactClassifierGPUTmpl = ComputeContactClassifierSphereGPU<GridT>;

  // === register factories ===
  ONIKA_AUTORUN_INIT(contact_force_sphere) { OperatorNodeFactory::instance()->register_factory("contact_sphere", make_grid_variant_operator<ComputeContactClassifierGPUTmpl>); }
} // namespace exaDEM
