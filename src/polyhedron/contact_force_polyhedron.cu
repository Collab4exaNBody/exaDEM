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
#include <exanb/core/xform.h>

#include <memory>

#include <exaDEM/vertices.hpp>
#include <exaDEM/contact_force_parameters.h>
#include <exaDEM/compute_contact_force.h>

#include <exaDEM/interaction/grid_cell_interaction.hpp>
#include <exaDEM/classifier/classifier.hpp>
#include <exaDEM/classifier/classifier_for_all.hpp>
#include <exaDEM/itools/itools.hpp>
#include <exaDEM/shapes.hpp>
#include <exaDEM/shape_detection.hpp>
#include <exaDEM/shape_detection_driver.hpp>
#include <exaDEM/drivers.h>
#include <exaDEM/multimat_parameters.h>
#include <exaDEM/contact_polyhedron.h>

namespace exaDEM
{
  using namespace exanb;
  using namespace polyhedron;


  template <bool multimat, bool cohesive, typename GridT, class = AssertGridHasFields<GridT, field::_radius>> 
    class ComputeContactClassifierPolyhedron : public OperatorNode
  {
    ADD_SLOT(GridT, grid, INPUT_OUTPUT, REQUIRED);
    ADD_SLOT(CellVertexField, cvf, INPUT, REQUIRED, DocString{"Store vertex positions for every polyhedron"});
    ADD_SLOT(Domain , domain, INPUT , REQUIRED );
    ADD_SLOT(ContactParams, config, INPUT, OPTIONAL);        // can be re-used for to dump contact network
    ADD_SLOT(ContactParams, config_driver, INPUT, OPTIONAL); // can be re-used for to dump contact network
    ADD_SLOT(MultiMatParamsT<ContactParams>, multimat_cp, INPUT, OPTIONAL, DocString{"List of contact parameters for simulations with multiple materials"});

    ADD_SLOT(double, dt, INPUT, REQUIRED);
    ADD_SLOT(bool, symetric, INPUT_OUTPUT, true, DocString{"Activate the use of symetric feature (contact law)"});
    ADD_SLOT(Drivers, drivers, INPUT, DocString{"List of Drivers {Cylinder, Surface, Ball, Mesh}"});
    ADD_SLOT(Classifier, ic, INPUT_OUTPUT, DocString{"Interaction lists classified according to their types"});
    ADD_SLOT(shapes, shapes_collection, INPUT_OUTPUT, DocString{"Collection of shapes"});
    // analyses
    ADD_SLOT(long, timestep, INPUT, REQUIRED);
    ADD_SLOT(std::string, dir_name, INPUT, REQUIRED, DocString{"Output directory name."});
    ADD_SLOT(std::string, interaction_basename, INPUT, REQUIRED, DocString{"Write an Output file containing interactions."});
    ADD_SLOT(long, analysis_interaction_dump_frequency, INPUT, REQUIRED, DocString{"Write an interaction dump file"});
    // private
    ADD_SLOT(bool, print_warning, PRIVATE, true, DocString{"This variable is used to ensure that warning messages are displayed only once."});
    // output
    ADD_SLOT(double, max_kn, INPUT_OUTPUT, 0, DocString{"Get the highest value of the input contact force parameters kn (used for dt_critical)"});

    public:

    inline std::string operator_name() { return "contact_polyhedron"; }

    inline std::string documentation() const override final { 
      return R"EOF(
        This operator computes forces between particles and particles/drivers using the contact law.
 
        Note that to use multmaterials version, you need to predefine the parameters with multimat_contact_params and drivers_contact_params.  

        YAML example:

					one_parameter_no_cohesion:
						- contact_polyhedron:
							 config: { kn: 10000, kt: 10000, kr: 0.1, mu: 0.1, damp_rate: 0.9}
							 config_driver: { kn: 10000, kt: 10000, kr: 0.1, mu: 0.3, damp_rate: 0.9}

					one_parameter_with_cohesion:
						- contact_polyhedron_with_cohesion:
							 config: { dncut: 0.1 m, kn: 10000, kt: 10000, kr: 0.1, fc: 0.05, mu: 0.1, damp_rate: 0.9}
							 config_driver: { dncut: 0.1 m, kn: 10000, kt: 10000, kr: 0.1, fc: 0.05, mu: 0.3, damp_rate: 0.9}

					multi_parameters_no_cohesion:
						- contact_polyhedron_multimat

					multi_parameters_with_cohesion:
						- contact_polyhedron_multimat_cohesion
      )EOF"; 
    }

    template<int start, int end, template<int, bool, typename> typename FuncT, typename XFormT, typename... Args>
      void loop_contact_force(Classifier& classifier, XFormT& cp_xform, Args &&... args)
      {
        FuncT<start, cohesive, XFormT> contact_law;
        contact_law.xform = cp_xform;
        run_contact_law<start>(parallel_execution_context(), classifier, contact_law, args...);
        if constexpr( start + 1 <= end )
        {
          loop_contact_force<start+1, end, FuncT>(classifier, cp_xform, std::forward<Args>(args)...);
        }
      }

    /** fill highest kn */
    void scan_kn()
    {
      if(!(*print_warning)) return;
      double kn = 0.0;
      if(config.has_value()) kn = std::max(kn, config->kn);
      if(config_driver.has_value()) kn = std::max(kn, config->kn);
      if(multimat_cp.has_value())
      {
        auto get_max_kn = [&kn] (const ContactParams& cp) -> void { kn = std::max(kn, cp.kn); };
        multimat_cp->apply(get_max_kn);
      }
      *max_kn = kn;
    }

    inline void execute() override final
    {
      if (grid->number_of_cells() == 0)
      {
        return;
      }

      check_slots();
      scan_kn();

      /** Analysis */
      const long frequency_interaction = *analysis_interaction_dump_frequency;
      bool write_interactions = (frequency_interaction > 0 && (*timestep) % frequency_interaction == 0);


      /** Get driver, vertices and particles data */
      const DriversGPUAccessor drvs = *drivers;
      const auto cells = grid->cells();
      auto* vertex_fields = cvf->data();

      /** Get Shape */
      const shape *const shps = shapes_collection->data();

      /** deform matrice */
      const Mat3d& xform = domain->xform();
      bool is_def_xform = !domain->xform_is_identity();

      const double time = *dt;
      auto &classifier = *ic;

      /** Contact fexaDEM/orce kernels */
      contact_law_driver<cohesive, Cylinder> cyli;
      contact_law_driver<cohesive, Surface> surf;
      contact_law_driver<cohesive, Ball> ball;


#     define __params__ cells, vertex_fields, cp, shps, time
#     define __params_driver__ cells, vertex_fields, drvs, cp_drvs, shps, time

      constexpr int poly_type_start = 0;
      constexpr int poly_type_end = 3;
      constexpr int stl_type_start = 7;
      constexpr int stl_type_end = 12;

      if constexpr (!multimat) /** Single mat */
      {
        const ContactParams& hkp = *config;
        ContactParams hkp_drvs;

        if (drivers->get_size() > 0 )
        {
          hkp_drvs = *config_driver;
        }

        const SingleMatContactParamsTAccessor<ContactParams> cp = {hkp};
        const SingleMatContactParamsTAccessor<ContactParams> cp_drvs = {hkp_drvs};

        if(is_def_xform)
        {
          LinearXForm cp_xform = {xform};
          loop_contact_force<poly_type_start, poly_type_end, contact_law>(classifier, cp_xform, __params__);
          loop_contact_force <stl_type_start,  stl_type_end, contact_law_stl>(classifier, cp_xform, __params_driver__);
        }
        else
        {
          NullXForm cp_xform;
          loop_contact_force<poly_type_start, poly_type_end, contact_law>(classifier, cp_xform, __params__);
          loop_contact_force <stl_type_start,  stl_type_end, contact_law_stl>(classifier, cp_xform, __params_driver__);
        }
        run_contact_law<InteractionTypeId::VertexCylinder>(parallel_execution_context(), classifier, cyli, __params_driver__);
        run_contact_law<InteractionTypeId::VertexSurface> (parallel_execution_context(), classifier, surf, __params_driver__);
        run_contact_law<InteractionTypeId::VertexBall>    (parallel_execution_context(), classifier, ball, __params_driver__);
      }
      else /** Multi materials */
      {
        const auto& contact_parameters = *multimat_cp;
        const MultiMatContactParamsTAccessor<ContactParams> cp = contact_parameters.get_multimat_accessor();
        const MultiMatContactParamsTAccessor<ContactParams> cp_drvs = contact_parameters.get_drivers_accessor();

        if(is_def_xform)
        {
          LinearXForm cp_xform = {xform};
          loop_contact_force<poly_type_start, poly_type_end, contact_law>(classifier, cp_xform, __params__);
          loop_contact_force <stl_type_start,  stl_type_end, contact_law_stl>(classifier, cp_xform, __params_driver__);
        }
        else
        {
          NullXForm cp_xform;
          loop_contact_force<poly_type_start, poly_type_end, contact_law>(classifier, cp_xform, __params__);
          loop_contact_force <stl_type_start,  stl_type_end, contact_law_stl>(classifier, cp_xform, __params_driver__);
        }
        run_contact_law<InteractionTypeId::VertexCylinder>(parallel_execution_context(), classifier, cyli, __params_driver__);
        run_contact_law<InteractionTypeId::VertexSurface> (parallel_execution_context(), classifier, surf, __params_driver__);
        run_contact_law<InteractionTypeId::VertexBall>    (parallel_execution_context(), classifier, ball, __params_driver__);
      }

#undef __params__
#undef __params_driver__

      if (write_interactions)
      {
        auto stream = itools::create_buffer(*grid, classifier);
        std::string ts = std::to_string(*timestep);
        itools::write_file(stream, (*dir_name), (*interaction_basename) + ts);
      }
    }

    void check_slots()
    {
      bool pw = *print_warning;

      if (*symetric == false)
      {
        color_log::error("contact_polyhedron", "The parameter symetric in contact classifier polyhedron has to be set to true.");
      }

      /** Some check mutlimat versus singlemat */
      if constexpr  (multimat) /** Multiple materials */
      {
        if( !multimat_cp.has_value() )
        {
          std::string msg = "You are using the multi-material contact force model, but the contact law parameters have not been defined.\n";
          msg +=            "Please specify the parameter values for each material pair using the operator \"multimat_contact_params\".";
          color_log::error(operator_name(), msg);
        }
        if( (*print_warning) && config.has_value() )
        {
          color_log::warning(operator_name(), "You are using the multi-material contact force operator, but you have also defined the input slot \"config\" which is intended for the single-material version. This slot will be ignored.");
          pw = false;
        }
        if( (*print_warning) && config_driver.has_value() )
        {
          color_log::warning(operator_name(), "You are using the multi-material contact force operator, but you have also defined the input slot \"config_driver\" which is intended for the single-material version. This slot will be ignored.");
          pw = false;
        }
      }
      if constexpr  (!multimat) /** Single material */
      {
        if( !config.has_value() )
        {
          std::string msg = "The input slot \"config\" is not defined, yet the single-material version of the contact operator is being used.\n";
          msg            += "Please specify the \"config\" input slot, and use the \"config_driver\" slot if you want to define a contact law between a particle and a driver."; 
          color_log::error(operator_name(), msg);
        }
        if( (*print_warning) && multimat_cp.has_value() )
        {
          std::string msg = "You have defined a list of contact law parameters for different material types,\n";
          msg            += "but you are using the version that only considers the parameter defined in the \"config\" input slot. \n";
          msg            += "The parameter list will be ignored. If you want to use it, please use the operator \n";
          msg            += "\"contact_sphere_multimat\" or \"contact_sphere_multimat_with_cohesion\".";
          color_log::warning(operator_name(), msg);
          pw = false;
        }
        /** Some global checks */
        /** Is cohesive force define while it's not used */
        if constexpr (!cohesive)
        {
          if(config->dncut > 0)
          {
            std::string msg = "dncut is != 0 while the cohesive force is not used.\n";
            msg            += "Please, use contact_sphere_with_cohesion operator.";
            color_log::error(operator_name(), msg);
          }
          if(drivers->get_size() > 0 && config_driver->dncut > 0)
          {
            std::string msg = "dncut is != 0 while the cohesive force is not used.\n";
            msg            += "Please, use contact_sphere_with_cohesion operator.";
            color_log::error(operator_name(), msg);
          }
        }
      }
      *print_warning = pw;
    }
  };

  template <class GridT> using ComputeContactClassifierPolySingleMatTmpl         = ComputeContactClassifierPolyhedron<false, false, GridT>;
  template <class GridT> using ComputeContactClassifierPolySingleMatCohesionTmpl = ComputeContactClassifierPolyhedron<false,  true, GridT>;
  template <class GridT> using ComputeContactClassifierPolyMultiMatTmpl          = ComputeContactClassifierPolyhedron< true, false, GridT>;
  template <class GridT> using ComputeContactClassifierPolyMultiMatCohesionTmpl  = ComputeContactClassifierPolyhedron< true,  true, GridT>;

  // === register factories ===
  ONIKA_AUTORUN_INIT(contact_force_polyhedron) { OperatorNodeFactory::instance()->register_factory("contact_polyhedron", make_grid_variant_operator<ComputeContactClassifierPolySingleMatTmpl>); }
  ONIKA_AUTORUN_INIT(contact_force_polyhedron_with_cohesion) { OperatorNodeFactory::instance()->register_factory("contact_polyhedron_with_cohesion", make_grid_variant_operator<ComputeContactClassifierPolySingleMatCohesionTmpl>); }
  ONIKA_AUTORUN_INIT(contact_force_polyhedron_sm) { OperatorNodeFactory::instance()->register_factory("contact_polyhedron_singlemat", make_grid_variant_operator<ComputeContactClassifierPolySingleMatTmpl>); }
  ONIKA_AUTORUN_INIT(contact_force_polyhedron_sm_with_cohesion) { OperatorNodeFactory::instance()->register_factory("contact_polyhedron_singlemat_with_cohesion", make_grid_variant_operator<ComputeContactClassifierPolySingleMatCohesionTmpl>); }
  ONIKA_AUTORUN_INIT(contact_force_polyhedron_mm) { OperatorNodeFactory::instance()->register_factory("contact_polyhedron_multimat", make_grid_variant_operator<ComputeContactClassifierPolyMultiMatTmpl>); }
  ONIKA_AUTORUN_INIT(contact_force_polyhedron_mm_with_cohesion) { OperatorNodeFactory::instance()->register_factory("contact_polyhedron_multimat_with_cohesion", make_grid_variant_operator<ComputeContactClassifierPolyMultiMatCohesionTmpl>); }
} // namespace exaDEM
