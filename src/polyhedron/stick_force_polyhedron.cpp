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
#include <exaDEM/inner_bond_parameters.h>
#include <exaDEM/multimat_parameters.h>
#include <exaDEM/stick_polyhedron.hpp>

namespace exaDEM
{
  using namespace exanb;
  using namespace polyhedron;


  template <bool multimat, typename GridT, class = AssertGridHasFields<GridT, field::_radius>> 
    class ComputeStickForce : public OperatorNode
  {
    ADD_SLOT(GridT, grid, INPUT_OUTPUT, REQUIRED);
    ADD_SLOT(CellVertexField, cvf, INPUT, REQUIRED, DocString{"Store vertex positions for every polyhedron"});
    ADD_SLOT(Domain , domain, INPUT , REQUIRED );
    ADD_SLOT(InnerBondParams, config, INPUT, OPTIONAL);        // can be re-used for to dump contact network
    ADD_SLOT(MultiMatParamsT<InnerBondParams>, multimat_ibp, INPUT, OPTIONAL, DocString{"List of inner bond parameters for simulations with multiple materials"});

    ADD_SLOT(double, dt, INPUT, REQUIRED);
    ADD_SLOT(Classifier, ic, INPUT_OUTPUT, DocString{"Interaction lists classified according to their types"});
    ADD_SLOT(shapes, shapes_collection, INPUT_OUTPUT, DocString{"Collection of shapes"});
    // analyses
    ADD_SLOT(long, timestep, INPUT, REQUIRED);
    ADD_SLOT(std::string, dir_name, INPUT, REQUIRED, DocString{"Output directory name."});
    ADD_SLOT(std::string, interaction_basename, INPUT, REQUIRED, DocString{"Write an Output file containing interactions."});
    // private
    ADD_SLOT(bool, print_warning, PRIVATE, true, DocString{"This variable is used to ensure that warning messages are displayed only once."});
    // output
    ADD_SLOT(double, max_kn, INPUT_OUTPUT, 0, DocString{"Get the highest value of the input contact force parameters kn (used for dt_critical)"});

    public:

    inline std::string operator_name() { return "stick_polyhedron"; }

    inline std::string documentation() const override final { 
      return R"EOF(
        This operator computes forces between sticked particles using the contact law.
 
        Note that to use multmaterials version, you need to predefine the parameters with multimat_contact_params.  

        YAML example:

					one_parameter:
						- contact_polyhedron:
							 config: { kn: 10000, kt: 10000, kr: 0.1, mu: 0.1, damp_rate: 0.9}

					multi_parameters:
						- contact_polyhedron_multimat
      )EOF"; 
    }

    template<int start, int end, template<int, typename> typename FuncT, typename XFormT, typename... Args>
      void loop_contact_force(Classifier& classifier, XFormT& cp_xform, Args &&... args)
      {
        FuncT<start, XFormT> contact_law;
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
      if(multimat_ibp.has_value())
      {
        auto get_max_kn = [&kn] (const InnerBondParams& cp) -> void { kn = std::max(kn, cp.kn); };
        multimat_ibp->apply(get_max_kn);
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

      /** Get vertices and particles data */
      const auto cells = grid->cells();
      auto* vertex_fields = cvf->data();

      /** Get Shape */
      const shape *const shps = shapes_collection->data();

      /** deform matrice */
      const Mat3d& xform = domain->xform();
      bool is_def_xform = !domain->xform_is_identity();

      const double time = *dt;
      auto &classifier = *ic;

#     define __params__ cells, vertex_fields, cp, shps, time

      if constexpr (!multimat) /** Single mat */
      {
        const InnerBondParams& ibp = *config;

        const SingleMatContactParamsTAccessor<InnerBondParams> cp = {ibp};

        if( is_def_xform ) 
        {
          stick_law<LinearXForm> func;
          func.xform = LinearXForm{xform};
          run_contact_law<InteractionTypeId::StickedParticles>(parallel_execution_context(), classifier, func, __params__);
        }
        else
        {
          stick_law<NullXForm> func;
          func.xform = NullXForm{};
          run_contact_law<InteractionTypeId::StickedParticles>(parallel_execution_context(), classifier, func, __params__);
        }
      }
      else /** Multi materials */
      {
        const auto& contact_parameters = *multimat_ibp;
        const MultiMatContactParamsTAccessor<InnerBondParams> cp = contact_parameters.get_multimat_accessor();

        if( is_def_xform ) 
        {
          stick_law<LinearXForm> func;
          func.xform = LinearXForm{xform};
          run_contact_law<InteractionTypeId::StickedParticles>(parallel_execution_context(), classifier, func, __params__);
        }
        else
        {
          stick_law<NullXForm> func;
          func.xform = NullXForm{};
          run_contact_law<InteractionTypeId::StickedParticles>(parallel_execution_context(), classifier, func, __params__);
        }
      }

#undef __params__
    }

    void check_slots()
    {
      bool pw = *print_warning;

      /** Some check mutlimat versus singlemat */
      if constexpr  (multimat) /** Multiple materials */
      {
        if( !multimat_ibp.has_value() )
        {
          std::string msg = "You are using the multi-material contact force model, but the contact law parameters have not been defined.\n";
          msg +=            "Please specify the parameter values for each material pair using the operator \"multimat_inner_bond_params\".";
          color_log::error(operator_name(), msg);
        }
        if( (*print_warning) && config.has_value() )
        {
          color_log::warning(operator_name(), "You are using the multi-material contact force operator, but you have also defined the input slot \"config\" which is intended for the single-material version. This slot will be ignored.");
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
        if( (*print_warning) && multimat_ibp.has_value() )
        {
          std::string msg = "You have defined a list of contact law parameters for different material types,\n";
          msg            += "but you are using the version that only considers the parameter defined in the \"config\" input slot. \n";
          msg            += "The parameter list will be ignored. If you want to use it, please use the operator \n";
          msg            += "\"contact_sphere_multimat\"";
          color_log::warning(operator_name(), msg);
          pw = false;
        }
      }
      *print_warning = pw;
    }
  };

  template <class GridT> using ComputeStickForceSingleMatTmpl = ComputeStickForce<false, GridT>;
  template <class GridT> using ComputeStickForceMultiMatTmpl  = ComputeStickForce< true, GridT>;

  // === register factories ===
  ONIKA_AUTORUN_INIT(stick_force_polyhedron) { OperatorNodeFactory::instance()->register_factory("stick_polyhedron", make_grid_variant_operator<ComputeStickForceSingleMatTmpl>); }
  ONIKA_AUTORUN_INIT(stick_force_polyhedron_sm) { OperatorNodeFactory::instance()->register_factory("stick_polyhedron_singlemat", make_grid_variant_operator<ComputeStickForceSingleMatTmpl>); }
  ONIKA_AUTORUN_INIT(stick_force_polyhedron_mm) { OperatorNodeFactory::instance()->register_factory("stick_polyhedron_multimat", make_grid_variant_operator<ComputeStickForceMultiMatTmpl>); }
} // namespace exaDEM
