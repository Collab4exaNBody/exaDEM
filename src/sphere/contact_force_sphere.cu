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

#include <exaDEM/contact_force_parameters.h>
#include <exaDEM/contact_params_checker.h>
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
#include <exaDEM/multimat_cp.h>

namespace exaDEM
{
  using namespace exanb;
  using namespace sphere;

  template <bool multimat, ContactLawType ContactLaw, AdhesionLawType AdhesionLaw, typename GridT, class = AssertGridHasFields<GridT, field::_vx, field::_vy, field::_vz, field::_mom, field::_orient, field::_vrot, field::_radius>> 
    class ComputeContactClassifierSphere : public OperatorNode
  {
    // attributes processed during computation
    using ComputeFields = FieldSet<field::_vrot, field::_arot>;
    static constexpr ComputeFields compute_field_set{};

    ADD_SLOT(GridT, grid, INPUT_OUTPUT, REQUIRED);
    ADD_SLOT(Domain , domain, INPUT , REQUIRED );
    ADD_SLOT(ContactParams, config, INPUT, OPTIONAL, DocString{"Contact parameters for sphere interactions"});      // can be re-used for to dump contact network
    ADD_SLOT(ContactParams, config_driver, INPUT, OPTIONAL, DocString{"Contact parameters for drivers, optional"}); // can be re-used for to dump contact network
    ADD_SLOT(ContactParamsMultiMat<ContactParams>, multimat_cp, INPUT, DocString{"List of contact parameters for simulations with multiple materials"});
    ADD_SLOT(double, dt, INPUT, REQUIRED, DocString{"Time step value"});
    ADD_SLOT(bool, symetric, INPUT_OUTPUT, REQUIRED, DocString{"Activate the use of symetric feature (contact law)"});
    ADD_SLOT(Drivers, drivers, INPUT, REQUIRED, DocString{"List of Drivers {Cylinder, Surface, Ball, Mesh}"});
    ADD_SLOT(Classifier<InteractionSOA>, ic, INPUT_OUTPUT, DocString{"Interaction lists classified according to their types"});
    // analysis
    ADD_SLOT(long, timestep, INPUT, REQUIRED);
    ADD_SLOT(long, analysis_interaction_dump_frequency, INPUT, REQUIRED, DocString{"Write an interaction dump file"});
    ADD_SLOT(std::string, dir_name, INPUT, REQUIRED, DocString{"Output directory name."});
    ADD_SLOT(std::string, interaction_basename, INPUT, REQUIRED, DocString{"Write an Output file containing interactions."});
    // private
    ADD_SLOT(bool, print_warning, PRIVATE, true, DocString{"This variable is used to ensure that warning messages are displayed only once."});
    // output
    ADD_SLOT(double, max_kn, INPUT_OUTPUT, 0, DocString{"Get the highest value of the input contact force parameters kn (used for dt_critical)"});

    public:

    inline std::string opertor_name() { return "contact_sphere"; }

    inline std::string documentation() const override final { return 
      R"EOF(
      This operator computes forces between particles and particles/drivers using the contact law.
      Note that to use multmaterials version, you need to predefine the parameters with multimat_contact_params and drivers_contact_params.  

      Naming system :
      contact_[InputType]_[Shape]_[ContactLaw]_[AdhesionLaw]
      [InputType]   : singlemat or multimat
      [Shape]       : sphere or polyhedron
      [ContactLaw]  : Hooke or cohesive
      [AdhesionLaw] : None or DMT
      For simplicity contact_sphere <=> contact_singlemat_sphere_hooke <=> contact_singlemat_sphere_hooke_none

      YAML example:
        - contact_sphere:
           symetric: true
           config: { kn: 100000, kt: 100000, kr: 0.1, mu: 0.9, damp_rate: 0.9}
           
        - contact_singlemat_sphere_hooke:
           symetric: true
           config: { kn: 100000, kt: 100000, kr: 0.1, mu: 0.9, damp_rate: 0.9}

        - contact_singlemat_sphere_cohesive_none:
           symetric: true
           config: { dncut: 0.1 m, kn: 100000, kt: 100000, kr: 0.1, fc: 0.05, mu: 0.9, damp_rate: 0.9}

        - contact_singlemat_sphere_hooke_dmt:
           symetric: true
           config: { dncut: 0.1 m, kn: 100000, kt: 100000, kr: 0.1, mu: 0.9, damp_rate: 0.9, gamma: 0.05}

        - contact_multimat_sphere_hooke_none:
           symetric: true

        - contact_multimat_sphere_hooke_dmt:
           symetric: true

        - contact_multimat_sphere_cohesive_none:
           symetric: true
      )EOF"; 
    }

    template<int start, int end, template<int, ContactLawType, AdhesionLawType, typename> typename FuncT, typename XFormT, typename T, typename... Args>
      void loop_contact_force(Classifier<T>& classifier, XFormT& cp_xform, Args &&... args)
      {
        FuncT<start, ContactLaw, AdhesionLaw, XFormT> contact_law;
        contact_law.xform = cp_xform;
        run_contact_law(parallel_execution_context(), start, classifier, contact_law, args...);
        if constexpr( start + 1 <= end )
        {
          loop_contact_force<start+1, end, FuncT, XFormT>(classifier, cp_xform, std::forward<Args>(args)...);
        }
      }

    template<bool is_sym, typename XFormT>
      void core(XFormT& xform)
      {
        const DriversGPUAccessor drvs = *drivers;
        auto *cells = grid->cells();

        const double time = *dt;
        auto &classifier = *ic;

        contact_law<is_sym, ContactLaw, AdhesionLaw, XFormT> sph = {xform};
        contact_law_driver<ContactLaw, AdhesionLaw, Cylinder, XFormT> cyl = {xform};
        contact_law_driver<ContactLaw, AdhesionLaw, Surface, XFormT> surf = {xform};
        contact_law_driver<ContactLaw, AdhesionLaw, Ball, XFormT> ball = {xform};

        if( !multimat ) /** single mat */
        {
          const ContactParams hkp = *config;
          ContactParams hkp_drvs{};

          if (drivers->get_size() > 0 )
          {
            hkp_drvs = *config_driver;
          }

          const SingleMatContactParamsTAccessor<ContactParams> cp = {hkp};
          const SingleMatContactParamsTAccessor<ContactParams> cp_drvs = {hkp_drvs};

          run_contact_law(parallel_execution_context(), 0, classifier, sph, cells, cp, time);
          run_contact_law(parallel_execution_context(), 4, classifier, cyl, cells, drvs, cp_drvs, time);
          run_contact_law(parallel_execution_context(), 5, classifier, surf, cells, drvs, cp_drvs, time);
          run_contact_law(parallel_execution_context(), 6, classifier, ball, cells, drvs, cp_drvs, time);

          constexpr int stl_type_start = 7;
          constexpr int stl_type_end = 9;
          loop_contact_force <stl_type_start,  stl_type_end, contact_law_stl, XFormT>(classifier, xform, cells, drvs, cp_drvs, time);
        }
        else
        {
          const auto& contact_parameters = *multimat_cp;
          const MultiMatContactParamsTAccessor<ContactParams> cp = contact_parameters.get_multimat_accessor();
          const MultiMatContactParamsTAccessor<ContactParams> cp_drvs = contact_parameters.get_drivers_accessor();
          run_contact_law(parallel_execution_context(), 0, classifier, sph, cells, cp, time);
          run_contact_law(parallel_execution_context(), 4, classifier, cyl, cells, drvs, cp_drvs, time);
          run_contact_law(parallel_execution_context(), 5, classifier, surf, cells, drvs, cp_drvs, time);
          run_contact_law(parallel_execution_context(), 6, classifier, ball, cells, drvs, cp_drvs, time);

          constexpr int stl_type_start = 7;
          constexpr int stl_type_end = 9;
          loop_contact_force <stl_type_start,  stl_type_end, contact_law_stl, XFormT>(classifier, xform, cells, drvs, cp_drvs, time);
        }
      }

    void save_results()
    {
      /** Analysis */
      const long frequency_interaction = *analysis_interaction_dump_frequency;
      bool write_interactions = (frequency_interaction > 0 && (*timestep) % frequency_interaction == 0);
      if(write_interactions)
      {
        auto &classifier = *ic;
        auto stream = itools::create_buffer(*grid, classifier);
        std::string ts = std::to_string(*timestep);
        itools::write_file(stream, *dir_name, (*interaction_basename) + ts);
      }
    }

    void check_slots()
    {
      bool pw = true;

      /** polyhedron interactions are defined while the contact sphere operator is used */
      {
        auto &classifier = *ic;
        for(int i = 1 ; i <= 3 ; i++)
        {
          auto& interactions = classifier.get_wave(i);
          if(interactions.size() > 0)
          {
            std::string msg = "The contact operator for spheres is being used, but polyhedron interactions are defined.\n";
            msg += "                        Please, use contact_polyhedron operators."; 
            color_log::error(opertor_name(), msg);
          }
        }
      }

      /** Some check multimat versus singlemat */
      if constexpr  (multimat) /** Multiple materials */
      {
        if( !multimat_cp.has_value() )
        {
          std::string msg = "You are using the multi-material contact force model, but the contact law parameters have not been defined.\n";
          msg            += "Please specify the parameter values for each material pair using the operator \"multimat_contact_params\".";
          color_log::error(opertor_name(), msg);
        }
        if( *print_warning && config.has_value() ) 
        {
          color_log::warning(opertor_name(), "You are using the multi-material contact force operator, but you have also defined the input slot \"config\" which is intended for the single-material version. This slot will be ignored.");
          pw = false;
        }
        if( *print_warning && config_driver.has_value() ) 
        {
          color_log::warning(opertor_name(), "You are using the multi-material contact force operator, but you have also defined the input slot \"config_driver\" which is intended for the single-material version. This slot will be ignored.");
          pw = false;
        }
      }

      if constexpr  (!multimat) /** Single material */
      {
        if( !config.has_value() ) 
        {
          std::string msg = "The input slot \"config\" is not defined, yet the single-material version of the contact operator is being used.\n";
          msg            += "Please specify the \"config\" input slot, and use the \"config_driver\" slot if you want to define a contact law between a particle and a driver.";
          color_log::error(opertor_name(), msg);
        }
        if( multimat_cp.has_value() )
        {
          std::string msg = " You have defined a list of contact law parameters for different material types, \n";
          msg            += "but you are using the version that only considers the parameter defined in the \"config\" input slot.\n";
          msg            += "The parameter list will be ignored. If you want to use it, please use the operators types \n";
          msg            += "\"contact_multimat_[Shape]_[ContactLaw]_[AdhesionLaw]\".";
          color_log::warning(opertor_name(), msg);
          pw = false;
        }
        /** Some global checks */
        /** Is cohesive force define while it's not used */


        if constexpr (ContactLaw != ContactLawType::Cohesive)
        {
          if(config->dncut > 0)
          {
            color_log::error(opertor_name(), "dncut is != 0 while the cohesive force is not used. Please, use contact_[InputType]_[Shape]_cohesive_[AdhesionLaw] operators.");
          }
          if(drivers->get_size() > 0 && config_driver->dncut > 0)
          {
            color_log::error(opertor_name(), "dncut is != 0 while the cohesive force is not used. Please, use contact_[InputType]_[Shape]_cohesive_[AdhesionLaw] operators.");
          }
        }
      }

      std::optional<ContactParams> cfg;
      std::optional<ContactParams> cfg_driver;

      if (config.has_value()) cfg = *config;
      if (config_driver.has_value()) cfg_driver = *config_driver;

      check_contact_params( opertor_name(), cfg, cfg_driver, ContactLaw, AdhesionLaw);
        

      *print_warning = pw;
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
      check_slots();
      scan_kn();

      if (grid->number_of_cells() == 0)
      {
        return;
      }

      if(!domain->xform_is_identity())
      {
        LinearXForm cp_xform = {domain->xform()};
        if(*symetric) core<true>(cp_xform);
        else core<false>(cp_xform);
      }
      else     
      {
        NullXForm cp_xform;
        if(*symetric) core<true>(cp_xform);
        else core<false>(cp_xform);
      }     

      save_results();
    }
  };

  template <class GridT> using  ComputeContactSingleMatSphereHookeNoneTmpl    = ComputeContactClassifierSphere<false, ContactLawType::Hooke,      AdhesionLawType::None, GridT>;  
  template <class GridT> using  ComputeContactSingleMatSphereHookeDmtTmpl     = ComputeContactClassifierSphere<false, ContactLawType::Hooke,      AdhesionLawType::DMT , GridT>; 
  template <class GridT> using  ComputeContactSingleMatSphereCohesiveNoneTmpl = ComputeContactClassifierSphere<false, ContactLawType::Cohesive,   AdhesionLawType::None, GridT>; 
  template <class GridT> using  ComputeContactSingleMatSphereCohesiveDmtTmpl  = ComputeContactClassifierSphere<false, ContactLawType::Cohesive,   AdhesionLawType::DMT , GridT>; 
  template <class GridT> using  ComputeContactMultiMatSphereHookeNoneTmpl     = ComputeContactClassifierSphere<true , ContactLawType::Hooke,      AdhesionLawType::None, GridT>; 
  template <class GridT> using  ComputeContactMultiMatSphereHookeDmtTmpl      = ComputeContactClassifierSphere<true , ContactLawType::Hooke,      AdhesionLawType::DMT , GridT>; 
  template <class GridT> using  ComputeContactMultiMatSphereCohesiveNoneTmpl  = ComputeContactClassifierSphere<true , ContactLawType::Cohesive,   AdhesionLawType::None, GridT>; 
  template <class GridT> using  ComputeContactMultiMatSphereCohesiveDmtTmpl   = ComputeContactClassifierSphere<true , ContactLawType::Cohesive,   AdhesionLawType::DMT , GridT>; 

  // === register factories ===
  ONIKA_AUTORUN_INIT(contact_force_sphere) { OperatorNodeFactory::instance()->register_factory("contact_sphere", make_grid_variant_operator<ComputeContactSingleMatSphereHookeNoneTmpl>); }
  ONIKA_AUTORUN_INIT(contact_force_sphere_sm_hooke) { OperatorNodeFactory::instance()->register_factory("contact_singlemat_sphere_hooke", make_grid_variant_operator<ComputeContactSingleMatSphereHookeNoneTmpl>); }
  ONIKA_AUTORUN_INIT(contact_force_sphere_sm_hooke_none) { OperatorNodeFactory::instance()->register_factory("contact_singlemat_sphere_hooke_none", make_grid_variant_operator<ComputeContactSingleMatSphereHookeNoneTmpl>); }
  ONIKA_AUTORUN_INIT(contact_force_sphere_sm_hooke_dmt) { OperatorNodeFactory::instance()->register_factory("contact_singlemat_sphere_hooke_dmt", make_grid_variant_operator<ComputeContactSingleMatSphereHookeDmtTmpl>); }
  ONIKA_AUTORUN_INIT(contact_force_sphere_sm_cohesive_none) { OperatorNodeFactory::instance()->register_factory("contact_singlemat_sphere_cohesive_none", make_grid_variant_operator<ComputeContactSingleMatSphereCohesiveNoneTmpl>); }
  ONIKA_AUTORUN_INIT(contact_force_sphere_sm_cohesive_dmt) { OperatorNodeFactory::instance()->register_factory("contact_singlemat_sphere_cohesive_dmt", make_grid_variant_operator<ComputeContactSingleMatSphereCohesiveDmtTmpl>); }

  ONIKA_AUTORUN_INIT(contact_force_sphere_mm_hooke) { OperatorNodeFactory::instance()->register_factory("contact_multimat_sphere_hooke", make_grid_variant_operator<ComputeContactMultiMatSphereHookeNoneTmpl>); }
  ONIKA_AUTORUN_INIT(contact_force_sphere_mm_hooke_none) { OperatorNodeFactory::instance()->register_factory("contact_multimat_sphere_hooke_none", make_grid_variant_operator<ComputeContactMultiMatSphereHookeNoneTmpl>); }
  ONIKA_AUTORUN_INIT(contact_force_sphere_mm_hooke_dmt) { OperatorNodeFactory::instance()->register_factory("contact_multimat_sphere_hooke_dmt", make_grid_variant_operator<ComputeContactMultiMatSphereHookeDmtTmpl>); }
  ONIKA_AUTORUN_INIT(contact_force_sphere_mm_cohesive_none) { OperatorNodeFactory::instance()->register_factory("contact_multimat_sphere_cohesive_none", make_grid_variant_operator<ComputeContactMultiMatSphereCohesiveNoneTmpl>); }
  ONIKA_AUTORUN_INIT(contact_force_sphere_mm_cohesive_dmt) { OperatorNodeFactory::instance()->register_factory("contact_multimat_sphere_cohesive_dmt", make_grid_variant_operator<ComputeContactMultiMatSphereCohesiveDmtTmpl>); }


} // namespace exaDEM
