# licensed to the apache software foundation (asf) under one
# or more contributor license agreements.  see the notice file
# distributed with this work for additional information
# regarding copyright ownership.  the asf licenses this file
# to you under the apache license, version 2.0 (the
# "license"); you may not use this file except in compliance
# with the license.  you may obtain a copy of the license at
# 
#   http://www.apache.org/licenses/license-2.0
# 
# unless required by applicable law or agreed to in writing,
# software distributed under the license is distributed on an
# "as is" basis, without warranties or conditions of any
# kind, either express or implied.  see the license for the
# specific language governing permissions and limitations
# under the license.
set(exadem_polyhedron_LINK_LIBRARIES exanbCompute exanbGridCellParticles exanbExtraStorage exadem_driver exadem_shape exadem_force_field exadem_grid exadem_interaction)

set(CONTACT_MODELS
    NONE
    COHESIVE
    DMT
)

set(CONTACT_MATS
    SINGLE
    MULTI
)

set(GENERATED_CU_FILES)

foreach(model ${CONTACT_MODELS})
  foreach(mat ${CONTACT_MATS})

    # =========================
    # Template type selection
    # =========================
    if(mat STREQUAL "SINGLE")
      if(model STREQUAL "NONE")
        set(TEMPLATE_TYPE "CCSMPHNoneTmpl")
      elseif(model STREQUAL "COHESIVE")
        set(TEMPLATE_TYPE "CCSMPHCohesiveTmpl")
      elseif(model STREQUAL "DMT")
        set(TEMPLATE_TYPE "CCSMPHDmtTmpl")
      endif()
      set(PREFIX "contact_singlemat_polyhedron")
    else() # MULTI
      if(model STREQUAL "NONE")
        set(TEMPLATE_TYPE "CCMMPHNoneTmpl")
      elseif(model STREQUAL "COHESIVE")
        set(TEMPLATE_TYPE "CCMMPHCohesiveTmpl")
      elseif(model STREQUAL "DMT")
        set(TEMPLATE_TYPE "CCMMPHDmtTmpl")
      endif()
      set(PREFIX "contact_multimat_polyhedron")
    endif()

    # =========================
    # Generated registrations
    # =========================

   if(model STREQUAL "NONE")
    set(REGISTRATIONS "
OperatorNodeFactory::instance()->register_factory(\"${PREFIX}\",
  make_grid_variant_operator<${TEMPLATE_TYPE}>);
OperatorNodeFactory::instance()->register_factory(\"${PREFIX}_hooke\",
  make_grid_variant_operator<${TEMPLATE_TYPE}>);
	OperatorNodeFactory::instance()->register_factory(\"${PREFIX}_hooke_none\",
  make_grid_variant_operator<${TEMPLATE_TYPE}>);
")

		elseif(model STREQUAL "COHESIVE")
      set(REGISTRATIONS "
OperatorNodeFactory::instance()->register_factory(\"${PREFIX}_hooke_cohesive\",
  make_grid_variant_operator<${TEMPLATE_TYPE}>);
")
    elseif(model STREQUAL "DMT")
      set(REGISTRATIONS "
OperatorNodeFactory::instance()->register_factory(\"${PREFIX}_hooke_dmt\",
  make_grid_variant_operator<${TEMPLATE_TYPE}>);
")
    endif()
    set(MODEL ${model})
    set(MAT ${mat})

    # =========================
    # Output file
    # =========================
    set(OUTPUT_FILE ${CMAKE_CURRENT_BINARY_DIR}/contact_force_${mat}_${model}.cu)

		message(STATUS "Generate: ${OUTPUT_FILE}")
    configure_file(
      ${CMAKE_CURRENT_SOURCE_DIR}/contact_force_polyhedron.cu.in
      ${OUTPUT_FILE}
      @ONLY
    )

    list(APPEND GENERATED_CU_FILES ${OUTPUT_FILE})

  endforeach()
endforeach()

list(APPEND exadem_polyhedron_EXTERNAL_SRCS ${GENERATED_CU_FILES})
xnb_add_plugin(exadem_polyhedron ${CMAKE_CURRENT_SOURCE_DIR})
