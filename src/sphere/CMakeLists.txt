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
set(exadem_sphere_LINK_LIBRARIES exanbGridCellParticles exadem_shape exadem_driver exadem_force_field exadem_grid exadem_interaction)

set(CONTACT_LAWS
    hooke
    hertz
)

set(CONTACT_MODELS
    none
    cohesive
    dmt
)

set(CONTACT_MATS
    singlemat
    multimat
)

set(GENERATED_CU_FILES)

foreach(law ${CONTACT_LAWS})
  foreach(model ${CONTACT_MODELS})
    foreach(mat ${CONTACT_MATS})
      # the Hertz law only has a multimat version (materials defined per group)
      if(law STREQUAL "hertz" AND mat STREQUAL "singlemat")
        continue()
      endif()

      # =========================
      # Template type selection
      # CC[SM|MM][SH|SHertz][None|Cohesive|Dmt]Tmpl, see contact_force_sphere_[hooke|hertz].cu.in
      # =========================
      if(mat STREQUAL "singlemat")
        set(MAT_TAG "SM")
        set(PREFIX "contact_singlemat_sphere")
      else() # multimat
        set(MAT_TAG "MM")
        set(PREFIX "contact_multimat_sphere")
      endif()

      if(law STREQUAL "hooke")
        set(LAW_TAG "SH")
      else() # hertz
        set(LAW_TAG "SHertz")
      endif()

      if(model STREQUAL "none")
        set(MODEL_TAG "None")
      elseif(model STREQUAL "cohesive")
        set(MODEL_TAG "Cohesive")
      else() # dmt
        set(MODEL_TAG "Dmt")
      endif()

      set(TEMPLATE_TYPE "CC${MAT_TAG}${LAW_TAG}${MODEL_TAG}Tmpl")

      # =========================
      # Generated registrations
      # =========================
      if(law STREQUAL "hertz")
        # Hertz: multimat only, named contact_sphere_hertz_[none|cohesive|dmt]
        set(REGISTRATIONS "
OperatorNodeFactory::instance()->register_factory(\"contact_sphere_hertz_${model}\",
  make_grid_variant_operator<${TEMPLATE_TYPE}>);
")
      elseif(law STREQUAL "hooke" AND model STREQUAL "none")
        # historical names of the default combinaison
        set(REGISTRATIONS "
OperatorNodeFactory::instance()->register_factory(\"${PREFIX}\",
  make_grid_variant_operator<${TEMPLATE_TYPE}>);
OperatorNodeFactory::instance()->register_factory(\"${PREFIX}_hooke\",
  make_grid_variant_operator<${TEMPLATE_TYPE}>);
OperatorNodeFactory::instance()->register_factory(\"${PREFIX}_hooke_none\",
  make_grid_variant_operator<${TEMPLATE_TYPE}>);
")
      else()
        set(REGISTRATIONS "
OperatorNodeFactory::instance()->register_factory(\"${PREFIX}_${law}_${model}\",
  make_grid_variant_operator<${TEMPLATE_TYPE}>);
")
      endif()
      set(LAW ${law})
      set(MODEL ${model})
      set(MAT ${mat})

      # =========================
      # Output file
      # =========================
      set(OUTPUT_FILE ${CMAKE_CURRENT_BINARY_DIR}/contact_force_sphere_${mat}_${law}_${model}.cu)

      message(STATUS "Generate: ${OUTPUT_FILE}")
      # one source file per contact law: contact_force_sphere_hooke.cu.in, contact_force_sphere_hertz.cu.in
      configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/contact_force_sphere_${law}.cu.in
        ${OUTPUT_FILE}
        @ONLY
      )

      list(APPEND GENERATED_CU_FILES ${OUTPUT_FILE})

    endforeach()
  endforeach()
endforeach()

## Default Combinaison
set(REGISTRATIONS "
OperatorNodeFactory::instance()->register_factory(\"contact_sphere\",
  make_grid_variant_operator<CCSMSHNoneTmpl>);
")

set(LAW "default")
set(MODEL "default")
set(MAT "combinaison")

set(OUTPUT_FILE ${CMAKE_CURRENT_BINARY_DIR}/contact_force_sphere.cu)
message(STATUS "Generate: ${OUTPUT_FILE}")
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/contact_force_sphere_hooke.cu.in
  ${OUTPUT_FILE}
  @ONLY
)

list(APPEND GENERATED_CU_FILES ${OUTPUT_FILE})

list(APPEND exadem_sphere_EXTERNAL_SRCS ${GENERATED_CU_FILES})
xnb_add_plugin(exadem_sphere ${CMAKE_CURRENT_SOURCE_DIR})
