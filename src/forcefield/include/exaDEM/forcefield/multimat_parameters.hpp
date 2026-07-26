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

#pragma once

#include <exanb/core/particle_type_id.h>

#include <exaDEM/color_log.hpp>
#include <exaDEM/forcefield/contact_force_accessor.hpp>

namespace exaDEM {
template <typename T>
using vector_t = onika::memory::CudaMMVector<T>;
using ReverseParticleTypeMap = std::map<uint64_t, std::string>;

/**
 * @brief Stores contact parameters for multiple material pairs and driver types.
 *
 * @tparam ContactParamsT The type used to represent contact parameters (e.g., ContactParams).
 */
template <typename ContactParamsT>
struct MultiMatParamsT {
  vector_t<ContactParamsT> multimat_cp_;       //< Contact parameters between material pairs
  vector_t<ContactParamsT> drivers_cp_;        //< Contact parameters between drivers and materials
  ParticleTypeMap type_map_;                   //< Maps particle types to material indices
  ParticleTypeMap driver_map_;                 //< Maps driver types to driver indices
  ReverseParticleTypeMap reverse_type_map_;    //< Maps material indices back to particle type names
  ReverseParticleTypeMap reverse_driver_map_;  //< Maps driver indices back to driver type names
  int number_of_groups_;                       //< Number of unique groups in the simulation
  int number_of_drivers_;                      //< Number of unique driver types in the simulation

  /**
   * @brief Returns the number of groups.
   */
  int size_groups() {
    assert(static_cast<int>(multimat_cp_.size()) >= number_of_groups_ * number_of_groups_);
    return number_of_groups_;
  }

  /**
   * @brief Returns the number of driver types.
   */
  int size_drivers() {
    assert(driver_map_.size() == reverse_driver_map_.size());
    assert(static_cast<int>(driver_map_.size()) >= number_of_drivers_);
    return number_of_drivers_;
  }

  /**
   * @brief Returns an accessor for material-to-material contact parameters.
   * @return Accessor to the material-to-material contact parameters.
   */
  MultiMatContactParamsTAccessor<ContactParamsT> get_multimat_accessor() const {
    MultiMatContactParamsTAccessor<ContactParamsT> res = {multimat_cp_.data(), number_of_groups_};
    return res;
  }

  /**
   * @brief Returns an accessor for driver-to-material contact parameters.
   * @return Accessor to the driver-to-material contact parameters.
   */
  MultiMatContactParamsTAccessor<ContactParamsT> get_drivers_accessor() const {
    MultiMatContactParamsTAccessor<ContactParamsT> res = {drivers_cp_.data(), number_of_drivers_};
    return res;
  }

  /**
   * @brief Applies a function to each element of `multimat_cp_` and `drivers_cp_`.
   *
   * This generic method accepts a functor (or lambda) and applies it
   * to all elements contained in `multimat_cp_` followed by those in `drivers_cp_`.
   *
   * @tparam Func The type of the functor or lambda to be applied.
   * @param func A reference to the functor or lambda to apply to each element.
   */
  template <typename Func>
  void apply(Func& func) {
    for (size_t i = 0; i < multimat_cp_.size(); i++) func(multimat_cp_[i]);
    for (size_t i = 0; i < drivers_cp_.size(); i++) func(drivers_cp_[i]);
  }

  /**
   * @brief Initializes the material-to-material contact parameters.
   *
   * Sets up the type maps (`type_map_` and `reverse_type_map_`) and allocates the
   * array of contact parameters (`multimat_cp_`) based on the provided input map.
   * Initializes the contact parameters for each material pair.
   *
   * @param input_map A map of particle types to material indices.
   * @param cp Initial contact parameters for each material pair (default is a default-constructed object).
   */
  void setup_multimat(int n_groups, ContactParamsT cp = {}) {
    number_of_groups_ = n_groups;
    multimat_cp_.resize(number_of_groups_ * number_of_groups_, cp);
  }

  void setup_multimat(const ParticleTypeMap& input_map, ContactParamsT cp = {}) {
    type_map_ = input_map;
    for (auto it : type_map_) {
      reverse_type_map_[it.second] = it.first;
    }
    setup_multimat(static_cast<int>(input_map.size()), cp);
  }

  /**
   * @brief Initializes the driver-to-material contact parameters.
   *
   * Sets up the type maps (`driver_map_` and `reverse_driver_map_`) and allocates the
   * array of contact parameters (`drivers_cp_`) based on the provided input map.
   * Initializes the contact parameters for interactions between drivers and materials.
   *
   * @param input_map A map of particle types to driver indices.
   * @param cp Initial contact parameters for each driver-material interaction (default is a default-constructed
   * object).
   */
  void setup_drivers(const ParticleTypeMap& input_map, ContactParamsT cp = {}) {
    driver_map_ = input_map;
    for (auto it : driver_map_) {
      reverse_driver_map_[it.second] = it.first;
    }
    number_of_drivers_ = input_map.size();
    assert(number_of_groups_ > 0);
    drivers_cp_.resize(number_of_drivers_ * number_of_groups_, cp);
  }

  /**
   * @brief Returns the index for material-to-material contact parameters.
   *
   * @param mat1 The index of the first material.
   * @param mat2 The index of the second material.
   * @return The calculated index for the material-to-material contact parameter.
   */
  ONIKA_HOST_DEVICE_FUNC inline int get_idx_multimat(int group1, int group2) {
    assert(group1 < number_of_groups_);
    assert(group2 < number_of_groups_);
    return group1 * number_of_groups_ + group2;
  }

  /**
   * @brief Returns the index for material-to-driver contact parameters.
   *
   * Computes the linear index in the contact parameter array for the interaction between a material and a driver.
   * Ensures the material and driver indices are valid before performing the calculation.
   *
   * @param mat The index of the material.
   * @param driver The index of the driver.
   * @return The calculated index for the driver-to-material contact parameter.
   */
  ONIKA_HOST_DEVICE_FUNC inline int get_idx_drivers(int group, int driver) {
    assert(group < number_of_groups_);
    assert(driver < number_of_drivers_);
    return group * number_of_drivers_ + driver;
  }

  /**
   * @brief Registers the contact parameters for a material-to-material interaction.
   *
   * Sets the contact parameters for both directions of the material-to-material interaction
   * (mat1, mat2) and (mat2, mat1), ensuring the contact parameters are symmetric.
   *
   * @param mat1 The index of the first material.
   * @param mat2 The index of the second material.
   * @param cp The contact parameters to be registered for the interaction.
   */
  void register_multimat(int group1, int group2, ContactParamsT& cp) {
    int id = get_idx_multimat(group1, group2);
    multimat_cp_[id] = cp;
    id = get_idx_multimat(group2, group1);
    multimat_cp_[id] = cp;
  }

  /**
   * @brief Registers the contact parameters for a driver-to-material interaction.
   *
   * Sets the contact parameters for the interaction between a material and a driver.
   *
   * @param mat The index of the material.
   * @param driver The index of the driver.
   * @param cp The contact parameters to be registered for the interaction.
   */
  void register_driver(int group, int driver, ContactParamsT& cp) {
    int id = get_idx_drivers(group, driver);
    drivers_cp_[id] = cp;
  }

  /**
   * @brief Checks the completeness of the contact parameters for materials and drivers.
   *
   * This function verifies that the contact parameters for all material-to-material and
   * driver-to-material interactions are defined (i.e., not equal to the default contact parameters).
   * If any contact parameters are missing, a warning is printed to the output.
   *
   * @param check_multimat If true, checks the completeness of material-to-material contact parameters.
   * @param check_driver If true, checks the completeness of driver-to-material contact parameters.
   * @return `true` if all contact parameters are defined, `false` otherwise.
   */
  bool check_completeness(bool check_multimat = true, bool check_driver = true) {
    bool check = true;
    ContactParamsT default_cp = {};
    for (int g1 = 0; g1 < number_of_groups_; g1++) {
      if (check_multimat) {
        // Check group-to-group parameters
        for (int g2 = 0; g2 <= g1; g2++) /** symmetric definition */
        {
          if (multimat_cp_[this->get_idx_multimat(g1, g2)] == default_cp) {
            color_log::warning("MultiMatParamsT::check_completeness",
                               "Contact force parameters for the pair (group: " + std::to_string(g1) +
                                   ", group: " + std::to_string(g2) + ") are not defined.");
            check = false;
          }
        }
      }
      // Check driver-to-group parameters
      if (check_driver) {
        for (int drv = 0; drv < number_of_drivers_; drv++) {
          if (drivers_cp_[this->get_idx_drivers(g1, drv)] == default_cp) {
            color_log::warning("MultiMatParamsT::check_completeness",
                               "Contact force parameters for the pair (group: " + std::to_string(g1) +
                                   ", driver: " + std::to_string(drv) + ") are not defined.");
            check = false;
          }
        }
      }
    }
    return check;
  }

  /**
   * @brief Displays the contact parameters for materials and drivers.
   *
   * Depending on the flags `multimat_mode` and `drivers_mode`, this function displays
   * the contact parameters for material-to-material and/or driver-to-material interactions.
   * It prints a formatted table with the contact parameters for the specified modes.
   *
   * @param multimat_mode If true, displays material-to-material contact parameters.
   * @param drivers_mode If true, displays driver-to-material contact parameters.
   */
  void display(bool multimat_mode = true, bool drivers_mode = true) {
    auto group_name = [this](int g) -> std::string {
      auto it = reverse_type_map_.find(g);
      return it != reverse_type_map_.end() ? it->second : "group_" + std::to_string(g);
    };
    display_header<ContactParamsT>();

    // Loop through all groups
    for (int g1 = 0; g1 < number_of_groups_; g1++) {
      // Display group-to-group parameters
      for (int g2 = 0; g2 <= g1; g2++) /** symmetric definition */
      {
        ContactParamsT& cp = multimat_cp_[this->get_idx_multimat(g1, g2)];
        display_multimat(group_name(g1), group_name(g2), cp);
      }

      // Display driver-to-group parameters
      if (drivers_mode) {
        for (int drv = 0; drv < number_of_drivers_; drv++) {
          display_multimat(group_name(g1), reverse_driver_map_[drv], drivers_cp_[this->get_idx_drivers(g1, drv)]);
        }
      }
    }
    display_end_table<ContactParamsT>();
  }

  /**
   * @brief Retrieves the contact parameters for a material-to-material interaction.
   *
   * @param mat1 The index of the first material.
   * @param mat2 The index of the second material.
   * @return A reference to the contact parameters for the material-to-material interaction.
   */
  ONIKA_HOST_DEVICE_FUNC inline ContactParamsT& get_multimat_cp(int group1, int group2) const {
    return multimat_cp_[this->get_idx_multimat(group1, group2)];
  }

  /**
   * @brief Retrieves the contact parameters for a driver-to-group interaction.
   *
   * @param group The group index of the particle.
   * @param driver The index of the driver.
   * @return A reference to the contact parameters for the driver-to-group interaction.
   */
  ONIKA_HOST_DEVICE_FUNC inline ContactParamsT& get_drivers_cp(int group, int driver) const {
    return drivers_cp_[this->get_idx_drivers(group, driver)];
  }
};
}  // namespace exaDEM
