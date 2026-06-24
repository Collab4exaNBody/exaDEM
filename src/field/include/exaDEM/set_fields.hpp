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

// onika
#include <onika/flat_tuple.h>
#include <onika/math/basic_types.h>

// exaNBody
#include <exanb/compute/compute_cell_particles.h>
#include <exanb/grid_cell_particles/particle_region.h>

namespace exaDEM {

/**
 * @brief Sets the value of an argument using a tuple of values.
 * @tparam idx The index of the value to extract from the tuple.
 * @tparam Tuple The type of the tuple containing values.
 * @tparam Arg The type of the argument to set.
 * @param values The tuple containing values.
 * @param arg The argument to set.
 */
template <int idx, typename Tuple, typename Arg>
ONIKA_HOST_DEVICE_FUNC static inline void setter(const Tuple& values, Arg& arg) {
  arg = values.get(onika::tuple_index<idx>);
}

/**
 * @brief Sets the values of multiple arguments using a tuple of values.
 * @tparam idx The index of the value to extract from the tuple.
 * @tparam Arg The type of the first argument to set.
 * @tparam Tuple The type of the tuple containing values.
 * @tparam Args The types of the remaining arguments to set.
 * @param values The tuple containing values.
 * @param arg The first argument to set.
 * @param args The remaining arguments to set.
 */
template <int idx, typename Arg, typename Tuple, typename... Args>
ONIKA_HOST_DEVICE_FUNC static inline void setter(const Tuple& values, Arg& arg, Args&... args) {
  arg = values.get(onika::tuple_index<idx>);
  setter<idx + 1>(values, args...);
}

/**
 * @brief Functor for setting values.
 *
 * This structure provides a functor for setting values of multiple types.
 * It defines an operator() that accepts arguments of any type,
 * along with a variadic pack of arguments of other types, and sets their values accordingly.
 *
 * @tparam Ts Variadic template parameter pack for types.
 */
template <typename... Ts>
struct SetFunctor {
  onika::FlatTuple<Ts...> default_values_; /**< Flat tuple of default values of types Ts. */

  /**
   * @brief Functor operator for setting values.
   * @tparam Args Variadic template parameter pack for additional argument types.
   * @param rx The value for the x-coordinate.
   * @param ry The value for the y-coordinate.
   * @param rz The value for the z-coordinate.
   * @param id The identifier.
   * @param args Additional arguments to set.
   */
  template <typename... Args>
  ONIKA_HOST_DEVICE_FUNC inline void operator()(Args&... args) const {
    static constexpr int first = 0;
    setter<first>(default_values_, args...);
  }
};

/**
 * @brief Functor for setting values.
 *
 * This structure provides a functor for setting values of multiple types.
 * It defines an operator() that accepts arguments of any type,
 * along with a variadic pack of arguments of other types, and sets their values accordingly.
 *
 * @tparam Ts Variadic template parameter pack for types.
 */
template <typename... Ts>
struct SetRegionFunctor {
  const ParticleRegionCSGShallowCopy region_; /**< Shallow copy of a particle region. */
  onika::FlatTuple<Ts...> default_values_;    /**< Flat tuple of default values of types Ts. */

  /**
   * @brief Functor operator for setting values.
   * @tparam Args Variadic template parameter pack for additional argument types.
   * @param rx The value for the x-coordinate.
   * @param ry The value for the y-coordinate.
   * @param rz The value for the z-coordinate.
   * @param id The identifier.
   * @param args Additional arguments to set.
   */
  template <typename... Args>
  ONIKA_HOST_DEVICE_FUNC inline void operator()(double rx, double ry, double rz, const uint64_t id,
                                                Args&... args) const {
    Vec3d r = {rx, ry, rz};
    if (region_.contains(r, id)) {
      static constexpr int first = 0;
      setter<first>(default_values_, args...);
    }
  }
};

/**
 * @brief Structure representing a functor with processing and default values.
 * This structure combines a functor with processing, default values.
 * @tparam Func The type of the functor with processing.
 * @tparam Ts Variadic template parameter pack for types of default values.
 */
template <typename Func, typename... Ts>
struct SetFunctorWithProcessing {
  mutable Func processing_;                /**< Functor with processing. */
  onika::FlatTuple<Ts...> default_values_; /**< Flat tuple of default values of types Ts. */

  /**
   * @brief Functor operator with a processing function.
   *
   * @tparam Args Variadic template parameter pack for additional argument types.
   * @param args Additional arguments to process.
   */
  template <typename... Args>
  ONIKA_HOST_DEVICE_FUNC inline void operator()(Args&... args) const {
    static constexpr int first = 0;
    setter<first>(default_values_, processing_(args)...);
  }
};

/**
 * @brief Structure representing a functor with processing and default values, along with a particle region.
 * This structure combines a functor with processing, default values, and a particle region.
 * @tparam Func The type of the functor with processing.
 * @tparam Ts Variadic template parameter pack for types of default values.
 */
template <typename Func, typename... Ts>
struct SetRegionFunctorWithProcessing {
  const ParticleRegionCSGShallowCopy region_; /**< Shallow copy of a particle region. */
  mutable Func processing_;                   /**< Functor with processing. */
  onika::FlatTuple<Ts...> default_values_;    /**< Flat tuple of default values of types Ts. */

  /**
   * @brief Functor operator with a processing function.
   *
   * @tparam Args Variadic template parameter pack for additional argument types.
   * @param rx The value for the x-coordinate.
   * @param ry The value for the y-coordinate.
   * @param rz The value for the z-coordinate.
   * @param id The identifier.
   * @param args Additional arguments to process.
   */
  template <typename... Args>
  ONIKA_HOST_DEVICE_FUNC inline void operator()(double rx, double ry, double rz, const uint64_t id,
                                                Args&... args) const {
    Vec3d r = {rx, ry, rz};
    if (region_.contains(r, id)) {
      static constexpr int first = 0;
      setter<first>(default_values_, processing_(args)...);
    }
  }
};

/**
 * @brief structure representing a functor with a genrator.
 * @tparam func the type of the functor with generator.
 */
template <typename Func>
struct GenSetFunctor {
  mutable Func gen_; /**< functor used to generate value. */

  // this function is only used to apply gen_ on parametes, note that gen_ can't return void.
  template <typename... Args>
  void do_nothing(Args... args) const {}

  /**
   * @brief functor operator with a processing function.
   *
   * @tparam args variadic template parameter pack for additional argument types.
   * @param args additional arguments to process.
   */
  template <typename... Args>
  inline void operator()(Args&... args) const {
    do_nothing(gen_(args)...);
  }
};

/**
 * @brief structure representing a functor with a genrator for a region.
 * @tparam func the type of the functor with generator.
 */
template <typename Func>
struct GenSetRegionFunctor {
  const ParticleRegionCSGShallowCopy region_; /**< Shallow copy of a particle region. */
  mutable Func gen_;                          /**< functor used to generate value. */

  // this function is only used to apply gen_ on parametes, note that gen_ can't return void.
  template <typename... Args>
  void do_nothing(Args... args) const {}

  /**
   * @brief functor operator with a processing function.
   *
   * @tparam args variadic template parameter pack for additional argument types.
   * @param args additional arguments to process.
   */
  template <typename... Args>
  inline void operator()(double rx, double ry, double rz, const uint64_t id, Args&... args) const {
    Vec3d r = {rx, ry, rz};
    if (region_.contains(r, id)) {
      do_nothing(gen_(args)...);
    }
  }
};

/**
 * @brief Structure representing a functor for filtered set operations with default values.
 * @tparam Ts Variadic template parameter pack for types of default values.
 */
template <typename... Ts>
struct FilteredSetFunctor {
  uint32_t filtered_type_;                 /**< The filtered type. */
  onika::FlatTuple<Ts...> default_values_; /**< Flat tuple of default values of types Ts. */

  /**
   * @brief Functor operator for filtered set operations.
   * @tparam Args Variadic template parameter pack for additional argument types.
   * @param type The filtered type.
   * @param args Additional arguments to process.
   */
  template <typename... Args>
  ONIKA_HOST_DEVICE_FUNC inline void operator()(uint32_t type, Args&... args) const {
    if (type == filtered_type_) {
      constexpr int first = 0;
      setter<first>(default_values_, args...);
    }
  }
};

/**
 * @brief Structure representing a functor for filtered set operations with default values and a particle region.
 * This structure combines a functor for filtered set operations, default values, and a particle region.
 * @tparam Ts Variadic template parameter pack for types of default values.
 */
template <typename... Ts>
struct FilteredSetRegionFunctor {
  const ParticleRegionCSGShallowCopy region_; /**< Shallow copy of a particle region. */
  uint32_t filtered_type_;                    /**< The filtered type. */
  onika::FlatTuple<Ts...> default_values_;    /**< Flat tuple of default values of types Ts. */

  /**
   * @brief Functor operator for filtered set operations.
   * @tparam Args Variadic template parameter pack for additional argument types.
   * @param rx The value for the x-coordinate.
   * @param ry The value for the y-coordinate.
   * @param rz The value for the z-coordinate.
   * @param id The particle identifier.
   * @param type The filtered type.
   * @param args Additional arguments to process.
   */
  template <typename... Args>
  ONIKA_HOST_DEVICE_FUNC inline void operator()(double rx, double ry, double rz, const uint64_t id, uint32_t type,
                                                Args&... args) const {
    if (type == filtered_type_) {
      Vec3d r = {rx, ry, rz};
      if (region_.contains(r, id)) {
        constexpr int first = 0;
        setter<first>(default_values_, args...);
      }
    }
  }
};
}  // namespace exaDEM

namespace exanb {
template <class... Ts>
struct ComputeCellParticlesTraits<exaDEM::SetFunctor<Ts...>> {
  static inline constexpr bool RequiresBlockSynchronousCall = false;
  static inline constexpr bool CudaCompatible = true;
};

template <class... Ts>
struct ComputeCellParticlesTraits<exaDEM::SetRegionFunctor<Ts...>> {
  static inline constexpr bool RequiresBlockSynchronousCall = false;
  static inline constexpr bool CudaCompatible = true;
};

template <class... Ts>
struct ComputeCellParticlesTraits<exaDEM::FilteredSetFunctor<Ts...>> {
  static inline constexpr bool RequiresBlockSynchronousCall = false;
  static inline constexpr bool CudaCompatible = true;
};

template <class... Ts>
struct ComputeCellParticlesTraits<exaDEM::FilteredSetRegionFunctor<Ts...>> {
  static inline constexpr bool RequiresBlockSynchronousCall = false;
  static inline constexpr bool CudaCompatible = true;
};

template <class... Ts>
struct ComputeCellParticlesTraits<exaDEM::GenSetRegionFunctor<Ts...>> {
  static inline constexpr bool RequiresBlockSynchronousCall = false;
  static inline constexpr bool CudaCompatible = false;
};

template <class... Ts>
struct ComputeCellParticlesTraits<exaDEM::GenSetFunctor<Ts...>> {
  static inline constexpr bool RequiresBlockSynchronousCall = false;
  static inline constexpr bool CudaCompatible = false;
};
}  // namespace exanb
