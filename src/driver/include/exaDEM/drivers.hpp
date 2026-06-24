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

#include <onika/flat_tuple.h>

#include <exaDEM/ball.hpp>
#include <exaDEM/cylinder.hpp>
#include <exaDEM/driver_base.hpp>
#include <exaDEM/rshape.hpp>
#include <exaDEM/surface.hpp>
#include <exaDEM/undefined_driver.hpp>

namespace exaDEM {
template <typename FuncT>
struct ApplyDriverFunctorTraits {
  static constexpr bool use_motion = false;
};

struct Drivers {
  /**
   * @brief Alias template for a CUDA memory managed vector.
   * @tparam T The type of elements in the vector.
   */
  template <typename T>
  using vector_t = onika::memory::CudaMMVector<T>;

  struct DriverTypeAndIndex {
    DRIVER_TYPE type_ = DRIVER_TYPE::UNDEFINED;
    int index_ = -1;
  };

  vector_t<DriverTypeAndIndex> type_index_; /**< Vector storing the types of drivers. */
  /** just a duplicate on CPU to avoid weird copies from GPU */
  std::vector<DriverTypeAndIndex> type_index_cpu_; /**< Vector storing the types of drivers. */
  // vector_t<Driver_params> motion_; /**< Vector storing the motion drivers parameters. */
  std::vector<Driver_params> motion_; /**< Vector storing the motion data drivers. */
  onika::FlatTuple<vector_t<Cylinder>, vector_t<Surface>, vector_t<Ball>, vector_t<RShapeDriver> > data_;

  /**
   * @brief Get the size of the Drivers collection.
   * @return The size of the Drivers collection.
   */
  inline size_t get_size() const { return type_index_.size(); }

  template <size_t driver_type>
  inline const auto& get_driver_vec() const {
    static_assert(driver_type != DRIVER_TYPE::UNDEFINED);
    return data_.get_nth_const<driver_type>();
  }

  template <size_t driver_type>
  inline auto& get_driver_vec() {
    static_assert(driver_type != DRIVER_TYPE::UNDEFINED);
    return data_.get_nth<driver_type>();
  }

  template <class T>
  inline const T& get_typed_driver(const int idx) const {
    constexpr DRIVER_TYPE t = get_type<T>();
    static_assert(t != DRIVER_TYPE::UNDEFINED);
    const auto& driver_vec = data_.get_nth_const<t>();
    assert(idx >= 0 && idx < type_index_.size());
    assert(type_index_[idx].type_ == t);
    assert(type_index_[idx].index_ >= 0 && type_index_[idx].index_ < driver_vec.size());
    return driver_vec[type_index_[idx].index_];
  }

  template <class T>
  inline T& get_typed_driver(const int idx) {
    constexpr DRIVER_TYPE t = get_type<T>();
    static_assert(t != DRIVER_TYPE::UNDEFINED);
    auto& driver_vec = data_.get_nth<t>();
    assert(idx >= 0 && idx < static_cast<int>(type_index_.size()));
    assert(type_index_[idx].type_ == t);
    assert(type_index_[idx].index_ >= 0 && type_index_[idx].index_ < static_cast<int>(driver_vec.size()));
    return driver_vec[type_index_[idx].index_];
  }

  template <class FuncT>
  inline auto apply(int idx, FuncT& func) {
    assert(idx >= 0 && idx < static_cast<int>(type_index_cpu_.size()) &&
           type_index_cpu_.size() == type_index_.size());
    DRIVER_TYPE t = type_index_cpu_[idx].type_;
    assert(t != DRIVER_TYPE::UNDEFINED);
    if constexpr (ApplyDriverFunctorTraits<FuncT>::use_motion) {
      if (t == DRIVER_TYPE::CYLINDER) {
        return func(data_.get_nth<DRIVER_TYPE::CYLINDER>()[type_index_cpu_[idx].index_], motion_[idx]);
      } else if (t == DRIVER_TYPE::SURFACE) {
        return func(data_.get_nth<DRIVER_TYPE::SURFACE>()[type_index_cpu_[idx].index_], motion_[idx]);
      } else if (t == DRIVER_TYPE::BALL) {
        return func(data_.get_nth<DRIVER_TYPE::BALL>()[type_index_cpu_[idx].index_], motion_[idx]);
      } else if (t == DRIVER_TYPE::RSHAPE) {
        return func(data_.get_nth<DRIVER_TYPE::RSHAPE>()[type_index_cpu_[idx].index_], motion_[idx]);
      }
      exanb::fatal_error() << "Internal error: unsupported driver type encountered" << std::endl;
      static Cylinder tmp;
      return func(tmp, motion_[idx]);
    } else {
      if (t == DRIVER_TYPE::CYLINDER) {
        return func(data_.get_nth<DRIVER_TYPE::CYLINDER>()[type_index_cpu_[idx].index_]);
      } else if (t == DRIVER_TYPE::SURFACE) {
        return func(data_.get_nth<DRIVER_TYPE::SURFACE>()[type_index_cpu_[idx].index_]);
      } else if (t == DRIVER_TYPE::BALL) {
        return func(data_.get_nth<DRIVER_TYPE::BALL>()[type_index_cpu_[idx].index_]);
      } else if (t == DRIVER_TYPE::RSHAPE) {
        return func(data_.get_nth<DRIVER_TYPE::RSHAPE>()[type_index_cpu_[idx].index_]);
      }
      exanb::fatal_error() << "Internal error: unsupported driver type encountered" << std::endl;
      static Cylinder tmp;
      return func(tmp);
    }
  }

  template <class FuncT>
  inline auto apply(const int idx, const FuncT& func) {
    assert(idx >= 0 && idx < static_cast<int>(type_index_cpu_.size()) &&
           type_index_cpu_.size() == type_index_.size());
    DRIVER_TYPE t = type_index_cpu_[idx].type_;
    assert(t != DRIVER_TYPE::UNDEFINED);
    if (t == DRIVER_TYPE::CYLINDER) {
      return func(data_.get_nth<DRIVER_TYPE::CYLINDER>()[type_index_cpu_[idx].index_]);
    } else if (t == DRIVER_TYPE::SURFACE) {
      return func(data_.get_nth<DRIVER_TYPE::SURFACE>()[type_index_cpu_[idx].index_]);
    } else if (t == DRIVER_TYPE::BALL) {
      return func(data_.get_nth<DRIVER_TYPE::BALL>()[type_index_cpu_[idx].index_]);
    } else if (t == DRIVER_TYPE::RSHAPE) {
      return func(data_.get_nth<DRIVER_TYPE::RSHAPE>()[type_index_cpu_[idx].index_]);
    }
    exanb::fatal_error() << "Internal error: unsupported driver type encountered" << std::endl;
    static Cylinder tmp;
    return func(tmp);
  }

  /**
   * @brief Adds a driver to the Drivers collection at the specified index.
   * @tparam T The type of driver to be added.
   * @param idx The index at which to add the driver.
   * @param Driver The driver to be added.
   * @details If the specified index is beyond the current size of the
   * collection, it will resize the collection accordingly. If a driver already
   * exists at the specified index, it will be replaced. If the type of driver
   * is undefined, it will throw a static assertion error.
   */
  template <typename T>
  inline void add_driver(const int idx, T& Driver, Driver_params& motion) {
    constexpr DRIVER_TYPE t = get_type<T>();
    static_assert(t != DRIVER_TYPE::UNDEFINED);
    const int size = type_index_.size();
    if (idx < size) {  // reallocation
      DRIVER_TYPE current_type = type(idx);
      if (current_type != DRIVER_TYPE::UNDEFINED) {
        exanb::lout << "You are currently removing a driver at index " << idx << std::endl;
        Driver.print();
      }
    } else {  // allocate
      type_index_.resize(idx + 1);
      type_index_cpu_.resize(idx + 1);
      motion_.resize(idx + 1);
    }
    type_index_[idx].type_ = t;
    type_index_cpu_[idx].type_ = t;
    auto& driver_vec = get_driver_vec<t>();
    type_index_[idx].index_ = driver_vec.size();
    type_index_cpu_[idx].index_ = driver_vec.size();
    driver_vec.push_back(Driver);
    motion_[idx] = motion;
  }

  /**
   * @brief Clears the Drivers collection, removing all drivers.
   */
  void clear() {
    type_index_.clear();
    type_index_cpu_.clear();
    motion_.clear();
    data_.get_nth<DRIVER_TYPE::CYLINDER>().clear();
    data_.get_nth<DRIVER_TYPE::SURFACE>().clear();
    data_.get_nth<DRIVER_TYPE::BALL>().clear();
    data_.get_nth<DRIVER_TYPE::RSHAPE>().clear();
  }

  // Accessors

  /**
   * @brief Returns the type of driver at the specified index.
   * @param idx The index of the driver.
   * @return The type of the driver at the specified index.
   */
  ONIKA_HOST_DEVICE_FUNC
  inline DRIVER_TYPE type(size_t idx) {
    assert(idx < type_index_.size());
    return type_index_[idx].type_;
  }

  /**
   * @brief Returns the data related to the drvier motion of driver at the specified index.
   * @param idx The index of the driver.
   * @return The data related to the motion at the specified index.
   */
  inline Driver_params& get_motion(const int idx) {
    assert(idx < static_cast<int>(motion_.size()));
    assert(motion_.size() == type_index_.size());
    return motion_[idx];
  }

  /**
   * @brief Checks if all drivers in the collection are well-defined.
   * @return True if all drivers are well-defined, false otherwise.
   */
  inline bool well_defined() const {
    for (const auto& it : type_index_) {
      if (it.type_ == DRIVER_TYPE::UNDEFINED) {
        return false;
      }
    }
    return true;
  }

  /**
   * @brief Prints Drivers informations.
   */
  inline void print_drivers() const {
    for (size_t i = 0; i < this->get_size(); i++) {
      auto t = type_index_cpu_[i].type_;
      if (t != DRIVER_TYPE::UNDEFINED) {
        exanb::lout << "Driver [" << i << "]:" << std::endl;
        MotionType motion_type;
        if (t == DRIVER_TYPE::CYLINDER) {
          auto& driver = data_.get_nth_const<DRIVER_TYPE::CYLINDER>()[type_index_cpu_[i].index_];
          driver.print();
          motion_type = driver.motion_type_;
        } else if (t == DRIVER_TYPE::SURFACE) {
          auto& driver = data_.get_nth_const<DRIVER_TYPE::SURFACE>()[type_index_cpu_[i].index_];
          driver.print();
          motion_type = driver.motion_type_;
        } else if (t == DRIVER_TYPE::BALL) {
          auto& driver = data_.get_nth_const<DRIVER_TYPE::BALL>()[type_index_cpu_[i].index_];
          driver.print();
          motion_type = driver.motion_type_;
        } else if (t == DRIVER_TYPE::RSHAPE) {
          auto& driver = data_.get_nth_const<DRIVER_TYPE::RSHAPE>()[type_index_cpu_[i].index_];
          driver.print();
          motion_type = driver.motion_type_;
        } else {
          continue;
        }
        motion_[i].print_driver_params(motion_type);
      }
    }
  }

  /**
   * @brief Prints statistics about the drivers in the collection.
   * @details This function prints the total number of drivers and the count of
   * each driver type.
   */
  inline void stats_drivers() const {
    std::array<int, DRIVER_TYPE_SIZE> Count;  // defined in driver_base.h
    for (auto& it : Count) {
      it = 0;
    }
    for (const auto& it : type_index_cpu_) {
      ++Count[it.type_];
    }
    exanb::lout << "Drivers Stats" << std::endl;
    exanb::lout << "Number of drivers: " << type_index_cpu_.size() << std::endl;
    for (size_t t = 0; t < DRIVER_TYPE_SIZE; t++) {
      exanb::lout << "Number of " << print(DRIVER_TYPE(t)) << "s: " << Count[t] << std::endl;
    }
  }
};

// read only proxy for drivers list
struct DriversGPUAccessor {
  size_t nb_drivers_ = 0;
  Drivers::DriverTypeAndIndex* const __restrict__ type_index_ = nullptr;
  onika::FlatTuple<Cylinder* __restrict__, Surface* __restrict__, Ball* __restrict__, RShapeDriver* __restrict__>
      data_ = {nullptr, nullptr, nullptr, nullptr};
  onika::FlatTuple<size_t, size_t, size_t, size_t> data_size_ = {0, 0, 0, 0};

  DriversGPUAccessor() = default;
  DriversGPUAccessor(const DriversGPUAccessor&) = default;
  DriversGPUAccessor(DriversGPUAccessor&&) = default;
  inline DriversGPUAccessor(Drivers& drvs)
      : nb_drivers_(drvs.type_index_.size()),
        type_index_(drvs.type_index_.data()),
        data_({drvs.data_.get_nth<0>().data(), drvs.data_.get_nth<1>().data(), drvs.data_.get_nth<2>().data(),
                drvs.data_.get_nth<3>().data()}),
        data_size_({drvs.data_.get_nth<0>().size(), drvs.data_.get_nth<1>().size(), drvs.data_.get_nth<2>().size(),
                     drvs.data_.get_nth<3>().size()}) {}

  template <class T>
  ONIKA_HOST_DEVICE_FUNC inline T& get_typed_driver(const int idx) const {
    constexpr DRIVER_TYPE t = get_type<T>();
    static_assert(t != DRIVER_TYPE::UNDEFINED);
    auto* __restrict__ driver_vec = data_.get_nth_const<t>();
    [[maybe_unused]] const size_t driver_vec_size = data_size_.get_nth_const<t>();
    assert(idx >= 0 && idx < static_cast<int>(nb_drivers_));
    assert(type_index_[idx].type_ == t);
    assert(type_index_[idx].index_ >= 0 && type_index_[idx].index_ < static_cast<int>(driver_vec_size));
    return driver_vec[type_index_[idx].index_];
  }
};
}  // namespace exaDEM
