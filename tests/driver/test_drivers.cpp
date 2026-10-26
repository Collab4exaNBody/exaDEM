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
#include "driver_test.hpp"

using namespace exaDEM;
using exanb::Vec3d;

namespace {
Ball make_ball(double radius) {
  BallFields fields;
  fields.radius_ = radius;
  fields.center_ = {0, 0, 0};
  MotionType motion = STATIONARY;
  return Ball(fields, motion);
}

Surface make_surface(double offset) {
  Surface surf;
  surf.fields_.offset_ = offset;
  surf.motion_type_ = STATIONARY;
  return surf;
}

Cylinder make_cylinder(double radius) {
  Cylinder cyl;
  cyl.fields_.radius_ = radius;
  cyl.motion_type_ = STATIONARY;
  return cyl;
}

// [0] surface, [1] ball, [2] cylinder, [3] ball
Drivers make_drivers() {
  Drivers drvs;
  Driver_params motion;
  Surface s = make_surface(1.0);
  Ball b1 = make_ball(2.0);
  Cylinder c = make_cylinder(3.0);
  Ball b2 = make_ball(4.0);
  drvs.add_driver(0, s, motion);
  drvs.add_driver(1, b1, motion);
  drvs.add_driver(2, c, motion);
  motion.const_vel_ = 7.0;  // each driver keeps its own motion parameters
  drvs.add_driver(3, b2, motion);
  return drvs;
}

// Returns a value depending on the driver type, to check the dispatch of Drivers::apply.
struct RadiusFunctor {
  double operator()(Ball& b) const { return b.fields_.radius_; }
  double operator()(Cylinder& c) const { return -c.fields_.radius_; }
  double operator()(Surface& s) const { return 100 + s.fields_.offset_; }
  double operator()(RShapeDriver&) const { return 1000; }
};
}  // namespace

UNIT_TEST(empty) {
  Drivers drvs;
  CHECK(drvs.get_size() == 0);
  CHECK(drvs.well_defined());
}

UNIT_TEST(add_driver) {
  Drivers drvs = make_drivers();
  CHECK(drvs.get_size() == 4);
  CHECK(drvs.type(0) == DRIVER_TYPE::SURFACE);
  CHECK(drvs.type(1) == DRIVER_TYPE::BALL);
  CHECK(drvs.type(2) == DRIVER_TYPE::CYLINDER);
  CHECK(drvs.type(3) == DRIVER_TYPE::BALL);
  CHECK(drvs.well_defined());
  // drivers are stored by type
  CHECK(drvs.get_driver_vec<DRIVER_TYPE::BALL>().size() == 2);
  CHECK(drvs.get_driver_vec<DRIVER_TYPE::SURFACE>().size() == 1);
  CHECK(drvs.get_driver_vec<DRIVER_TYPE::CYLINDER>().size() == 1);
  CHECK(drvs.get_driver_vec<DRIVER_TYPE::RSHAPE>().size() == 0);
}

UNIT_TEST(typed_access) {
  Drivers drvs = make_drivers();
  CHECK_NEAR(drvs.get_typed_driver<Ball>(1).fields_.radius_, 2.0, 0.0);
  CHECK_NEAR(drvs.get_typed_driver<Ball>(3).fields_.radius_, 4.0, 0.0);
  CHECK_NEAR(drvs.get_typed_driver<Cylinder>(2).fields_.radius_, 3.0, 0.0);
  CHECK_NEAR(drvs.get_typed_driver<Surface>(0).fields_.offset_, 1.0, 0.0);

  // access by reference
  drvs.get_typed_driver<Ball>(3).fields_.center_ = {1, 2, 3};
  CHECK_VEC3_NEAR(drvs.get_typed_driver<Ball>(3).position(), (Vec3d{1, 2, 3}), 0.0);

  const Drivers& cdrvs = drvs;
  CHECK_NEAR(cdrvs.get_typed_driver<Ball>(1).fields_.radius_, 2.0, 0.0);
}

UNIT_TEST(motion_parameters) {
  Drivers drvs = make_drivers();
  CHECK_NEAR(drvs.get_motion(1).const_vel_, 0.0, 0.0);
  CHECK_NEAR(drvs.get_motion(3).const_vel_, 7.0, 0.0);
}

UNIT_TEST(apply) {
  Drivers drvs = make_drivers();
  RadiusFunctor func;
  CHECK_NEAR(drvs.apply(0, func), 101.0, 0.0);
  CHECK_NEAR(drvs.apply(1, func), 2.0, 0.0);
  CHECK_NEAR(drvs.apply(2, func), -3.0, 0.0);
  CHECK_NEAR(drvs.apply(3, func), 4.0, 0.0);

  // a mutable functor modifies the driver
  auto grow = [](auto& drv) {
    if constexpr (std::is_same_v<std::decay_t<decltype(drv)>, Ball>) {
      drv.fields_.radius_ *= 2;
    }
  };
  drvs.apply(3, grow);
  CHECK_NEAR(drvs.get_typed_driver<Ball>(3).fields_.radius_, 8.0, 0.0);
}

UNIT_TEST(gap_is_not_well_defined) {
  Drivers drvs;
  Driver_params motion;
  Ball b = make_ball(1.0);
  drvs.add_driver(2, b, motion);  // indexes 0 and 1 are left undefined
  CHECK(drvs.get_size() == 3);
  CHECK(drvs.type(0) == DRIVER_TYPE::UNDEFINED);
  CHECK(drvs.type(2) == DRIVER_TYPE::BALL);
  CHECK(!drvs.well_defined());
  Surface s = make_surface(0.0);
  Cylinder c = make_cylinder(1.0);
  drvs.add_driver(0, s, motion);
  drvs.add_driver(1, c, motion);
  CHECK(drvs.well_defined());
}

UNIT_TEST(replace_driver) {
  Drivers drvs = make_drivers();
  Driver_params motion;
  Cylinder c = make_cylinder(9.0);
  drvs.add_driver(1, c, motion);  // the ball at index 1 is replaced by a cylinder
  CHECK(drvs.get_size() == 4);
  CHECK(drvs.type(1) == DRIVER_TYPE::CYLINDER);
  CHECK_NEAR(drvs.get_typed_driver<Cylinder>(1).fields_.radius_, 9.0, 0.0);
  CHECK_NEAR(drvs.get_typed_driver<Ball>(3).fields_.radius_, 4.0, 0.0);  // other drivers are unchanged
}

UNIT_TEST(clear) {
  Drivers drvs = make_drivers();
  drvs.clear();
  CHECK(drvs.get_size() == 0);
  CHECK(drvs.get_driver_vec<DRIVER_TYPE::BALL>().size() == 0);
  CHECK(drvs.motion_.size() == 0);
}

UNIT_TEST(gpu_accessor) {
  Drivers drvs = make_drivers();
  DriversGPUAccessor acc(drvs);
  CHECK(acc.nb_drivers_ == 4);
  CHECK_NEAR(acc.get_typed_driver<Ball>(3).fields_.radius_, 4.0, 0.0);
  CHECK_NEAR(acc.get_typed_driver<Surface>(0).fields_.offset_, 1.0, 0.0);
  acc.get_typed_driver<Cylinder>(2).fields_.radius_ = 5.0;  // the accessor points to the drivers data
  CHECK_NEAR(drvs.get_typed_driver<Cylinder>(2).fields_.radius_, 5.0, 0.0);
}

UNIT_TEST_MAIN()
