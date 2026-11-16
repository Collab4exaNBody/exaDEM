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

#include <onika/math/basic_types_def.h>
#include <onika/math/basic_types_operators.h>
#include <onika/math/quaternion.h>

namespace exaDEM {
// basic types and math functions imported from exanb (onika::math)
using ::exanb::AABB;
using ::exanb::cross;
using ::exanb::dot;
using ::exanb::IJK;
using ::exanb::Mat3d;
using ::exanb::norm;
using ::exanb::Quaternion;
using ::exanb::Vec3d;
}  // namespace exaDEM
