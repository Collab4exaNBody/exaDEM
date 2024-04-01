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
#include <exaDEM/driver_base.h>

namespace exaDEM
{
  using namespace exanb;

  struct UndefinedDriver
  {
    /**
     * @brief Get the type of the driver (in this case, UNDEFINED).
     * @return The type of the driver.
     */
    constexpr DRIVER_TYPE get_type() {return DRIVER_TYPE::UNDEFINED;}

    /**
     * @brief Print information about the undefined driver.
     */
    void print()
    {
      std::cout << "Driver Type: UNDEFINED" << std::endl;
    }
  };
}
