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

#include <yaml-cpp/yaml.h>
#include <onika/physics/units.h>
//#include <exaDEM/unit_system.h>

namespace exaDEM
{
  struct ContactParams
  {
    double dncut = 0;
    double kn;
    double kt;
    double kr;
    double fc = 0;
    double mu;
    double damp_rate;
  };
} // namespace exaDEM

// Yaml conversion operators, allows to read potential parameters from config file
namespace YAML
{
  using exaDEM::ContactParams;
  using exanb::lerr;
  using onika::physics::Quantity;

  template <> struct convert<ContactParams>
  {
    static bool decode(const Node &node, ContactParams &v)
    {
      v = ContactParams{}; // initializes defaults values
      if (!node.IsMap())
      {
        return false;
      }
      if (!node["dncut"] && !node["fc"])
      {
        v.dncut = 0.0;
        v.fc = 0.0;
      }
      else if(node["dncut"] && node["fc"])
      {
        v.dncut = node["dncut"].as<Quantity>().convert();
        v.fc = node["fc"].as<Quantity>().convert();
      }
      else if(!node["dncut"])
      {
        lerr << "dncut is missing\n";
        return false;
      }
      else if(!node["fc"])
      {
        lerr << "fc is missing\n";
        return false;
      }

      if (!node["kn"])
      {
      }
      if (!node["kt"])
      {
        lerr << "kt is missing\n";
        return false;
      }
      if (!node["kr"])
      {
        lerr << "kr is missing\n";
        return false;
      }
      if (!node["mu"])
      {
        lerr << "mu is missing\n";
        return false;
      }
      if (!node["damp_rate"])
      {
        lerr << "damp_rate is missing\n";
        return false;
      }

      v.kn = node["kn"].as<Quantity>().convert();
      v.kt = node["kt"].as<Quantity>().convert();
      v.kr = node["kr"].as<Quantity>().convert();
      v.mu = node["mu"].as<Quantity>().convert();
      v.damp_rate = node["damp_rate"].as<Quantity>().convert();

      return true;
    }
  };
} // namespace YAML
