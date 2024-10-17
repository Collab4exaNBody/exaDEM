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
#include <exanb/core/basic_types.h>

namespace exaDEM
{
  struct ContactParams
  {
    double rcut;
    double dncut;
    double m_kn;
    double m_kt;
    double m_kr;
    double m_fc;
    double m_mu;
    double m_damp_rate;

    std::string convert_to_string() const
    {
      std::string res = "{";
      res += "rcut: " + std::to_string(rcut) + "m, ";
      res += "dncut: " + std::to_string(dncut) + ", ";
      res += "kn: " + std::to_string(m_kn) + ", ";
      res += "kt: " + std::to_string(m_kt) + ", ";
      res += "kr: " + std::to_string(m_kr) + ", ";
      res += "fc: " + std::to_string(m_fc) + ", ";
      res += "mu: " + std::to_string(m_mu) + ", ";
      res += "damp_rate: " + std::to_string(m_damp_rate) + "}";
      return res;
    }
  };
} // namespace exaDEM

// Yaml conversion operators, allows to read potential parameters from config file
namespace YAML
{
  using exaDEM::ContactParams;
  using exanb::lerr;
  using exanb::Quantity;
  using exanb::UnityConverterHelper;

  template <> struct convert<ContactParams>
  {
    static bool decode(const Node &node, ContactParams &v)
    {
      if (!node.IsMap())
      {
        return false;
      }
      if (!node["rcut"])
      {
        lerr << "rcut is missing\n";
        return false;
      }
      if (!node["dncut"])
      {
        lerr << "dncut is missing\n";
        return false;
      }
      if (!node["kn"])
      {
        lerr << "kn is missing\n";
        return false;
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
      if (!node["fc"])
      {
        lerr << "fc is missing\n";
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

      v = ContactParams{}; // initializes defaults values

      v.rcut = node["rcut"].as<Quantity>().convert();
      v.dncut = node["dncut"].as<Quantity>().convert();
      v.m_kn = node["kn"].as<Quantity>().convert();
      v.m_kt = node["kt"].as<Quantity>().convert();
      v.m_kr = node["kr"].as<Quantity>().convert();
      v.m_fc = node["fc"].as<Quantity>().convert();
      v.m_mu = node["mu"].as<Quantity>().convert();
      v.m_damp_rate = node["damp_rate"].as<Quantity>().convert();

      return true;
    }
  };
} // namespace YAML
