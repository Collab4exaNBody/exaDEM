#pragma once
#include <exanb/core/basic_types.h>

namespace exaDEM
{
  struct HookeParams
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
}

// Yaml conversion operators, allows to read potential parameters from config file
namespace YAML
{
  using exaDEM::HookeParams;
  using exanb::UnityConverterHelper;
  using exanb::Quantity;
  using exanb::lerr;

  template<> struct convert<HookeParams>
  {
    static bool decode(const Node& node, HookeParams& v)
    {    
      if( !node.IsMap() ) { return false; }
      if( ! node["rcut"] ) { lerr<<"rcut is missing\n"; return false; }
      if( ! node["dncut"] ) { lerr<<"dncut is missing\n"; return false; }
      if( ! node["kn"] ) { lerr<<"kn is missing\n"; return false; }
      if( ! node["kt"] ) { lerr<<"kt is missing\n"; return false; }
      if( ! node["kr"] ) { lerr<<"kr is missing\n"; return false; }
      if( ! node["fc"] ) { lerr<<"fc is missing\n"; return false; }
      if( ! node["mu"] ) { lerr<<"mu is missing\n"; return false; }
      if( ! node["damp_rate"] ) { lerr<<"damp_rate is missing\n"; return false; }

      v = HookeParams{}; // initializes defaults values

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
}
