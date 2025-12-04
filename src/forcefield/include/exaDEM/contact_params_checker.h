#pragma once
#include <exaDEM/contact_force_parameters.h>
#include <exaDEM/compute_contact_force.h>   // pour ContactLawType / AdhesionLawType
#include <exaDEM/color_log.hpp>

namespace exaDEM
{
  inline void check_contact_params( const bool lconfig, 
                                    const bool ldriver,
                                    const std::string opertor_name,
                                    const ContactParams& p,
                                    const ContactParams& d,
                                    ContactLawType contact_law,
                                    AdhesionLawType adhesion_law)
  {

  // Contact Law checker
  //----------------------------------------------------------------
  // Cohesive
  if (contact_law != ContactLawType::Cohesive)
  {
     if(lconfig)
     {
        if(p.dncut > 0)
        {
          std::string msg = "dncut is != 0 while the cohesive force is not used.\n";
          msg += "                        Please, use contact_[InputType]_[Shape]_cohesive_[AdhesionLaw] operators.";
          color_log::error(opertor_name, msg);
        }
     }

     if(ldriver)
     {
        if(d.dncut > 0)
        {
          std::string msg = "dncut is != 0 while the cohesive force is not used.\n";
          msg += "                        Please, use contact_[InputType]_[Shape]_cohesive_[AdhesionLaw] operators.";
          color_log::error(opertor_name, msg);
        }  
     }
  }

  // Adhesion Law checker
  //----------------------------------------------------------------

  // None
  if (adhesion_law == AdhesionLawType::None)
  {
    if(lconfig)
    {
      if (p.gamma > 0.0)
      {
        std::string msg = "Law inconsistency.\n";
        msg += "                        You have defined gamma > 0.0 but no adhesion law (DMT/JKR) → gamma not accounted."; 
        color_log::warning(opertor_name,msg);
      }
    }
    if(ldriver)
    {
      if (d.gamma > 0.0)
      {
        std::string msg = "Law inconsistency.\n";
        msg += "                        You have defined gamma > 0.0 but no adhesion law (DMT/JKR) → gamma not accounted."; 
        color_log::warning(opertor_name,msg);
      }
    }
  }

  // DMT / JKR
  if (adhesion_law == AdhesionLawType::DMT )//|| adhesion_law == AdhesionLawType::JKR)
  {
    if(lconfig)
    {
      if (p.gamma <= 0.0)
      {
        std::string msg = "Adhesion Law (DMT/JKR) is defined but no gamma is provided.\n";
        msg += "                        Please, define gamma."; 
        color_log::error(opertor_name, msg);
      }
    }
    if(ldriver)
    {
      if (d.gamma <= 0.0)
      {
        std::string msg = "Adhesion Law (DMT/JKR) is defined but no gamma is provided.\n";
        msg += "                        Please, define gamma."; 
        color_log::error(opertor_name, msg);
      }
    }
  }


}
