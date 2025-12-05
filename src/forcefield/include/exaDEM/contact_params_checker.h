#pragma once
#include <exaDEM/contact_force_parameters.h>
#include <exaDEM/compute_contact_force.h>   // pour ContactLawType / AdhesionLawType
#include <exaDEM/color_log.hpp>

namespace exaDEM
{
  inline void check_contact_params( 
                                    const std::string opertor_name,
                                    std::optional<ContactParams> p,
                                    std::optional<ContactParams> d,
                                    ContactLawType contact_law,
                                    AdhesionLawType adhesion_law)
  {

    bool has_p = static_cast<bool>(p);
    bool has_d = static_cast<bool>(d);
    std::cout << "has_p: " << has_p << " has_d: " << has_d << "\n";

  // Contact Law checker
  //----------------------------------------------------------------
  // Cohesive
  if (contact_law != ContactLawType::Cohesive)
  {
     if(p)
     {
        if(p->dncut > 0)
        {
          std::string msg = "dncut is != 0 while the cohesive force is not used.\n";
          msg += "                        Please, use contact_[InputType]_[Shape]_cohesive_[AdhesionLaw] operators.";
          color_log::error(opertor_name, msg);
        }
     }

     if(d)
     {
        if(d->dncut > 0)
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
    if(p)
    {

      if (p->gamma > 0.0)
      {
        std::string msg = "Law inconsistency.\n";
        msg += "                        You have defined gamma > 0.0 but no adhesion law (DMT/JKR) → gamma not accounted."; 
        color_log::warning(opertor_name,msg);
      }
    }
    if(d)
    {
      if (d->gamma > 0.0)
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
    std::cout << "checking DMT\n";

    if(p)
    {
              std::cout << "checking p gamma\n";

      if (p->gamma <= 0.0)
      {
        std::string msg = "Adhesion Law (DMT/JKR) is defined but no gamma is provided.\n";
        msg += "                        Please, define gamma."; 
        color_log::error(opertor_name, msg);
      }
    }
    if(d)
    {
                std::cout << "checking d gamma\n";

      if (d->gamma <= 0.0)
      {
        std::string msg = "Adhesion Law (DMT/JKR) is defined but no gamma is provided.\n";
        msg += "                        Please, define gamma."; 
        color_log::error(opertor_name, msg);
      }
    }
  }

  }
}
