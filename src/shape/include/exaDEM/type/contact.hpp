#pragma once

namespace exaDEM
{
  struct contact {
    bool is_contact = false;
    double dn = 0;
    Vec3d normal;
    Vec3d position; // contact position
  }; 
};
