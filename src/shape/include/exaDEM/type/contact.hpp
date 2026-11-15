#pragma once

#include <exaDEM/basic_type.hpp>

namespace exaDEM {
struct contact {
  bool is_contact = false;  // true if contact is detected
  double dn = 0;            // normal gap (distance - sum of radii): negative if penetration, positive if separation
  Vec3d normal;             // contact normal
  Vec3d position;           // contact position
};
}  // namespace exaDEM
