#pragma once

namespace exaDEM {
struct contact {
  bool is_contact = false;  // true if contact is detected
  double dn = 0;            // normal gap (distance - sum of radii): negative if penetration, positive if separation
  exanb::Vec3d normal;      // contact normal
  exanb::Vec3d position;    // contact position
};
}  // namespace exaDEM
