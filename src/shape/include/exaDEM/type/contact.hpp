#pragma once

namespace exaDEM {
struct contact {
  bool is_contact = false;
  double dn = 0;
  exanb::Vec3d normal;
  exanb::Vec3d position;  // contact position
};
}  // namespace exaDEM
