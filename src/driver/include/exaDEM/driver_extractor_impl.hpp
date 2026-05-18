#pragma once

namespace exaDEM {
// Functor to extract specified fields from a driver and format them as a string
struct DriverExtractFunc {
  std::string stream;           // Output buffer accumulating extracted data
  extractor::Tracker& tracker;  // Tracker defining which fields to extract

  // Extract all fields specified in tracker and append to stream
  template <typename DriverT>
  inline void operator()(DriverT& driver) {
    for (auto& field : tracker.fields) {
      extract(field, driver);
      stream += " ";
    }
  }

  // Extract a single field value from the driver and append to stream
  // @param field The field type to extract (from Dictionnary enum)
  // @param driver The driver object to extract data from
  template <typename DriverT>
  void extract(extractor::Dictionnary field, DriverT& driver) {
    using exaDEM::extractor::Dictionnary;
    // Cache commonly accessed driver properties
    const exanb::Vec3d center = driver.position();        // Center position
    const exanb::Vec3d vel = driver.velocity();           // Linear velocity
    const exanb::Vec3d vrot = driver.angular_velocity();  // Angular velocity
    const exanb::Vec3d force = driver.forces();           // Accumulated forces
    // Driver type identifier
    if (field == Dictionnary::type) {
      stream += print(driver.get_type());
    }
    // Position components (x, y, z)
    else if (field == Dictionnary::rx) {
      stream += std::to_string(center.x);
    } else if (field == Dictionnary::ry) {
      stream += std::to_string(center.y);
    } else if (field == Dictionnary::rz) {
      stream += std::to_string(center.z);
    }
    // Linear velocity components (vx, vy, vz)
    else if (field == Dictionnary::vx) {
      stream += std::to_string(vel.x);
    } else if (field == Dictionnary::vy) {
      stream += std::to_string(vel.y);
    } else if (field == Dictionnary::vz) {
      stream += std::to_string(vel.z);
    }
    // Angular velocity components (vrotx, vroty, vrotz)
    else if (field == Dictionnary::vrotx) {
      stream += std::to_string(vrot.x);
    } else if (field == Dictionnary::vroty) {
      stream += std::to_string(vrot.y);
    } else if (field == Dictionnary::vrotz) {
      stream += std::to_string(vrot.z);
    }
    // Force components (fx, fy, fz)
    else if (field == Dictionnary::fx) {
      stream += std::to_string(force.x);
    } else if (field == Dictionnary::fy) {
      stream += std::to_string(force.y);
    } else if (field == Dictionnary::fz) {
      stream += std::to_string(force.z);
    }
  }
};
}  // namespace exaDEM