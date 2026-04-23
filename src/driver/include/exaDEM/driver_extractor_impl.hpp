#pragma once

namespace exaDEM {
  struct DriverExtractFunc {
    std::string stream;
    extractor::Tracker& tracker;

    template<typename DriverT>
    inline void operator()(DriverT& driver) {
      for (auto& field : tracker.fields) {
        extract(field, driver);
        stream += " "; 
      }
    }
    void extract(extractor::Dictionnary field, Cylinder& driver) {
      using exaDEM::extractor::Dictionnary;
      if (field == Dictionnary::type) {
        stream += std::to_string(4);
      } else if (field == Dictionnary::rx) {
        stream += std::to_string(driver.center.x);
      } else if (field == Dictionnary::ry) {
        stream += std::to_string(driver.center.y);
      } else if (field == Dictionnary::rz) {
        stream += std::to_string(driver.center.z);
      } else if (field == Dictionnary::vx) {
        stream += std::to_string(driver.vel.x);
      } else if (field == Dictionnary::vy) {
        stream += std::to_string(driver.vel.y);
      } else if (field == Dictionnary::vz) {
        stream += std::to_string(driver.vel.z);
      } else if (field == Dictionnary::vrotx) {
        stream += std::to_string(driver.vrot.x);
      } else if (field == Dictionnary::vroty) {
        stream += std::to_string(driver.vrot.y);
      } else if (field == Dictionnary::vrotz) {
        stream += std::to_string(driver.vrot.z);
      }
    }

    void extract(extractor::Dictionnary field, Surface& driver) {
      using exaDEM::extractor::Dictionnary;
      if (field == Dictionnary::type) {
        stream += std::to_string(5);
      } else if (field == Dictionnary::rx) {
        stream += std::to_string(driver.center.x);
      } else if (field == Dictionnary::ry) {
        stream += std::to_string(driver.center.y);
      } else if (field == Dictionnary::rz) {
        stream += std::to_string(driver.center.z);
      } else if (field == Dictionnary::vx) {
        stream += std::to_string(driver.vel.x);
      } else if (field == Dictionnary::vy) {
        stream += std::to_string(driver.vel.y);
      } else if (field == Dictionnary::vz) {
        stream += std::to_string(driver.vel.z);
      } else if (field == Dictionnary::vrotx) {
        stream += std::to_string(driver.vrot.x);
      } else if (field == Dictionnary::vroty) {
        stream += std::to_string(driver.vrot.y);
      } else if (field == Dictionnary::vrotz) {
        stream += std::to_string(driver.vrot.z);
      }
    }
    void extract(extractor::Dictionnary field, Ball& driver) {}
    void extract(extractor::Dictionnary field, RShapeDriver& driver) {}
  };
}  // namespace exaDEM
