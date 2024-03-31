#pragma once
#include <exaDEM/driver_base.h>

namespace exaDEM
{
  using namespace exanb;

  struct UndefinedDriver
  {
    /**
     * @brief Get the type of the driver (in this case, UNDEFINED).
     * @return The type of the driver.
     */
    constexpr DRIVER_TYPE get_type() {return DRIVER_TYPE::UNDEFINED;}

    /**
     * @brief Print information about the undefined driver.
     */
    void print()
    {
      std::cout << "Driver Type: UNDEFINED" << std::endl;
    }
  };
}
