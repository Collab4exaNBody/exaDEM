#pragma once

#include <drivers.h>

namespace exaDEM
{
  using namespace exanb;

  struct driver_io_writer
  {
    std::string filename;
    inline  operator() (Drivers& drvs)
    {
      std::stringstream buffer;
			size_t size = drvs.size();
      buffer << size  << std::endl;
      for(int i = 0; i < size ; i++)
      {
        //drvs[i].write(bufffer);
      }
    }
  };
}
