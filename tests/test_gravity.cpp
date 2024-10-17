#include <exaDEM/gravity_force.h>

#include <iostream>

// check function
template <typename T> bool my_check(T &res, T &check)
{
  bool ret;
  ret = res == check;
  return ret;
}

template <typename... Args> void is_equal(Args &&...a_args)
{
  bool is_equal = my_check(a_args...);
  if (!is_equal)
  {
    std::abort();
  }
}

#define NAME(Y) check_##Y;
#define CHECK(X) is_equal(X,NAME(X));	;

int main()
{
  using namespace exanb;
  // ** default test ** //
  static constexpr Vec3d default_gravity = {0.0, 0.0, -9.807};
  exaDEM::GravityForceFunctor default_test{default_gravity};

  // ** input parameters ** //
  double fx(1.0);
  double fy(10.0);
  double fz(100.0);
  double mass(2);

  // ** results ** //
  double check_mass = mass;
  double check_fx = fx;
  double check_fy = fy;
  double check_fz = 100.0 + mass * default_gravity.z;

  // ** run default run ** //
  default_test(mass, fx, fy, fz);

  // ** checks ** //
  CHECK(mass);
  CHECK(fx);
  CHECK(fy);
  CHECK(fz);

  return EXIT_SUCCESS;
}

#undef CHECK
#undef NAME
