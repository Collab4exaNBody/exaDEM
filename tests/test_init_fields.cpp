#include <exaDEM/init_fields.h>

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

#define NAME(Y) check_##Y
#define CHECK(X) is_equal(X,NAME(X));	

int main()
{
  using namespace exanb;
  // ** default test ** //
  constexpr Vec3d default_vec3d = {541, 0.2, 78};
  constexpr double default_double = 666;
  std::tuple<Vec3d, double> default_values = std::make_tuple(default_vec3d, default_double);
  exaDEM::initFunctor<Vec3d, double> default_test{default_values};

  // ** input parameters ** //
  double my_double(-151.0);
  Vec3d my_vec3d = {4556, 884, 124856};

  // ** expected results ** //
  double check_my_double = default_double;
  Vec3d check_my_vec3d = default_vec3d;

  // ** run default run ** //
  default_test(my_vec3d, my_double);

  // ** checks ** //
  CHECK(my_double);
  CHECK(my_vec3d);

  return EXIT_SUCCESS;
}

#undef CHECK
#undef NAME
