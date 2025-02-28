#include <exaDEM/compute_contact_force.h>

int main()
{
  using exaDEM::compute_contact_force;
  using exanb::Vec3d;

  Vec3d ft = {0.0, 0.0, 0.0};
  // define variables
  // particle i
  double rxi(0.0);
  double ryi(0.0);
  double rzi(0.0);

  double vxi(0.0);
  double vyi(0.0);
  double vzi(0.0);

  double massi(0.0);

  double Ri(2.0);

  double fxi(0.0);
  double fyi(0.0);
  double fzi(0.0);

  Vec3d momi = {0.0, 0.0, 0.0};
  Vec3d angveli = {0.0, 0.0, 0.0};

  // particle j
  double rxj(1.0);
  double ryj(1.0);
  double rzj(1.0);

  double vxj(0.0);
  double vyj(0.0);
  double vzj(0.0);

  double massj(0.0);

  double Rj(2.0);

  double fxj(0.0);
  double fyj(0.0);
  double fzj(0.0);
  double momxj(0.0);
  double momyj(0.0);
  double momzj(0.0);

  Vec3d momj = {0.0, 0.0, 0.0};
  Vec3d angvelj = {0.0, 0.0, 0.0};

  const double dncut = 0.1;
  const double dt = 0.0;
  const double kn = 0.0;
  const double kt = 0.0;
  const double kr = 0.0;
  const double fc = 0.0;
  const double mu = 0.0;
  const double dampRate = 0.0;

  compute_contact_force(dncut, dt, kn, kt, kr, fc, mu, dampRate, ft, rxi, ryi, rzi, vxi, vyi, vzi, massi, Ri, fxi, fyi, fzi, momi, angveli, rxj, ryj, rzj, vxj, vyj, vzj, massj, Rj, fxj, fyj, fzj, momj, angvelj);

  return EXIT_SUCCESS;
}
