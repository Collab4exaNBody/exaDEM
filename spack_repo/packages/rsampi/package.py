from spack import *

class Rsampi(CMakePackage):
    """C++ library : HPC implementation of RSA algorithm for maximal Poisson-sphere sampling 
		"""

    homepage = "https://github.com/MarcJos/RSA_MPI"
    git = "https://github.com/MarcJos/RSA_MPI.git"

    version('0.1.0', git='https://github.com/MarcJos/RSA_MPI.git', branch='v0.1.0')

    depends_on("cmake")
    depends_on("openmpi")
    build_system("cmake", default="cmake@3.26.3")

    @run_before("cmake")
    def pre_install(self):
         bash = which("bash")
         bash("./Installation/Pre-install.sh")

    def cmake_args(self):
        args = []
        return args
