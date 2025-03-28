from spack import *

class Onika(CMakePackage):
    """Onika is runtime library.
    """

    homepage = "https://github.com/Collab4exaNBody/onika"
    git = "https://github.com/Collab4exaNBody/onika.git"


    version("main",  git='https://github.com/Collab4exaNBody/onika.git', branch='main')
    version("exadem-exanbody-2.0",  git='https://github.com/Collab4exaNBody/onika.git', tag='exadem-exanbody-2.0')

    depends_on("cmake")
    variant("cuda", default=False, description="Support for GPU")
    depends_on("yaml-cpp")
    depends_on("openmpi")
    depends_on("cuda", when="+cuda")
    default_build_system = "cmake"
    build_system("cmake", default="cmake")

    variant(
        "build_type",
        default="Release", 
        values=("Release", "Debug", "RelWithDebInfo"),
        description="CMake build type",
        )

    def cmake_args(self):
      args = [ self.define_from_variant("-DONIKA_BUILD_CUDA=ON", "cuda"), ]
      return args
