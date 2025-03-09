from spack import *

class Exadem(CMakePackage):
    """ExaDEM is a DEM Simulation Code using the ExaNBody framework.
		"""

    homepage = "https://github.com/Collab4exaNBody/exaDEM"
    git = "https://github.com/Collab4exaNBody/exaDEM.git"

    version("main", git='https://github.com/Collab4exaNBody/exaDEM.git',  branch='main') 
    version("1.1.0", git='https://github.com/Collab4exaNBody/exaDEM.git', branch='v1.1.0')
    version("1.0.2", git='https://github.com/Collab4exaNBody/exaDEM.git', tag='v1.0.2', preferred=True )
    version("1.0.1", git='https://github.com/Collab4exaNBody/exaDEM.git', tag='v1.0.1')
    variant("cuda", default=False, description="Support for GPU")
    variant("rsampi", default=False, description="Support for particles generation using the RSA package")

    depends_on("cmake@3.27.9")
    depends_on("yaml-cpp@0.6.3")
    depends_on("openmpi")
    depends_on("exanbody")
    depends_on("cuda", when="+cuda")
    depends_on("rsampi", when="+rsampi")

# v1.1.0
    depends_on("exanbody@exadem-exanbody-2.0", when="@1.1.0")
    depends_on("onika@exadem-exanbody-2.0", when="@1.1.0")
    depends_on("exanbody+cuda", when="+cuda")
    depends_on("onika+cuda", when="+cuda")


    build_system("cmake", default="cmake")

    @run_before("install")
    def pre_install(self):
        with working_dir(self.build_directory):
            # When building shared libraries these need to be installed first
            if self.spec.version <= Version("1.0.2"):
              make("UpdatePluginDataBase")

    def cmake_args(self):
        if self.spec.version <= Version("1.0.2"):
          args = [self.define_from_variant("-DRSA_MPI=ON", "rsampi"), self.define_from_variant("-DXNB_BUILD_CUDA=ON", "cuda"), ]
        else:
          args = [self.define_from_variant("-DRSA_MPI=ON", "rsampi"), ]
        return args
