from spack import *

class Exadem(CMakePackage):
    """ExaDEM is a DEM Simulation Code using the ExaNBody framework.
		"""

    homepage = "https://github.com/Collab4exaNBody/exaDEM"
    git = "https://github.com/Collab4exaNBody/exaDEM.git"


    version("main", commit="6d2d509abb7334ef7b6dc2f36b700747bfd12fab")
    variant("cuda", default=False, description="Support for GPU")

    depends_on("cmake")
    depends_on("yaml-cpp")
    depends_on("exanbody")
    depends_on("cuda", when="+cuda")


    build_system("cmake", default="cmake")

    @run_before("install")
    def pre_install(self):
        with working_dir(self.build_directory):
            # When building shared libraries these need to be installed first
            make("UpdatePluginDataBase")

    def cmake_args(self):
        args = [
          self.define_from_variant("-DXNB_BUILD_CUDA=ON", "cuda"),
        ]
        return args
