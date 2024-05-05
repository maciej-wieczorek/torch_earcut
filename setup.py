from setuptools import find_packages, setup
from torch.utils import cpp_extension
from collections import defaultdict

BUILD_ARGS = defaultdict(lambda: ["-std=c++20", "-Ofast", "-flto", "-fopenmp"])
BUILD_ARGS["msvc"] = ["/std:c++20", "/O2", "/openmp"]
    
class BuildExtensionWithArgs(cpp_extension.BuildExtension):
    def build_extensions(self):
        args = BUILD_ARGS[self.compiler.compiler_type]
        for ext in self.extensions:
            ext.extra_compile_args = args
        cpp_extension.BuildExtension.build_extensions(self)
        
setup(name="torch_earcut",
      author = "Antoni Nowinowski",
      author_email = "antoninowinowski@hotmail.com",
      description = "PyTorch binding for the Mapbox Earcut library.",
      version="0.1.0",
      packages=find_packages(where="src"),
      package_dir={"": "src"},
      install_requires=["torch"],
      ext_modules=[cpp_extension.CppExtension(name="torch_earcut.cpp", sources=["src/torch_earcut.cpp"])],
      cmdclass={"build_ext": BuildExtensionWithArgs}
)