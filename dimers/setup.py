from distutils.core import setup, Extension
import numpy.distutils.misc_util

c_ext = Extension("dimers",
	sources=["dimers.cpp", "hamiltonian.cpp","manydualworms.cpp", "mcsevolve.cpp"],
	extra_compile_args=['-std=c++11', '-fopenmp'],
	extra_link_args= ['-lgomp'],
	language="c++")

setup(
    ext_modules=[c_ext],
    include_dirs=numpy.distutils.misc_util.get_numpy_include_dirs(),
)
