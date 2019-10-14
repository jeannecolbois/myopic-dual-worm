from distutils.core import setup, Extension
from numpy.distutils.misc_util import Configuration
import numpy.distutils.misc_util

def configuration(parent_package='', top_path=None):
	config = Configuration("dimers", parent_package, top_path);
	config.add_extension(name="dimers",
                sources=["dimers_numpy.cpp", "hamiltonian.cpp","manydualworms.cpp", "mcsevolve.cpp"],
		include_dirs=numpy.distutils.misc_util.get_numpy_include_dirs(),
                extra_compile_args=['-std=c++11', '-fopenmp'],
                extra_link_args= ['-lgomp'],
                language="c++")

	return config

if __name__=="__main__":
	from numpy.distutils.core import setup
	setup(configuration = configuration)
#setup(name="dimers",
#    ext_modules=[c_ext]#, include_dirs =numpy.distutils.misc_util.get_numpy_include_dirs()
#)
