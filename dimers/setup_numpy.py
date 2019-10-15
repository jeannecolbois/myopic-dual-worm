#!/usr/bin/env python

import numpy as np
from distutils.core import setup, Extension
#from numpy.distutils.core import Extension
#from numpy.distutils.misc_util import setup
#import numpy.distutils.misc_util

#links =numpy.distutils.misc_util.get_numpy_include_dirs();
#links.append('-lgomp')
cext = Extension(name = "dimers",
                sources=["dimers_numpy.cpp", "hamiltonian.cpp","manydualworms.cpp", "mcsevolve.cpp", "updatespinstates.cpp"],
                extra_compile_args=['-std=c++11', '-fopenmp'],
                extra_link_args=['-lgomp'],
                language="c++")


if __name__=="__main__":
	setup(ext_modules=[cext], include_dirs = [np.get_include()])
#, include_dirs=numpy.distutils.misc_util.get_numpy_include_dirs())
