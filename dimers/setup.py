#!/usr/bin/env python

import numpy as np
from numpy.distutils.core import setup
from numpy.distutils.core import Extension

cext = Extension(name = "dimers",
                sources=["hamiltonian.cpp", "updatespinstates.cpp", "magneticdualworms.cpp", "measupdate.cpp", "ssf.cpp", "ssfsevolve.cpp","mcsevolve.cpp", "manydualworms.cpp", "magneticmcsevolve.cpp", "dimers.cpp"],
                extra_compile_args=['-std=c++11', '-fopenmp'],
                include_dirs = [np.get_include()],
                extra_link_args=['-lgomp'],
                language="c++")


if __name__=="__main__":
	setup(ext_modules=[cext])
