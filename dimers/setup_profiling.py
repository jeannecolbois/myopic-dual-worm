#!/usr/bin/env python

import numpy as np
from numpy.distutils.core import setup
from numpy.distutils.core import Extension

cext = Extension(name = "dimers",
                sources=["hamiltonian.cpp", 
                "updatespinstates.cpp", "measupdate.cpp",
                "magneticmcsevolve.cpp","magneticdualworms.cpp",
                "ssf.cpp", "ssfsevolve.cpp",
                "ssf_gen.cpp","genssfsevolve.cpp",
                "mcsevolve.cpp", "manydualworms.cpp",
                "dimers.cpp"],
                extra_compile_args=['-std=c++11', '-fopenmp','-g','-pg','-Og'],
                include_dirs = [np.get_include()],
                extra_link_args=['-lgomp','-g','-pg'],
                language="c++")


if __name__=="__main__":
	setup(ext_modules=[cext])
