# Monte Carlo simulations in C++ with python wrapper for managing the runs and the analysis
Implementing 
-> Cluster algorithms for frustrated two-dimensional Ising antiferromagnets via dual worm constructions, Geet Rakala and Kedar Damle, Phys. Rev. E 96, (2017)
-> Estimating errors reliably in Monte Carlo simulations of the Ehrenfest model American Journal of Physics 78, Vinay Ambegaokar and ,(2010); https://doi.org/10.1119/1.3247985
-> Feedback-optimized parallel tempering Monte Carlo,  Helmut G Katzgraber, Simon Trebst, David A Huse and Matthias Troyer, J. Stat. Mech. (2006) 

# Getting jupyter-notebooks

https://jupyter.org/install

# Installing the dimer module
Go to ./dimers/

Type:

(sudo) python3 setup.py install --user (--record dimersinstall.txt)

Note: the --record option allows to save the location of installed files so that uninstalling is easier.

# Testing
Go to ./dualworm-kagome/ProfilingAndTesting/

Open the ProfilerTest.ipynb jupyter notebook and run it. You should get a final output "Everything seems to be running smoothly".

# Ground states
Go to ./dualworm-kagome/GroundStates/

Each notebook has already the right parameters to generate states in the corresponding ground state phase. The resuls are saved in the corresponding subfolder.
This should be reasonably self-explanatory.

# Quick Kagome "ED"

Go to ./dualworm-kagome/ExampleKagome

Have fun with the jupyter-notebook TestEDKagomeLattice.ipynb
