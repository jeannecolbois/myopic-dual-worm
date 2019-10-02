What's this doing?
------------------
These are a few python functions which help to perform a Monte-Carlo algorithm to study the Ising model on either the triangular or the kagome lattice. The general idea is to create a dual worm algorithm allowing the study of the system.

 -> Dimers.cpp:

 -> Hamiltonian.cpp:
    Defines the function: double hamiltonian(double J1, std::vector<std::tuple<double,int*,int,int>> interactions, int* state, int statesize)
    This function allows to compute the energy of the system given
      > the state of the system (i.e. a list of values 1 and -1 depending on the dimer state on each bond)
      > a table of tuples (coupling, table of (ref bond, list of dimers having this coupling with ref))

 -> Manydualworms.cpp:

 -> Myopic.pro: Qt creator handle of the above code.





How should I compile it?
------------------------
