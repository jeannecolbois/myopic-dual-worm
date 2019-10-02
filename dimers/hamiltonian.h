#ifndef HAMILTONIAN_H
#define HAMILTONIAN_H
#include <vector>
#include <tuple>

double hamiltonian(double J1, std::vector<std::tuple<double,int*,int,int>> interactions, int* state, int statesize);
#endif //HAMILTONIAN_H
