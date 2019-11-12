#ifndef MAGSSPS_H
#define MAGSSPS_H
#include <vector>
#include <tuple>


void sspsevolve(double J1,
  double h, int* states, int statesize, int* spinstates, int spinstatesize, int* s2p,
  int nd, double* betas, double* energies, int* failedupdates, int nbt, int nthreads);
#endif //MAGSSPS_H
