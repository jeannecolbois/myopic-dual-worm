#ifndef MAGssfS_H
#define MAGssfS_H
#include <vector>
#include <tuple>


void ssfsevolve(double J1,
  double h, int* states, int statesize, int* spinstates, int spinstatesize, int* s2p,
  int nd, double* betas, double* energies, int* failedupdates, int nbt, int nthreads, int iters);
#endif //MAGssfS_H
