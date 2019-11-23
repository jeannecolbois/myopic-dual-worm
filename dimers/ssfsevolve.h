#ifndef MAGssfS_H
#define MAGssfS_H
#include <vector>
#include <tuple>


void ssfsevolve(double J1, int* states, int statesize, int* spinstates, int spinstatesize, int* s2p,
  int nd,  double* walker2params, double* energies, int* failedupdates,
  int nbwalkers, int nthreads, int iters);
#endif //MAGssfS_H
