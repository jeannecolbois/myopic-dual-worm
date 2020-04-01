#ifndef GENssfS_H
#define GENssfS_H
#include <vector>
#include <tuple>
#include <omp.h>


void genssfsevolve(double J1, std::vector<std::tuple<double, int*, int, int>> interactions,
  int* states, int statesize, int* spinstates, int spinstatesize, int* s2p,
  int nd,  double* walker2params, double* walker2ids, double* energies, int* failedupdates,
  int nbwalkers, int nthreads, int iters, int nt, int nh);
#endif //GENssfS_H
