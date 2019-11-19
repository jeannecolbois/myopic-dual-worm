#ifndef MAGMCSEVOLVE_H
#define MAGMCSEVOLVE_H
#include <vector>
#include <tuple>
#include <omp.h>

void magneticmcsevolve(double J1, std::vector<std::tuple<double, int*, int, int>> interactions,
  double h, int* states, int statesize, int* spinstates, int spinstatesize,
  int* d_nd, int n_nd, int* d_vd, int n_vd, int* d_wn,
  int* sidlist, int* didlist, int nbit, double* betas,
  double* energies, int* failedupdates, int nbt, int nmaxiter, int niterworm, int nthreads);
#endif //MAGMCSEVOLVE_H
