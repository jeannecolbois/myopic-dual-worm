#ifndef MCSEVOLVE_H
#define MCSEVOLVE_H
#include <vector>
#include <tuple>
#include <omp.h>

void mcsevolve(double J1, std::vector<std::tuple<double, int*, int, int>> interactions, int* states, int state_size, int* d_nd, int n_nd, int* d_vd, int n_vd, int* d_wn, double *betas, double *energies, int ntb, int nmaxiter, int niterworm, int nthreads);

#endif //MCSEVOLVE_H
