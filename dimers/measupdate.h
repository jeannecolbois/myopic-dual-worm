#ifndef MEASUPDATE_H
#define MEASUPDATE_H
#include <omp.h>

void  measupdates(double p, double J1, int version,
  int* states, int* spinstates, double* energies,
  int* walker2ids, int nbwalkers, int nh,
  int* sidlist, int* nnspins, int* s2p,
  int statesize, int spinstatesize,
  int nthreads, int nbitscan, int nn, int ndims);
  double  measupdate(double p, double J1, int version,
    int* state, int* spinstate,
    int* sidlist, int* nnspins, int* s2p,
    int nbitscan, int nn, int ndims)  ;
#endif //MEASUPDATE_H
