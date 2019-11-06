#ifndef MEASUPDATE_H
#define MEASUPDATE_H
#include <omp.h>

void  measupdates(int* states, int* spinstates, int* stat_temps,
  int* sidlist, int* didlist, int* nnspins, int* s2p, int nbstat, int statesize,
  int spinstatesize, int nthreads, int nbit, int nn, int ndims, double p);
void  measupdate(int* state, int* spinstate, int* sidlist,
  int* didlist, int* nnspins, int* s2p, int nbit, int nn, int ndims, double p) ;
#endif //MEASUPDATE_H
