#ifndef USPINSTATES_H
#define USPINSTATES_H
#include <omp.h>


void updatespinstates(int* states, int* spinstates, int* stat_temps,
  int* sidlist, int* didlist, int nbstat, int statesize,
  int spinstatesize, int nthreads, int nbit, bool randspinstate);
void updatespinstate(int* state, int* spinstate, int* sidlist,
  int* didlist, int nbit, bool randspinstate);
#endif //USPINSTATES_H
