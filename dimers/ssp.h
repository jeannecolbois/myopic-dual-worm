#ifndef ssp_H
#define ssp_H

#include <vector>
#include <tuple>

// returns the difference of energy after performing a worm update on a single state
std::tuple<double, int> ssps(double J1, double h,
  int* state, int* spinstate, int spinstatesize,
  int* s2p, int nd, double beta) ;
#endif // ssp_H
