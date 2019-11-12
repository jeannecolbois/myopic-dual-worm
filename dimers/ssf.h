#ifndef ssf_H
#define ssf_H

#include <vector>
#include <tuple>

// returns the difference of energy after performing a worm update on a single state
std::tuple<double, int> ssfs(double J1, double h,
  int* state, int* spinstate, int spinstatesize,
  int* s2p, int nd, double beta, int iters) ;
#endif // ssf_H
