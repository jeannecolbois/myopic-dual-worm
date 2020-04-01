#ifndef ssf_gen_H
#define ssf_gen_H

#include <vector>
#include <tuple>


// returns the difference of energy after performing a single spin flip update
// on a single state
std::tuple<double, int> genssfs(double J1,
  std::vector<std::tuple<double, int*, int, int>> interactions,
  double h,
  int* state, int* spinstate, int spinstatesize,
  int* s2p, int nd, double beta, int iters) ;
#endif // ssf_gen_H
