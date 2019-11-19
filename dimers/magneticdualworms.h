#ifndef magneticdualworms_H
#define magneticdualworms_H

#include <vector>
#include <tuple>

// returns the difference of energy after performing a worm update on a single state
std::tuple<double, bool, std::vector<int>, std::vector<int>> magneticdualworms(double J1,
  std::vector<std::tuple<double, int*, int, int>> interactions, double h,
  int* state, int* spinstate, int statesize, int spinstatesize,
  int* d_nd, int n_nd, int* d_vd, int n_vd, int* d_wn,
  int* sidlist, int* didlist, int nbit,
  double beta, int saveloops, int nmaxiter, int iterworm);

#endif // magneticdualworms_H
