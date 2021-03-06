#ifndef manydualworms_H
#define manydualworms_H

#include <vector>
#include <tuple>

// returns the difference of energy after performing a worm update on a single state
std::tuple<double, bool, std::vector<int>, std::vector<int>> manydualworms(double J1,
  std::vector<std::tuple<double, int*, int, int>> interactions, int* state,
  int statesize, int* d_nd, int n_nd, int* d_vd, int n_vd, int* d_wn,
  double beta, int saveloops, int nmaxiter, int iterworm);
std::tuple<double, int> dualworm(bool* loopclosed, int* w1, int* w2, double J1,
  std::vector<std::tuple<double, int*, int, int>> interactions, int* state,
  int statesize, int* d_nd, int n_nd, int* d_vd, int n_vd, int* d_wn,
  double beta, int saveloops, std::vector<int> &update, int nmaxiter);
#endif // manydualworms_H
