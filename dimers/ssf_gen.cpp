#include "ssf_gen.h"
#include <random>
#include <cmath>
#include <algorithm>
#include "mersenne.h"
using namespace std;

std::tuple<double, int> genssfs(double J1,
  std::vector<std::tuple<double, int*, int, int>> interactions,
  double h,
  int* state, int* spinstate, int spinstatesize,
  int* s2p, int nd, double beta, int iters) {
    // // save the state and spinstate
    // int *savestate = new int[statesize];
    // for( int dim = 0; dim < statesize; dim++){
    //     savestate[dim] = state[dim];
    // }
    // int *savespinstate = new int[spinstatesize];
    // for( int spin = 0; spin < spinstatesize; spin++){
    //     savespinstate[spin] = spinstate[spin];
    // }

    // define distributions
    uniform_int_distribution<int> int_distrib(0, spinstatesize-1);
    uniform_real_distribution<double> double_distrib(0.0, 1.0);
    // make spinstatesize ssf attempts
    double deltaE = 0.0;
    int rejected = 0;

    /* INITIALISAION */
    // vector for dimers indices
    vector<int> dimids(nd);
    // couplings pre-allocation
    auto couplings = interactions[0];
    double J = get<0>(couplings); //coupling
    int* M = get<1>(couplings); // Matrix of paths of dimers coupled to a given dimer
    int num_paths = get<2>(couplings); // number of paths
    int num_nei = get<3>(couplings); // number of dimers/paths

    for(int ssfiter = 0; ssfiter < spinstatesize*iters; ssfiter++) {

      // select a spin at random
      int spinid = int_distrib(random_gen());

      // compute the energy difference related to flipping the spin
      double en = 0.0;
      // 1 - couplings energy with the four dimers not flipped
      //    a) List of dimers to flip, and J1 energy
      for(int nid = 0; nid < nd; nid++){
        dimids[nid] = s2p[spinid*nd + nid];
        en += J1*state[dimids[nid]];
      }

      //   b) compute the energy without flipping taking each path into account once
      for(int nid = 0; nid < nd; nid++){
        int d = dimids[nid];
        for(auto couplings: interactions) { // other contributions
          J = get<0>(couplings); //coupling
          M = get<1>(couplings); // Matrix of paths of dimers coupled to a given dimer
          num_paths = get<2>(couplings); // number of paths
          num_nei = get<3>(couplings); // number of dimers/paths

          // for the dimer d, look at all the neighbours it's connected to and its interactions
          // more precisely: on each path, perform the product, then sum over the paths

          for(int path = 0; path <num_paths; path++) {
            double prod = state[d];
            for(int nei = 0; nei < num_nei; nei ++) {
              int dnei = M[(d*num_paths + path) * num_nei + nei];
              for(int nnid = 0; nnid < nid; nnid++){
                if(dnei == dimids[nnid]){prod = 0;} // if we went through this path once, we're not counting it again
              }
              prod = prod * state[dnei]; //product of dimer values
            }
            en += J * prod; //sum over the paths
          }
        }
      }

      // 2 - couplings energy with the four dimers flipped
      double ennew = 0;
      //    a) flip the dimers and compute the J1 energy
      for(int nid = 0; nid < nd; nid++){
        int dimid = dimids[nid];
        state[dimid] *= -1;
        ennew += J1*state[dimid];
      }

      //    b) compute the energy with the flipped dimers
      for(int nid = 0; nid < nd; nid++){
        int d = dimids[nid];
        for(auto couplings: interactions) { // other contributions
          J = get<0>(couplings); //coupling
          M = get<1>(couplings); // Matrix of paths of dimers coupled to a given dimer
          num_paths = get<2>(couplings); // number of paths
          num_nei = get<3>(couplings); // number of dimers/paths

          // for the dimer d, look at all the neighbours it's connected to and its interactions
          // more precisely: on each path, perform the product, then sum over the paths

          for(int path = 0; path <num_paths; path++) {
            double prod = state[d];
            for(int nei = 0; nei < num_nei; nei ++) {
              int dnei = M[(d*num_paths + path) * num_nei + nei];
              for(int nnid = 0; nnid < nid; nnid++){
                if(dnei == dimids[nnid]){prod = 0;} // if we went through this path once, we're not counting it again
              }
              prod = prod * state[dnei]; //product of dimer values
            }
            ennew += J * prod; //sum over the paths
          }
        }
      }

      // 3 - magnetic field
      ennew = ennew - en + 2*h*spinstate[spinid]; // compute the energy difference
      // compute the acceptance probability
      double p = exp(-beta*ennew);
      // accept or reject:
      double r = double_distrib(random_gen());
      if(r < p){// accept -> change energy and update the spin state
        deltaE += ennew;
        spinstate[spinid] *= -1;
      }else{// reject
        rejected += 1;
        // revert the state
        for(int nid = 0; nid < nd; nid++){
          int dimid = dimids[nid];
          state[dimid] *= -1;
        }
      }
    }
    tuple<double, int> resultingtuple = make_tuple(deltaE, rejected);
    return resultingtuple;
  }
