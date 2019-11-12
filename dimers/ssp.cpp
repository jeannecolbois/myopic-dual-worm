#include "ssp.h"
#include <random>
#include <cmath>
#include <algorithm>
#include "mersenne.h"
using namespace std;
std::tuple<double, int> ssps(double J1, double h,
  int* state, int* spinstate, int spinstatesize,
  int* s2p, int nd, double beta) {
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
    // make spinstatesize ssp attempts
    double deltaE = 0.0;
    int rejected = 0;

    for(int sspiter = 0; sspiter < spinstatesize; sspiter++) {
      // select a spin at random
      int spinid = int_distrib(random_gen());

      // compute the energy difference related to flipping the spin
      // 1 - magnetic field
      double en = 2*h*spinstate[spinid];
      // 2 - and the corresponding dimer plaquette
      for(int nid = 0; nid < nd; nid++){
        int dimid = s2p[spinid*nd + nid];
        en += -2*J1*state[dimid];
      }

      // compute the acceptance probability
      double p = exp(-beta*en);
      // accept or reject:
      double r = double_distrib(random_gen());
      if(r < p){// accept -> change energy and update the spin and dimer state
        deltaE += en;
        spinstate[spinid] *= -1;
        for(int nid = 0; nid < nd; nid++){
          int dimid = s2p[spinid*nd + nid];
          state[dimid] *= -1;
        }
      }else{// reject
        rejected += 1;
      }
    }
    tuple<double, int> resultingtuple = make_tuple(deltaE, rejected);
    return resultingtuple;
  }
