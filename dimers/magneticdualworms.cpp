#include "magneticdualworms.h"

#include <random>
#include <cmath> //exp
#include <algorithm>

#include "mersenne.h" // random_gen()
#include "manydualworms.h"
#include "updatespinstates.h"

using namespace std;

std::tuple<double, bool, std::vector<int>, std::vector<int>> magneticdualworms(double J1,
  std::vector<std::tuple<double, int*, int, int>> interactions, double h,
  int* state, int* spinstate, int statesize, int spinstatesize,
  int* d_nd, int n_nd, int* d_vd, int n_vd, int* d_wn,
  int* sidlist, int* didlist, int nbit,
  double beta, int saveloops, int nmaxiter, int iterworm) {
    // save a copy of the state:
    int *savestate = new int[statesize];
    for( int dim = 0; dim < statesize; dim++){
        savestate[dim] = state[dim];
    }
    ////Call manydualworms
    tuple<double, bool, vector<int>, vector<int>> resultworm = manydualworms(J1,
      interactions, state, statesize, d_nd, n_nd, d_vd, n_vd, d_wn, beta,
      saveloops, nmaxiter, iterworm);
    double& deltaE = get<0>(resultworm);// treating as reference so no need to update resultworm at the end!
    bool& updated = get<1>(resultworm);// treating as reference so no need to update resultworm at the end!
    if(updated){
        //// Update the spin state from the new state and compute the old magnetic
        // energy at the same time
        int *savespinstate = new int[spinstatesize];

        for( int spin = 0; spin < spinstatesize; spin++){
          savespinstate[spin] = spinstate[spin];
        }

        bool randspinstate = false;
        updatespinstate(state, spinstate, sidlist, didlist, nbit, randspinstate);
        //// Compute the acceptance ratio
        //    1- compute the difference of magnetic energy
        int spinsdiff = 0;
        for( int spin = 0; spin < spinstatesize; spin++){
          spinsdiff += spinstate[spin] - savespinstate[spin];
        }
        //    2- compute exp(-beta*DELTAE) (but DeltaE = -h*spinsdiff)
        double p = exp(beta*h*spinsdiff);
        // Accept or reject by throwing a dice
        uniform_real_distribution<double> real_distrib(0.0, 1.0);
        double r = real_distrib(random_gen());
        if(r<p){// accept -> change energy
          deltaE += -h*spinsdiff;
        }else{// reject
          // revert state
          for( int dim = 0; dim < statesize; dim++){
            state[dim] = savestate[dim];
          }
          // revert spinstate
          for( int spin = 0; spin < spinstatesize; spin++){
            spinstate[spin] = savespinstate[spin];
          }
          // revert energy change
          deltaE = 0;
          // state that no update
          updated = false;
        }
        //clean up
        delete[] savestate;
        delete[] savespinstate;
    }else{// if not updated
      //clean up and return the result from manydualworms
      delete[] savestate;
    }

    return resultworm;
  }