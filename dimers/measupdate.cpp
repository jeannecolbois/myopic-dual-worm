#include "measupdate.h"
#include <random>
#include "mersenne.h"

using namespace std;

void  measupdates(double p, double J1, int version,
  int* states, int* spinstates, double* energies,
  int* walker2ids, int nbwalkers, int nh,
  int* sidlist, int* nnspins, int* s2p,
  int statesize, int spinstatesize,
  int nthreads, int nbitscan, int nn, int ndims) {
    #pragma omp parallel for schedule(dynamic,1) num_threads(nthreads)
    for(int walker = 0; walker < nbwalkers; walker++) {
      // get the pointer to the walker in state and spinstate
      int *state = &states[walker*statesize];
      int *spinstate = &spinstates[walker*spinstatesize];
      // bid = walker2ids[walker*2];
      // hid = walker2ids[walker*2+1];
      energies[walker2ids[walker*2]*nh+walker2ids[walker*2+1]] += measupdate(p, J1, version,
        state, spinstate,
        sidlist, nnspins, s2p,
        nbitscan, nn, ndims);
    }
  }

double  measupdate(double p, double J1, int version,
  int* state, int* spinstate,
  int* sidlist, int* nnspins, int* s2p,
  int nbitscan, int nn, int ndims) {
      uniform_real_distribution<double> real_distrib(0.0, 1.0);
      double dE = 0.0;
      for(int it = 0; it < nbitscan; it++){ // sidlist gives an optimal way to go through all the spins
        int sid = sidlist[it+1];
        int s = spinstate[sid];


        if(s == -1){//if the spin is down, we get a chance at flipping it
          int sum = 0;
          for(int sneiid = 0; sneiid < nn; sneiid++){
            int snei = nnspins[nn*sid + sneiid];
            sum += spinstate[snei];
          }
          //spinstate[sid] = 1;
          if(version == 0){
            if(sum == 0){// then flip with prob p
              //double r = real_distrib(random_gen());

              //int val = spinstate[sid];
              /*if(r<p){
                spinstate[sid] = 1;
                for(int pid = 0; pid < ndims; pid++){//and flip the corresponding dimers
                  int did = s2p[sid*ndims + pid];
                  state[did]*=-1;
                }
              }*/
            }
          }
          if(version == 1){
            // in this case, p is interpreted as htip/Jtip, and the energy is updated
            //dE = -2*p+2*sum
            //dE <0 <> 2*sum < 2*p
            //spinstate[sid] = 1;
            if(sum <= p){// then flip
              //spinstate[sid] = 1;
              /*for(int pid = 0; pid < ndims; pid++){//and flip the corresponding dimers
                int did = s2p[sid*ndims + pid];
                state[did]*=-1;
              }
              dE += 2*J1*sum; // not -2*p because this is not physical...
              */
            }
          }
        }
      }
      return dE;
    }
