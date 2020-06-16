#include "measupdate.h"
#include <random>
#include "mersenne.h"

using namespace std;

void  measupdates(int* states, int* spinstates, int* stat_temps,
  int* sidlist, int* didlist, int* nnspins, int* s2p, int nbstat, int statesize,
  int spinstatesize, int nthreads, int nbit, int nn, int ndims, double p, int version) {
    #pragma omp parallel for schedule(dynamic,1) num_threads(nthreads)
    for(int resindex = 0; resindex < nbstat; resindex++){
      // for each result index temperature, we are going to update the spinstate
      // first, get the  temperature index so we know where to look in spinstates
      int tindex = stat_temps[resindex];
      // get the pointer to the tindex element in state and spinstate
      int *state = &states[tindex*statesize];
      int *spinstate = &spinstates[tindex*spinstatesize];
      measupdate(state, spinstate, sidlist, didlist, nnspins, s2p, nbit, nn, ndims, p, version);
    }
  }

  void  measupdate(int* state, int* spinstate, int* sidlist,
    int* didlist, int* nnspins, int* s2p, int nbit, int nn, int ndims, double p, int version) {
      uniform_real_distribution<double> real_distrib(0.0, 1.0);
      for(int it = 0; it < nbit; it++){
        int sid = sidlist[it+1];
        int s = spinstate[sid];
        if(s == -1){//if the spin is down, we get a chance at flipping it
          int sum = 0;
          for(int sneiid = 0; sneiid < nn; sneiid++){
            int snei = nnspins[nn*sid + sneiid];
            sum += spinstate[snei];
          }
          if(version == 0){
            if(sum == 0){// then flip with prob p
              double r = real_distrib(random_gen());
              if(r<p){
                spinstate[sid] = 1;
                for(int pid = 0; pid < ndims; pid++){//and flip the corresponding dimers
                  int did = s2p[sid*ndims + pid];
                  state[did]*=-1;
                }
              }
            }
          }
          if(version == 1){
            // in this case, p is interpreted as htip/Jtip
            //dE = -2*p+2*sum
            //dE <0 <> 2*sum < 2*p
            if(sum <= p){// then flip
              spinstate[sid] = 1;
              for(int pid = 0; pid < ndims; pid++){//and flip the corresponding dimers
                int did = s2p[sid*ndims + pid];
                state[did]*=-1;
              }
            }
          }
        }
      }
    }
