#include "measupdates.h"
#include <random>
#include "mersenne.h"
#include <omp.h>
#include <cmath>
#include <vector>


using namespace std;
void measupdates(double J1, double htip, double Ttip, double pswitch, bool uponly, int* states, int statesize,
 int* spinstates, int spinstatesize, int* s2p, int ndims, int* sidlist, int nbsid,
 int* walker2ids, double* energies, int nbwalkers, int nthreads, int nt,
 int nh, int* updatelists, bool saveupdates) {
  #pragma omp parallel for schedule(dynamic, 1) num_threads(nthreads)

  for(int walker = 0; walker < nbwalkers; walker++ ){
  // for each walker, measupdates
    int *state = &states[walker*statesize];
    int *spinstate =  &spinstates[walker*spinstatesize];
    int bid = walker2ids[walker*2];
    int hid = walker2ids[walker*2 + 1];

    int *updatelist = &updatelists[(bid*nh + hid)*nbsid];

    double dE = measupdate(J1, htip, Ttip, pswitch, uponly, state, spinstate, statesize, spinstatesize,
    s2p, ndims, sidlist, nbsid, updatelist, saveupdates);


    energies[bid*nh + hid] = energies[bid*nh + hid] + dE;
  }
}

double measupdate(double J1, double htip, double Ttip, double pswitch, bool uponly, int* state, int* spinstate,
 int statesize, int spinstatesze, int* s2p,int ndims, int* sidlist, int nbsid, int* updatelist,  bool saveupdates){
  double deltaE = 0.0;

  // define distributions
  int sid = 0;
  int s = 0;
  int dnei = 0;
  int sum = 0;
  double ediff =0;
  bool accept = false;
  double r = 0;
  double r1 = 0;

  // define distribution
  uniform_real_distribution<double> double_distrib(0.0, 1.0);
  vector<double> exps(10, 0);
  // pre-compute the exponential:
  if(Ttip !=0){
    for(int sum= -4; sum < 5; sum+=2){
      for(int s = -1; s < 2; s+=2){
        double ediff = -(2*J1*sum - 2*htip)*s;
        // make sure to have a double division:
        exps[sum/2+2+(s+1)/2] = exp(-ediff/(double)Ttip);
      }
    }
  }


  for(int listid = 0; listid < nbsid; listid++){
    accept = false;
    sid = sidlist[listid];
    s = spinstate[sid];
    if((s == -1 && uponly)|| !uponly){
      sum = 0;

      for(int dneiid = 0; dneiid < ndims; dneiid++){
        // notice that s*state -> spinstate : d = s*snei -> snei = s*d
        sum += s*state[s2p[sid*ndims + dneiid]];
      }

      ediff = -(2*sum*J1 - 2*htip)*s;

      // throw a dice:
      r = double_distrib(random_gen());
      if(Ttip == 0) {
        if(ediff < 0 && r < pswitch){
          accept = true;
        }else{
          accept = false;
        }
      }else{
        // compute the acceptance probability
        double p = exps[sum/2+2+(s+1)/2];
        r1 = double_distrib(random_gen());
        // accept or rejects
        if(r<pswitch and r1<p ){
          accept = true;
        }else{
          accept = false;
        }

      }

      if(accept){
        deltaE += -2*J1*(double)sum*s;
        spinstate[sid] *= -1; // flip the spin
        for(int dneiid = 0; dneiid < ndims; dneiid++){
          dnei = s2p[sid*ndims + dneiid];
          state[dnei] *= -1; // flip the corresponding dimers (single spin flip)
        }
        if(saveupdates){
          updatelist[listid] = 1;
        }
      }
    }
  }
  return deltaE;
}

//void measupdate(double J1, double htip, int* states, int statesize,
//int* spinstates, int spinstatesize, int* s2p, int nd, double*){
//
//}
