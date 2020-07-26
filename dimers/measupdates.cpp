#include "measupdates.h"
#include <random>
#include "mersenne.h"
#include <omp.h>


using namespace std;
void measupdates(double J1, double htip, double Ttip, int* states, int statesize,
 int* spinstates, int spinstatesize, int* s2p, int ndims, int* sidlist, int nbsid,
 int* walker2ids, double* energies, int nbwalkers, int nthreads, int nt,
 int nh) {
  #pragma omp parallel for schedule(dynamic, 1) num_threads(nthreads)

  for(int walker = 0; walker < nbwalkers; walker++ ){
  // for each walker, measupdates
    int *state = &states[walker*statesize];
    int *spinstate =  &spinstates[walker*spinstatesize];

    double dE = measupdate(J1, htip, Ttip, state, spinstate, statesize, spinstatesize,
    s2p, ndims, sidlist, nbsid);

    int bid = walker2ids[walker*2];
    int hid = walker2ids[walker*2 + 1];
    energies[bid*nh + hid] = energies[bid*nh + hid] + dE;
  }
}

double measupdate(double J1, double htip, double Ttip, int* state, int* spinstate,
 int statesize, int spinstatesze, int* s2p,int ndims, int* sidlist, int nbsid){
   double deltaE = 0.0;

   // define distributions
   int sid = 0;
   int s = 0;
   int dnei = 0;
   int sum = 0;
   bool accept = false;

   // define distribution
   uniform_real_distribution<double> double_distrib(0.0, 1.0);
   vector<double> exps(9, 0);
   // pre-compute the exponential:
   if(Ttip !=0){
     for(int sumval = 0; sumval < 5; sumval++){
       double ediff = 2*J1*(2*(sumval - 2)) - 2*htip;
       exps[sumval] = exp(-ediff/Ttip);
     }
   }
   for(int listid = 0; listid < nbsid; listid++){
     accept = false;
     sid = sidlist[listid];
     s = spinstate[sid];
     if(s==-1){ // chance at flipping
       sum = 0;
       for(int dneiid = 0; dneiid < ndims; dneiid++){
          sum -= state[s2p[sid*ndims + dneiid]]; // d = s*snei -> snei = -d
       }

       if(Ttip == 0) {
         if(sum*J1 < htip){
          // flip
          accept = true;
         }
       }else{
         // throw a dice
         double r = double_distrib(random_gen());
         // compute the acceptance probability
         double p = exps[sum/2+2];
         // accept or rejects
         if(r<p){
           accept = true;
         }
       }

       if(accept){
         deltaE += 2*J1*sum;
         spinstate[sid] *= -1;
         for(int dneiid = 0; dneiid < ndims; dneiid++){
            dnei = s2p[sid*ndims + dneiid]; // d = s*snei -> snei = -d
            state[dnei] *= -1;
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
