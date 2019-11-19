#include "magneticmcsevolve.h"
#include "magneticdualworms.h"
#include "manydualworms.h"

using namespace std;
void magneticmcsevolve(double J1, vector<tuple<double, int*, int, int>> interactions,
  double h, int* states, int statesize, int* spinstates, int spinstatesize,
  int* d_nd, int n_nd, int* d_vd, int n_vd, int* d_wn,
  int* sidlist, int* didlist, int nbit, double *betas,
  double *energies, int* failedupdates, int nbt, int nmaxiter, int niterworm, int nthreads) {

     // for loop over the temperatures (to be parallelized)
     #pragma omp parallel for schedule(dynamic, 1) num_threads(nthreads)
     for(int tindex = 0; tindex < nbt; tindex++) {
        // for each temperature, do manydualworms niterworm number of times
        // get the pointer to the tindex element it state
        int *state = &states[tindex*statesize];
        int *spinstate = &spinstates[tindex*spinstatesize];
        double beta = betas[tindex];
        int saveloops = 0; // don't save loops
            //manydual worms
            tuple<double, bool, vector<int>, vector<int>> resultworm = magneticdualworms(J1,
              interactions, h, state, spinstate, statesize, spinstatesize, d_nd, n_nd,
              d_vd, n_vd, d_wn, sidlist, didlist, nbit, beta,
              saveloops, nmaxiter, niterworm);
            //add things to e
            energies[tindex] = energies[tindex] + get<0>(resultworm);
            if(!get<1>(resultworm)){// if not updated
                failedupdates[tindex] += 1;
            }
     }
}
