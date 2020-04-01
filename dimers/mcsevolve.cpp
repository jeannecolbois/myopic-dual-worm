#include "mcsevolve.h"
#include "manydualworms.h"
using namespace std;

void mcsevolve(double J1, std::vector<std::tuple<double, int*, int, int>> interactions,
  int* states, int statesize, int* d_nd, int n_nd, int* d_vd, int n_vd,
  int* d_wn, double *betas, double *energies,int* failedupdates, int nbt, int nmaxiter, int niterworm,
  int nthreads) {

     // for loop over the temperatures (to be parallelized)
     #pragma omp parallel for schedule(dynamic, 1) num_threads(nthreads)
     for(int tindex = 0; tindex < nbt; tindex++) {
        // for each temperature, do manydualworms niterworm number of times
        // get the pointer to the tindex element it state
        int *state = &states[tindex*statesize];
        double beta = betas[tindex];
            //manydual worms
            tuple<double, bool> resultworm = manydualworms(J1,
              interactions, state, statesize, d_nd, n_nd, d_vd, n_vd, d_wn, beta, nmaxiter, niterworm);
            energies[tindex] = energies[tindex] + get<0>(resultworm);
            //add things to e
            if(!get<1>(resultworm)){
                failedupdates[tindex] += 1;
            }
     }
}
