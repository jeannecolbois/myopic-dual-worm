#include "ssfsevolve.h"
#include "ssf.h"
#include <omp.h>
using namespace std;
void ssfsevolve(double J1,
  double h, int* states, int statesize, int* spinstates, int spinstatesize, int* s2p,
  int nd, double* betas, double* energies, int* failedupdates, int nbt, int nthreads, int iters) {
    #pragma omp parallel for schedule(dynamic, 1) num_threads(nthreads)
    for(int tindex = 0; tindex < nbt; tindex++) {
       // for each temperature, do manydualworms niterworm number of times
       // get the pointer to the tindex element it state
       int *state = &states[tindex*statesize];
       int *spinstate = &spinstates[tindex*spinstatesize];
       double beta = betas[tindex];
           tuple<double, int> result = ssfs(J1, h, state,
           spinstate, spinstatesize, s2p, nd, beta, iters);
           //add things to e
           energies[tindex] = energies[tindex] + get<0>(result);
           failedupdates[tindex] += get<1>(result);
    }
}
