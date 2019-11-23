#include "ssfsevolve.h"
#include "ssf.h"
#include <omp.h>
using namespace std;
void ssfsevolve(double J1, int* states, int statesize, int* spinstates, int spinstatesize, int* s2p,
  int nd,  double* walker2params, double* energies, int* failedupdates,
  int nbwalkers, int nthreads, int iters) {
    #pragma omp parallel for schedule(dynamic, 1) num_threads(nthreads)
    for(int walker = 0; walker < nbwalkers; walker++) {
       // for each temperature, do manydualworms niterworm number of times
       // get the pointer to the tindex element it state
       int *state = &states[walker*statesize];
       int *spinstate = &spinstates[walker*spinstatesize];

       double beta = walker2params[walker*2];
       double h = walker2params[walker*2 + 1];
       tuple<double, int> result = ssfs(J1, h, state,
       spinstate, spinstatesize, s2p, nd, beta, iters);
       //add things to e
       energies[walker] = energies[walker] + get<0>(result);
       failedupdates[walker] += get<1>(result);
    }
}
