#include "genssfsevolve.h"
#include "ssf_gen.h"

using namespace std;
void genssfsevolve(double J1, std::vector<std::tuple<double, int*, int, int>> interactions,
  int* states, int statesize, int* spinstates, int spinstatesize, int* s2p,
  int nd,  double* walker2params, double* walker2ids, double* energies, int* failedupdates,
  int nbwalkers, int nthreads, int iters, int nt, int nh, bool fullstateupdate) {
    #pragma omp parallel for schedule(dynamic, 1) num_threads(nthreads)
    for(int walker = 0; walker < nbwalkers; walker++) {
       // for each temperature, do manydualworms niterworm number of times
       // get the pointer to the tindex element it state
       int *state = &states[walker*statesize];
       int *spinstate = &spinstates[walker*spinstatesize];

       double beta = walker2params[walker*2];
       double h = walker2params[walker*2 + 1];
       tuple<double, int> result = genssfs(J1, interactions, h, state,
       spinstate, spinstatesize, s2p, nd, beta, iters, fullstateupdate);
       //add things to e
       int bid = walker2ids[walker*2];
       int hid = walker2ids[walker*2 + 1];
       energies[bid*nh + hid] = energies[bid*nh + hid] + get<0>(result);
       failedupdates[bid*nh + hid] += get<1>(result);
    }
}
