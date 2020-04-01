#include "magneticmcsevolve.h"
#include "magneticdualworms.h"
#include "manydualworms.h"

using namespace std;
void magneticmcsevolve(double J1, std::vector<std::tuple<double, int*, int, int>> interactions,
  int* states, int statesize, int* spinstates, int spinstatesize,
  int* d_nd, int n_nd, int* d_vd, int n_vd, int* d_wn,
  int* sidlist, int* didlist, int nbit, double* walker2params, double* walker2ids,
  double* energies, int* failedupdates, int nbwalkers, int nmaxiter, int niterworm,
  int nthreads, int nt, int nh) {

     // for loop over the temperatures (to be parallelized)
     #pragma omp parallel for schedule(dynamic, 1) num_threads(nthreads)
     for(int walker = 0; walker < nbwalkers; walker++) {
        // for each temperature, do manydualworms niterworm number of times
        // get the pointer to the tindex element it state
        int *state = &states[walker*statesize];
        int *spinstate = &spinstates[walker*spinstatesize];

        double beta = walker2params[walker*2];
        double h = walker2params[walker*2 + 1];
            //manydual worms
            tuple<double, bool> resultworm = magneticdualworms(J1,
              interactions, h, state, spinstate, statesize, spinstatesize, d_nd, n_nd,
              d_vd, n_vd, d_wn, sidlist, didlist, nbit, beta,
              nmaxiter, niterworm);

            //add things to e
            int bid = walker2ids[walker*2];
            int hid = walker2ids[walker*2+1];

            energies[bid*nh + hid] += get<0>(resultworm);
            if(!get<1>(resultworm)){// if not updated
                failedupdates[bid*nh + hid] += 1;
            }
     }
}
