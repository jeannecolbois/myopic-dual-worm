#include "hamiltonian.h"

using namespace std;

//computes the total energy of the system given the dimer state and the interaction paths
double hamiltonian(double J1, vector<tuple<double,int*,int,int>> interactions, int* state, int statesize) {
    //energy due to J1
    double sum1 = 0.0;
    for(int n = 0; n < statesize; n++) {
        sum1 += J1 * state[n];
    }

    //energy due to the other Js
    double sum = 0.0;
    for(auto  couplings : interactions) {
        double J = get<0>(couplings);
        int* M = get<1>(couplings);
        int num_paths = get<2>(couplings);
        int num_nei = get<3>(couplings);

        //for each dimer, look at all the neighbours to which it's connected and its interactions with them
        for(int dim = 0; dim < statesize; dim ++){
            //more specifically: perform the product on each path and sum, taking into account how may dimers are in a given path
            for(int path = 0; path < num_paths; path++) {
                double prod = state[dim];
                for(int nei = 0; nei < num_nei; nei++) {
                    prod = prod * state[M[(dim * num_paths + path) * num_nei + nei]];
                }
                sum += J* prod / (num_nei + 1); // i.e. if we are summing over paths of two dimers, we are taking each path into account twice
            }
        }
    }

    double energy = sum1 + sum;
    return energy; // energy and not energy/site because there is no generic link between the number of dimers and the number of sites
}
