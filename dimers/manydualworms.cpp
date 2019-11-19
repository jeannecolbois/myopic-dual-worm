#include "manydualworms.h"
#include <random>
#include <cmath>
#include <algorithm>
#include "mersenne.h"
using namespace std;

tuple<double, bool, vector<int>, vector<int>> manydualworms(double J1,
  vector<tuple<double, int*, int, int>> interactions, int* state, int statesize,
  int* d_nd, int n_nd, int* d_vd, int n_vd, int* d_wn, double beta, int saveloops,
  int nmaxiter, int iterworm) {
    /* INITIALISAION */
    // save a copy of the state:
    int *savestate = new int[statesize];
    for( int dim = 0; dim < statesize; dim++){
        savestate[dim] = state[dim];
    }

    // initialise the energy
    double deltaE = 0.0;

    bool loopclosed = true;
    int w1 = 0;
    int w2 = 0;

    vector<int> update(0);
    if (saveloops) {
        for(int db = 0; db < statesize; db++) {
            update.push_back(0);
        }
    }
    vector<int> looplengths(0);
    for(int wormiter = 0; (wormiter < iterworm) && loopclosed; wormiter++ ) { // we only continue if the loops we are building are closed
        tuple<double, int> oneworm = dualworm(&loopclosed, &w1, &w2, J1, interactions, state, statesize, d_nd, n_nd, d_vd, n_vd, d_wn, beta, saveloops, update, nmaxiter);
       	deltaE += get<0>(oneworm);
        if(saveloops){
          	looplengths.push_back(get<1>(oneworm));
        }
    }
    if ((w1%2 == 1 || w2%2 == 1) && loopclosed) { // if we built 10 loops and the winding numbers are odd, we are going to add loops to correct the winding numbers. If we gave up on a loop, we give up.
    	do {
        	tuple<double, int> oneworm = dualworm(&loopclosed, &w1, &w2, J1, interactions, state, statesize, d_nd, n_nd, d_vd, n_vd, d_wn, beta, saveloops, update, nmaxiter);
        	deltaE += get<0>(oneworm);
        	if(saveloops){
            		looplengths.push_back(get<1>(oneworm));
        	}
    	} while ((w1%2 == 1 || w2%2 == 1) && loopclosed); // while the winding numbers are odd and while we are building closed loops. If the loop is not closed, go out and give up.
    }
    vector<int> effective_update(0);

    // if loop not closed
    bool updated = false;
    if(!loopclosed) {
        updated = false;
        deltaE = 0;
        update.clear();
        looplengths.clear();
        for( int dim = 0; dim < statesize; dim++){
            state[dim] = savestate[dim];
        }
    }else{
        updated = true;
        if(saveloops) {
            for (int db=0 ; db < statesize; db++){
                if(update[db]%2 == 1){
                    effective_update.push_back(db);
                }
            }
        }
    }

    //clean up savestate
    delete[] savestate;
    tuple<double, bool, vector<int>, vector<int>> resultingtuple = make_tuple(deltaE, updated, effective_update, looplengths);
    return resultingtuple;
}

tuple<double, int> dualworm(bool* loopclosed, int* w1, int*w2, double J1, vector<tuple<double, int*, int, int>> interactions, int* state, int statesize, int* d_nd, int n_nd, int* d_vd, int n_vd, int* d_wn, double beta, int saveloops, vector<int> &update, int nmaxiter) {

    // max size of the loop = 5* the maximum number of iterations
    int maxiter = nmaxiter*statesize;

    int looplength = 0;

    /* PICK UP A DIMER AT RANDOM AND SAVE ITS NUMBER */
    // define random generator
    // static std::random_device true_random;
    // static std::mt19937_64 mersenne_engine(true_random()); // seed with true random

    // define distribution
    uniform_int_distribution<int> int_distrib(0, statesize-1);

    // select a dimer at random
    int entry_dimer = int_distrib(random_gen());//using random mersenne generator
    int n_dimer = entry_dimer;

    /* LOOP WHILE DIMER =/= FIRST DIMER */
    double deltaE = 0.0;
    int iter = 0;
    do {
        /* GET THE CORRESPONDING V-SITE DIMERS */
        vector<int> v_d(n_vd + 1);
        v_d[0] = n_dimer; // v_d[0] is the dimer index itself
        for(int vnei = 0; vnei < n_vd; vnei++) {
            v_d[vnei + 1] = d_vd[n_dimer*n_vd + vnei]; // =d_vd[n_dimer][vnei], i.e. the indices of the dimers touching through a v-site
        }

        /* COMPUTE THE PROBABILITY TABLE */
        // FIRST: compute the weights
        vector<double> energies(n_vd + 1, 0);
        vector<double> weights(n_vd+1,0);

        for(int id = 0; id < n_vd + 1; id++) {
            //get the dimer corresponding to id
            int dim = v_d[id];

            // flip the two dimers
            state[n_dimer] = -state[n_dimer];
            state[dim] = -state[dim];

            // compute the energy
            double en = 0;
            for(auto d : v_d) {
                en += J1 * state[d]; // J1 contribution
                for(auto couplings: interactions) { // other contributions
                    double J = get<0>(couplings); //coupling
                    int* M = get<1>(couplings); // Matrix of paths of dimers coupled to a given dimer
                    int num_paths = get<2>(couplings); // number of paths
                    int num_nei = get<3>(couplings); // number of dimers/paths

                    // for the dimer d, look at all the neighbours it's connected to and its interactions
                    // more precisely: on each path, perform the product, then sum over the paths

                    for(int path = 0; path <num_paths; path++) {
                        double prod = state[d];
                        for(int nei = 0; nei < num_nei; nei ++) {
                            prod = prod * state[M[(d*num_paths + path) * num_nei + nei]]; //product of dimer values
                        }
                        en += J * prod; //sum over the paths
                    }
                }
            }
            energies[id] = en;
            weights[id] = exp(-beta*(en - energies[0])); // /!\ weights relative to W0 which corresponds to coming back.

            // flip the two dimers again
            state[n_dimer] = -state[n_dimer];
            state[dim] = -state[dim];
        }

        //SECOND: determine whether bounce or zero bounce solution and compute the transition probability "matrix" T
        auto it_weights_max = max_element(weights.begin(), weights.end()); // iterator pointing to the max element
        int id_wmax = distance(weights.begin(), it_weights_max); // index corresponding to the max element
        double wmax = *it_weights_max;
        double sum_wmin = 0;
        for(int id = 0; id < n_vd + 1; id ++) {
            if(id != id_wmax) {
                sum_wmin += weights[id];
            }
        }

        // transition probability
        vector<double> T(n_vd + 1, 0.0);

        if (wmax <= sum_wmin) {
            //zero-bounce
            T[0] = 0.0; // corresponding to the zero-bounce guarantee
            T[1] = 0.5*(weights[0] + weights[1] - weights[2]);
            T[2] = 0.5*(weights[0] + weights[2] - weights[1]);
        }else{
            //one bounce
            if (id_wmax == 0) {
                T[0] = weights[0] - weights[1] - weights[2];
                T[1] = weights[1];
                T[2] = weights[2];
            } else if (id_wmax == 1) {
                T[0] = 0.0;
                T[1] = 1.0;
                T[2] = 0.0;
            } else if (id_wmax == 2) {
                T[0] = 0;
                T[1] = 0;
                T[2] = 1.0;
            }
        }

        /* CHOSE WHICH MOVE TO MAKE RANDOMLY, TAKING THE T MATRIX INTO ACCOUNT */
        // new dimer index: exit dimer
        int e_dimer(0);
        //random parameter
        double r = uniform_real_distribution<double>(0.0, 1.0)(random_gen());

        // make the move and the corresponding updates
        if(r < T[0]) {
            // the exit dimer is the entrance dimer
            e_dimer = n_dimer;
            deltaE += 0.0;
        } else if( r < T[0] + T[1] ) {
            // the exit dimer is the one with index 1 in v_d
            e_dimer = v_d[1];
            deltaE += energies[1] - energies[0];
        } else {
            // the exit dimer is the one with index 2 in v_d
            e_dimer = v_d[2];
            deltaE += energies[2] - energies[0];
        }
        state[n_dimer] = -state[n_dimer];
        state[e_dimer] = -state[e_dimer];

        //save the loops if needed
        if (saveloops) {
            update[n_dimer] += 1;
            update[e_dimer] += 1;
            looplength += 2;
        }


        // check wether these dimers cross the winding lines and update the winding numbers accordingly
        // NOTICE THAT we are only interested in the parity of the winding numbers and we are not actually computing the winding numbers,
        // just checking whether the loop loops an even or odd amount of time. If we were computing the winding numbers, the direction in which the dual bond is
        // crossed should be taken into account.
        *w1 += d_wn[n_dimer*2 + 0]; // first column: 0 if n_dimer not on line 1, 1 if n_dimer on line 1
        *w1 += d_wn[e_dimer*2 + 0];

        *w2 += d_wn[n_dimer*2 + 1]; // second column: 0 if n_dimer not on line 2, 1 if n_dimer on line 2
        *w2 += d_wn[e_dimer*2 + 1];

        /* CHECK WHETHER THE FIRST DIMER IS IN THE N-SITES OF THE LAST DIMER */
        // IF YES: pick it up
        // IF NO: pick a radom one

        // list of the dimers touching e_dimer via an n site:
        vector<int> n_d(n_nd);
        bool found(false);
        for(int nnei = 0; nnei < n_nd; nnei++) {
            n_d[nnei] = d_nd[e_dimer*n_nd + nnei];
            if(n_d[nnei] == entry_dimer) {
                n_dimer = entry_dimer; // if we find the initial dimer in the neighbours, set it as next dimer
                found = true;
            }
        }
        if(e_dimer == entry_dimer) {
            n_dimer = e_dimer;
            found = true;
        }

        // if the loop isn't closed, pick one of the n-site connected dimers at random
        if(!found){
            int id = uniform_int_distribution<int>(0,n_nd-1)(random_gen());
            n_dimer = n_d[id];
        }
        iter++;

    }while(n_dimer != entry_dimer && iter < maxiter);

    if(n_dimer != entry_dimer) {
        deltaE = 0;
        *loopclosed = false;
    }

    tuple<double, int> result = make_tuple(deltaE, looplength);
    return result;
}
