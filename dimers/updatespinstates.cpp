#include "updatespinstates.h"
#include <random>
#include "mersenne.h"

using namespace std;

void updatespinstates(int* states, int* spinstates, int* stat_temps,
	int* sidlist, int* didlist, int nbstat, int statesize,
	int spinstatesize, int nthreads, int nbit) {

	#pragma omp parallel for schedule(dynamic,1) num_threads(nthreads)
	for(int resindex = 0; resindex < nbstat; resindex++){
		// for each result index temperature, we are going to update the spinstate
		// first, get the  temperature index so we know where to look in spinstates
		int tindex = stat_temps[resindex];
		// get the pointer to the tindex element in state and spinstate
		int *state = &states[tindex*statesize];
		int *spinstate = &spinstates[tindex*spinstatesize];
		updatespinstate(state, spinstate, sidlist, didlist, nbit);
	}
}



void updatespinstate(int* state, int* spinstate, int* sidlist,
	int* didlist, int nbit){
	uniform_int_distribution<int> int_distrib(0, 1);
	int s = int_distrib(random_gen())*2 -1;
	spinstate[sidlist[0]] = s;
	for(int it = 0; it < nbit; it++){
		int sid = sidlist[it+1];
		int dbid = didlist[it];
		s = s*state[dbid];
		spinstate[sid] = s;
	}
}
