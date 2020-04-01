#include "updatespinstates.h"
#include <random>
#include "mersenne.h"

using namespace std;

void updatespinstates(int* states, int* spinstates, int* stat_walkers,
	int* sidlist, int* didlist, int nbstat, int statesize,
	int spinstatesize, int nthreads, int nbit, bool randspinstate) {

	#pragma omp parallel for schedule(dynamic,1) num_threads(nthreads)
	for(int resindex = 0; resindex < nbstat; resindex++){
		// for each result index temperature, we are going to update the spinstate
		// first, get the  temperature index so we know where to look in spinstates
		int windex = stat_walkers[resindex];
		// get the pointer to the windex element in state and spinstate
		int *state = &states[windex*statesize];
		int *spinstate = &spinstates[windex*spinstatesize];
		updatespinstate(state, spinstate, sidlist, didlist, nbit, randspinstate);
	}
}



void updatespinstate(int* state, int* spinstate, int* sidlist,
	int* didlist, int nbit, bool randspinstate){
		int s = 1;
		if(randspinstate){
			uniform_int_distribution<int> int_distrib(0, 1);
			s = int_distrib(random_gen())*2 -1;
			spinstate[sidlist[0]] = s;
		}else{// we make sure that the spinstate[sidlist[0]] stays the same
			s = spinstate[sidlist[0]];
		}

	for(int it = 0; it < nbit; it++){
		int sid = sidlist[it+1];
		int dbid = didlist[it];
		s*=state[dbid];
		spinstate[sid] = s;
	}
}
