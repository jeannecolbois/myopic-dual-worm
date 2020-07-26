#ifndef MEAS_H
#define MEAS_H
#include <omp.h>


void measupdates(double J1, double htip, double Ttip, int* states, int statesize,
 int* spinstates, int spinstatesize, int* s2p, int ndims, int* sidlist, int nbsid,
 int* walker2ids, double* energies, int nbwalkers, int nthreads, int nt,
 int nh);
double measupdate(double J1, double htip, double Ttip, int* state, int* spinstate,
 int statesize, int spinstatesze, int* s2p,int ndims, int* sidlist, int nbsid);

 #endif //MEAS_H
