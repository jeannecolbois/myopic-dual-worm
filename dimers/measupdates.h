#ifndef MEAS_H
#define MEAS_H
#include <omp.h>


void measupdates(double J1, double htip, double Ttip, double pswitch, bool uponly, int* states, int statesize,
 int* spinstates, int spinstatesize, int* s2p, int ndims, int* sidlist, int nbsid,
 int* walker2ids, double* energies, int nbwalkers, int nthreads, int nt,
 int nh, int* updatelists, bool saveupdates);
double measupdate(double J1, double htip, double Ttip, double pswitch, bool uponly, int* state, int* spinstate,
 int statesize, int spinstatesze, int* s2p,int ndims, int* sidlist, int nbsid, int* updatelist, bool saveupdates);

 #endif //MEAS_H
