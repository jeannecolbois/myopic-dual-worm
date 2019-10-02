
#include <Python.h> // methods in the API
//#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h> // interact with numpy arrays
#include <cmath>

#include "hamiltonian.h"
#include "manydualworms.h"
#include "mcsevolve.h"



using namespace std;

//----------FUNCTIONS TO PARSE A LIST ----------//
//*** GET J1 ***//
double getJ1(PyObject *list_obj) ;

//*** Get TUPLE ***//
PyObject* getTuple(PyObject *list_obj, int index);

//*** PARSE TUPLE ***//

//For J
double parseTupleJ(PyObject *tuple_obj);

// For M
int* parseTupleM(PyObject *tuple_obj, PyObject **PtrM_array);

//*** GET INTERACTIONS ***//
vector<tuple<double, int*, int, int>> getInteractions(PyObject *list_obj, vector<PyObject*>& Marray_list, bool* ok);

///////////////////INCLUDE SOME EXTRA MODULARISATION !!!!!!!!!!!!!!!!!!!!!!!!!!!!///////////////////////////////////



//-----------------------ACTUAL WRAPPING OF THE VARIOUS FUNCTIONS IN A PYTHON MODULE ---------------------------------------//
// This is strongly inspired from "Python modules in C" by Dan Foreman-Mackey : http://dan.iel.fm/posts/python-c-extensions/

/* Docstrings */
static char module_docstring[] =
        "This module provides an interface to work on the dimer models and make a worm update.";
static char hamiltonian_docstring[] =
        "Takes a list of (couplings, dimers related by this coupling) and a state and computes the energy.";
static char manydualworms_docstring[] =
        "Updates a dimers state and performs a myopic worm update which takes into account the interactions of the dimers. Returns the energy difference, as well as some statistics on the loops built";
static char mcsevolve_docstring[] =
	"From a set of states, performs a myopic worm update of the set of dimer states taking into account the dimers interactions and the temperature associated with each state. Returns ... ";
/* Python interfaces to the C++ functions */

// > Everything is a PyObject
// > module_method(pointer to the module instance in python, pointer to the tuple of arguments of the function)
static PyObject* dimers_hamiltonian(PyObject *self, PyObject *args);
static PyObject* dimers_manydualworms(PyObject *self, PyObject *args);
static PyObject* dimers_mcsevolve(PyObject *self, PyObject *args);
/* Module specifications */
static PyMethodDef module_methods[] = {
    {"hamiltonian", dimers_hamiltonian, METH_VARARGS, hamiltonian_docstring},
    {"manydualworms", dimers_manydualworms, METH_VARARGS, manydualworms_docstring},
    {"mcsevolve", dimers_mcsevolve, METH_VARARGS, mcsevolve_docstring},
    {nullptr, nullptr, 0, nullptr}
};

/* Initialize the module */

// I'm pretty sure this function is called when the module is imported into python
// It HAS to be called PyInit_{ModuleName}

PyMODINIT_FUNC PyInit_dimers(void) {
    PyObject *module; // this is the self pointed to earlier

    static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "dimers", // module name
        module_docstring,
        -1, // size of the module state (i.e. sthg like the number of attributes)
        module_methods,
        nullptr,// this and next: fancy stuff, nullptr if not needed
        nullptr,
        nullptr,
        nullptr
    };

    module = PyModule_Create(&moduledef);
    if(!module) return nullptr;

    /* Load numpy functionality */
    import_array();

    return module;
}

/* Now it's time to actually make the translation between C++ and Python */

//****** HAMILTONIAN *******//

static PyObject* dimers_hamiltonian(PyObject *self, PyObject *args) {
    PyObject *list_obj, *state_obj;

    if(!PyArg_ParseTuple(args,"OO", &list_obj, &state_obj)) //take the list and the state as pointers to PyObjects
        return nullptr;

    //----------------------------//
    /* Interpret the list output */
    //----------------------------//
    // check that the list_obj is indeed a list
    if(!PyList_Check(list_obj)) {
        PyErr_Format(PyExc_ValueError,
                         "DIMERS.cpp : Give mea list to work with, I'm line %d", \
                         __LINE__);
        return nullptr;
    }

    // first, get J1 which is the first element of the list
    double J1 = getJ1(list_obj);
    if (std::isnan(J1)) {
        return nullptr;
    }

    // if the list is longer than only one element
    int listsize = PyList_Size(list_obj);
    vector<tuple<double, int*, int, int>> interactions(listsize - 1);
    vector<PyObject*> Marray_list(listsize-1);

    bool ok;
    interactions = getInteractions(list_obj, Marray_list, &ok);
    if(ok == false) return nullptr;



    //---------------------------------//
    /* Interpret the pointer to state */
    //---------------------------------//

    PyObject *state_array = PyArray_FROM_OTF(state_obj, NPY_INT32, NPY_IN_ARRAY);
    if(state_array == nullptr) {
        Py_XDECREF(state_array); // if there was an issue, decrement the reference (i.e. remove a label) if there is one
        PyErr_Format(PyExc_ValueError, "DIMERS.cpp : There was an issue with line %d", __LINE__);
        return nullptr;
    }

    //check that state has the dimensions expected
    if(PyArray_NDIM(state_array) != 1) { // number of dimensions of the "tensor" state array
        PyErr_Format(PyExc_ValueError, "DIMERS.cpp : There was an issue with line %d", __LINE__);
        return nullptr;
    }

    //get the size of the state
    int state_size = (int)PyArray_DIM(state_array, 0); //length of the 0th dimension of the "tensor" state_array, cast as int

    //get pointer as Ctype
    int *state = (int*)PyArray_DATA(state_array);

    //----------------------------------//
    /* Call the hamiltonian C-function */
    //----------------------------------//
    PyThreadState* threadState = PyEval_SaveThread();
    double energy = hamiltonian(J1, interactions, state, state_size);
    PyEval_RestoreThread(threadState);
    /* Clean up */
    Py_DECREF(state_array); // decrement the reference "sum"
    for(auto marray : Marray_list) {
        Py_DECREF(marray); // idem
    }

    /* Build the output */
    PyObject *ret = Py_BuildValue("d", energy);
    return ret;
}

//****** MCSEVOLVE *****//
static PyObject* dimers_mcsevolve(PyObject *self, PyObject *args) {
    /* What we want to get from the arguments:
     * >> A list of couplings [J1, (J2, M2), (J3, M3), ...]
     * >> A table of state objects (numpy table) describing the current states of the systems (table of 1d numpy tables)
     * >> A table of temperatures (numpy)
     * >> A numpy table of energies
     * >> A numpy table of loops to save
     * >> A numpy table of dimers connected to each dimer through an n site
     * >> A numpy table of dimers connected to each dimer through a v site
     * >> A numpy table of which dimers to count for each winding number
     * >> A number of iterations
     * >> A maximal number of loops occupancy */

     //------------------------------------------------------INPUT INTERPRETATION---------------------------------------------------------------------//
     PyObject *list_obj, *states_obj, *betas_obj, *energies_obj, *d_nd_obj, *d_vd_obj, *d_wn_obj; //*saveloops_obj,
     int niterworm;
     int nmaxiter;
     int nthreads;
     // take the arguments as pointers + int
     if(!PyArg_ParseTuple(args,"OOOOOOOiii", &list_obj, &states_obj, &betas_obj, &energies_obj,  &d_nd_obj, &d_vd_obj, &d_wn_obj, &niterworm, &nmaxiter, &nthreads))
	return nullptr;
    //----------------------------//
    /* Interpret the list input  */
    //----------------------------//
    // check that the list_obj is indeed a list
    if(!PyList_Check(list_obj)) {
        PyErr_Format(PyExc_ValueError,
                         "DIMERS.cpp : Give mea list to work with, I'm line %d", \
                         __LINE__);
        return nullptr;
    }

    // first, get J1 which is the first element of the list
    double J1 = getJ1(list_obj);
    if (std::isnan(J1)) {
        return nullptr;
    }

    // if the list is longer than only one element
    int listsize = PyList_Size(list_obj);
    vector<tuple<double, int*, int, int>> interactions(listsize - 1);
    vector<PyObject*> Marray_list(listsize-1);

    bool ok;
    interactions = getInteractions(list_obj, Marray_list, &ok);
    if(ok == false) return nullptr;
     if(!PyList_Check(list_obj)) {
        PyErr_Format(PyExc_ValueError,
                         "DIMERS.cpp : Give mea list to work with, I'm line %d", \
                         __LINE__);
        return nullptr;
     }

    //-------------------------------//
    /* Interpret the table of states */
    //-------------------------------//

    // because we want to be able to update the state as well: NPY_INOUT_ARRAY and not NPY_IN_ARRAY
    PyObject *states_array = PyArray_FROM_OTF(states_obj, NPY_INT32, NPY_INOUT_ARRAY);
    if(states_array == nullptr) {
        Py_XDECREF(states_array);
        PyErr_Format(PyExc_ValueError, "DIMERS.cpp : There was an issue with line %d", __LINE__);
        return nullptr;
    }

    int nt_states = (int)PyArray_DIM(states_array, 0);
    int state_size = (int)PyArray_DIM(states_array, 1);
    //check that state has the dimensions expected
    if(PyArray_NDIM(states_array) != 2) {
        PyErr_Format(PyExc_ValueError, "DIMERS.cpp : There was an issue with line %d", __LINE__);
        return nullptr;
    }

    //get pointer as Ctype
    int *states = (int*)PyArray_DATA(states_array);

    //--------------------------------------//
    /* Interpret the table of temperatures */
    //--------------------------------------//

    // because we want to be able to update the state as well: NPY_INOUT_ARRAY and not NPY_IN_ARRAY
    PyObject *betas_array = PyArray_FROM_OTF(betas_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    if(betas_array == nullptr) {
        Py_XDECREF(betas_array);
        PyErr_Format(PyExc_ValueError, "DIMERS.cpp : There was an issue with line %d", __LINE__);
        return nullptr;
    }

    int ntb = (int)PyArray_DIM(betas_array, 0);
    //check that state has the dimensions expected
    if(PyArray_NDIM(betas_array) != 1) {
        PyErr_Format(PyExc_ValueError, "DIMERS.cpp : There was an issue with line %d", __LINE__);
        return nullptr;
    }

    if(nt_states != ntb) {
        PyErr_Format(PyExc_ValueError, "DIMERS.cpp : Non-consistent number of temperatures and states, see line %d", __LINE__);
        return nullptr;
    }
    //get pointer as Ctype
    double *betas = (double*)PyArray_DATA(betas_array);

    //---------------------------------//
    /* Interpret the table of energies */
    //---------------------------------//

    PyObject *energies_array = PyArray_FROM_OTF(energies_obj, NPY_DOUBLE, NPY_INOUT_ARRAY); //INOUT: we want to update the energies
    if(energies_array == nullptr) {
        Py_XDECREF(energies_array);
        PyErr_Format(PyExc_ValueError, "DIMERS.cpp : There was an issue with line %d", __LINE__);
        return nullptr;
    }

    int nte = (int)PyArray_DIM(energies_array, 0);
    //check that state has the dimensions expected
    if(PyArray_NDIM(energies_array) != 1) {
        PyErr_Format(PyExc_ValueError, "DIMERS.cpp : There was an issue with line %d", __LINE__);
        return nullptr;
    }

    if(nte != ntb) {
        PyErr_Format(PyExc_ValueError, "DIMERS.cpp : Non-consistent number of energies and states, see line %d", __LINE__);
        return nullptr;
    }

    //get pointer as Ctype
    double *energies = (double*)PyArray_DATA(energies_array);


    //-----------------------------------------//
    /* Interpret the pointers to dimer tables */
    //-----------------------------------------//

    PyObject *d_nd_array = PyArray_FROM_OTF(d_nd_obj, NPY_INT32, NPY_IN_ARRAY); // dimers to dimers via n-sites
    PyObject *d_vd_array = PyArray_FROM_OTF(d_vd_obj, NPY_INT32, NPY_IN_ARRAY); // dimers to dimers via v-sites

    if(d_nd_array == nullptr || d_vd_array == nullptr) {
        Py_XDECREF(d_nd_array);
        Py_XDECREF(d_vd_array);
        PyErr_Format(PyExc_ValueError, "DIMERS.cpp : There was an issue with line %d", __LINE__);
        return nullptr;
    }

    //check that they have the dimensions expected
    if(PyArray_NDIM(d_nd_array) != 2) {
        Py_XDECREF(d_nd_array);
        Py_XDECREF(d_vd_array);
        PyErr_Format(PyExc_ValueError, "DIMERS.cpp : There was an issue with line %d", __LINE__);
        return nullptr;
    }
    if(PyArray_NDIM(d_vd_array) != 2) {
        Py_XDECREF(d_nd_array);
        Py_XDECREF(d_vd_array);
        PyErr_Format(PyExc_ValueError, "DIMERS.cpp : There was an issue with line %d", __LINE__);
        return nullptr;
    }

    // get the number of neighbours
    int nnei_d_nd = (int)PyArray_DIM(d_nd_array, 1);
    int nnei_d_vd = (int)PyArray_DIM(d_vd_array, 1);

    //prevent any big issue (the algorithm works only for systems with two exit dimers)
    if(nnei_d_vd != 2) {
        Py_XDECREF(d_nd_array);
        Py_XDECREF(d_vd_array);
        PyErr_Format(PyExc_ValueError, "DIMERS.cpp : There was an issue with line %d", __LINE__);
        return nullptr;
    }

    //get pointer as Ctype
    int *d_nd = (int*)PyArray_DATA(d_nd_array);
    int *d_vd = (int*)PyArray_DATA(d_vd_array);

    //------------------------------------------------//
    /* Interpret the pointer to winding number table */
    //------------------------------------------------//
    // get the pointer as numpy array
    PyObject *d_wn_array = PyArray_FROM_OTF(d_wn_obj, NPY_INT32, NPY_IN_ARRAY);

    if(d_wn_array == nullptr) {
        Py_XDECREF(d_wn_array);
        PyErr_Format(PyExc_ValueError, "DIMERS.cpp : There was an issue with line %d", __LINE__);
        return nullptr;
    }

    //check the dimensions
    if(PyArray_NDIM(d_wn_array) != 2) {
        Py_XDECREF(d_wn_array);
        PyErr_Format(PyExc_ValueError, "DIMERS.cpp : There was an issue with line %d", __LINE__);
        return nullptr;
    }

    //get pointer as Ctype
    int *d_wn = (int*)PyArray_DATA(d_wn_array);

    //------------------------------------------------------FUNCTION CALL---------------------------------------------------------------------//

    //------------------------------------//
    /* Call the manydualworms C function */
    //------------------------------------//
    PyThreadState* threadState = PyEval_SaveThread(); // release the GIL
    mcsevolve(J1, interactions, states, state_size, d_nd, nnei_d_nd, d_vd, nnei_d_vd, d_wn, betas, energies, ntb, nmaxiter, niterworm, nthreads); //saveloops = 0
    PyEval_RestoreThread(threadState); // claim the GIL

    // Clean up
    Py_DECREF(states_array);
    Py_DECREF(betas_array);
    //Py_DECREF(saveloops_array);
    Py_DECREF(d_nd_array);
    Py_DECREF(d_vd_array);
    Py_DECREF(d_wn_array);
    for(auto marray : Marray_list) { // clean up as well the array(s) created for the couplings in getInteractions
        Py_DECREF(marray);
    }
    //build output

    /*PyObject *effective_update_list = PyList_New(effective_update.size()); //list of dual bonds, i.e. list of int
    for (int i = 0; i < effective_update.size(); i++) {
        PyObject *dualbond = Py_BuildValue("i", effective_update[i]);
        PyList_SetItem(effective_update_list, i, dualbond);
    }

    PyObject *looplengthslist = PyList_New(looplengths.size()); //list of length of loops, i.e. list of int
    for (int i = 0; i < looplengths.size(); i++) {
        PyObject *looplength = Py_BuildValue("i", looplengths[i]);
        PyList_SetItem(looplengthslist, i, looplength);
    }

    /* Build the output */
    PyObject *ret = Py_BuildValue("d", 0);
    return ret;
}

//****** MANYDUALWORMS *******//
static PyObject* dimers_manydualworms(PyObject *self, PyObject *args) {
    /* What we want to get from the arguments:
     * >> A list of couplings [J1, (J2,M2), (J3,M3),...]
     * >> A state object (numpy table) describing the current state of the system
     * >> A numpy table of dimers connected to each dimer through an n site
     * >> A numpy table of dimers connected to each dimer through a v site
     * >> A numpy table of which dimers to count for each winding number
     * >> A temperature */

    PyObject *list_obj, *state_obj, *d_nd_obj, *d_vd_obj, *d_wn_obj;
    double beta;
    int saveloops;
    int nmaxiter;
    int iterworm;

    //take the five arguments as pointers to PyObjects + the temperature as a double
    if(!PyArg_ParseTuple(args,"OOOOOdpii", &list_obj, &state_obj, &d_nd_obj, &d_vd_obj, &d_wn_obj, &beta, &saveloops, &nmaxiter, &iterworm))
        return nullptr;


    //----------------------------//
    /* Interpret the list output */
    //----------------------------//
    // check that the list_obj is indeed a list

    if(!PyList_Check(list_obj)) {
        PyErr_Format(PyExc_ValueError,
                         "DIMERS.cpp : Give mea list to work with, I'm line %d", \
                         __LINE__);
        return nullptr;
    }

    // first, get J1 which is the first element of the list
    double J1 = getJ1(list_obj);
    if (std::isnan(J1)) {
        return nullptr;
    }

    // if the list is longer than only one element
    int listsize = PyList_Size(list_obj);
    vector<tuple<double, int*, int, int>> interactions(listsize - 1);
    vector<PyObject*> Marray_list(listsize-1);

    bool ok;
    interactions = getInteractions(list_obj, Marray_list, &ok);
    if(ok == false) return nullptr;

    //---------------------------------//
    /* Interpret the pointer to state */
    //---------------------------------//

    // because we want to be able to update the state as well: NPY_INOUT_ARRAY and not NPY_IN_ARRAY
    PyObject *state_array = PyArray_FROM_OTF(state_obj, NPY_INT32, NPY_INOUT_ARRAY);
    if(state_array == nullptr) {
        Py_XDECREF(state_array);
        PyErr_Format(PyExc_ValueError, "DIMERS.cpp : There was an issue with line %d", __LINE__);
        return nullptr;
    }

    int state_size = (int)PyArray_DIM(state_array, 0);
    //check that state has the dimensions expected
    if(PyArray_NDIM(state_array) != 1) {
        PyErr_Format(PyExc_ValueError, "DIMERS.cpp : There was an issue with line %d", __LINE__);
        return nullptr;
    }

    //get pointer as Ctype
    int *state = (int*)PyArray_DATA(state_array);

    //-----------------------------------------//
    /* Interpret the pointers to dimer tables */
    //-----------------------------------------//

    PyObject *d_nd_array = PyArray_FROM_OTF(d_nd_obj, NPY_INT32, NPY_IN_ARRAY); // dimers to dimers via n-sites
    PyObject *d_vd_array = PyArray_FROM_OTF(d_vd_obj, NPY_INT32, NPY_IN_ARRAY); // dimers to dimers via v-sites

    if(d_nd_array == nullptr || d_vd_array == nullptr) {
        Py_XDECREF(d_nd_array);
        Py_XDECREF(d_vd_array);
        PyErr_Format(PyExc_ValueError, "DIMERS.cpp : There was an issue with line %d", __LINE__);
        return nullptr;
    }

    //check that they have the dimensions expected
    if(PyArray_NDIM(d_nd_array) != 2) {
        Py_XDECREF(d_nd_array);
        Py_XDECREF(d_vd_array);
        PyErr_Format(PyExc_ValueError, "DIMERS.cpp : There was an issue with line %d", __LINE__);
        return nullptr;
    }
    if(PyArray_NDIM(d_vd_array) != 2) {
        Py_XDECREF(d_nd_array);
        Py_XDECREF(d_vd_array);
        PyErr_Format(PyExc_ValueError, "DIMERS.cpp : There was an issue with line %d", __LINE__);
        return nullptr;
    }

    // get the number of neighbours
    int nnei_d_nd = (int)PyArray_DIM(d_nd_array, 1);
    int nnei_d_vd = (int)PyArray_DIM(d_vd_array, 1);

    //prevent any big issue (the algorithm works only for systems with two exit dimers)
    if(nnei_d_vd != 2) {
        Py_XDECREF(d_nd_array);
        Py_XDECREF(d_vd_array);
        PyErr_Format(PyExc_ValueError, "DIMERS.cpp : There was an issue with line %d", __LINE__);
        return nullptr;
    }

    //get pointer as Ctype
    int *d_nd = (int*)PyArray_DATA(d_nd_array);
    int *d_vd = (int*)PyArray_DATA(d_vd_array);

    //------------------------------------------------//
    /* Interpret the pointer to winding number table */
    //------------------------------------------------//
    // get the pointer as numpy array
    PyObject *d_wn_array = PyArray_FROM_OTF(d_wn_obj, NPY_INT32, NPY_IN_ARRAY);

    if(d_wn_array == nullptr) {
        Py_XDECREF(d_wn_array);
        PyErr_Format(PyExc_ValueError, "DIMERS.cpp : There was an issue with line %d", __LINE__);
        return nullptr;
    }

    //check the dimensions
    if(PyArray_NDIM(d_wn_array) != 2) {
        Py_XDECREF(d_wn_array);
        PyErr_Format(PyExc_ValueError, "DIMERS.cpp : There was an issue with line %d", __LINE__);
        return nullptr;
    }

    //get pointer as Ctype
    int *d_wn = (int*)PyArray_DATA(d_wn_array);

    //------------------------------------//
    /* Call the manydualworms C function */
    //------------------------------------//
    PyThreadState* threadState = PyEval_SaveThread();
    tuple<double, vector<int>, vector<int>> resultworm = manydualworms(J1, interactions, state, state_size, d_nd, nnei_d_nd, d_vd, nnei_d_vd, d_wn, beta, saveloops, nmaxiter, iterworm);
    PyEval_RestoreThread(threadState);
    double deltaE = get<0>(resultworm);
    vector<int> effective_update = get<1>(resultworm);
    vector<int> looplengths = get<2>(resultworm);

    // Clean up
    Py_DECREF(state_array);
    Py_DECREF(d_nd_array);
    Py_DECREF(d_vd_array);
    Py_DECREF(d_wn_array);
    for(auto marray : Marray_list) { // clean up as well the array(s) created for the couplings in getInteractions
        Py_DECREF(marray);
    }
    //build output

    PyObject *effective_update_list = PyList_New(effective_update.size()); //list of dual bonds, i.e. list of int
    for (int i = 0; i < effective_update.size(); i++) {
        PyObject *dualbond = Py_BuildValue("i", effective_update[i]);
        PyList_SetItem(effective_update_list, i, dualbond);
    }

    PyObject *looplengthslist = PyList_New(looplengths.size()); //list of length of loops, i.e. list of int
    for (int i = 0; i < looplengths.size(); i++) {
        PyObject *looplength = Py_BuildValue("i", looplengths[i]);
        PyList_SetItem(looplengthslist, i, looplength);
    }

    PyObject *ret = Py_BuildValue("dOO", deltaE, effective_update_list, looplengthslist);
    return ret;
}

//----------FUNCTIONS TO PARSE A LIST OF COUPLINGS----------//

//*** GET J1 ***//
double getJ1(PyObject *list_obj) {
    PyObject *J1_obj = PyList_GetItem(list_obj, 0);

    double J1 = PyFloat_AsDouble(J1_obj);
    if (PyErr_Occurred()) {
        PyErr_Format(PyExc_ValueError, "DIMERS.cpp : There was an issue with line %d", __LINE__);
        return NAN;
    }

    return J1;
}

//*** Get TUPLE ***//
PyObject* getTuple(PyObject *list_obj, int index) {
    // get the tuple
    PyObject *tuple_obj = PyList_GetItem(list_obj, index);

    // check that no error occured
    if (PyErr_Occurred()) {
        PyErr_Format(PyExc_ValueError, "DIMERS.cpp : There was an issue with line %d", __LINE__);
        return nullptr;
    }

    // chekc that it is a tuple
    if(!PyTuple_Check(tuple_obj)) {
        PyErr_Format(PyExc_ValueError, "DIMERS.cpp : There was an issue with line %d", __LINE__);
        return nullptr;
    }

    //check that it is what is expected by the remainder of the program
    int tuple_size = PyTuple_Size(tuple_obj);
    if(tuple_size != 2) {
        PyErr_Format(PyExc_ValueError, "DIMERS.cpp : There was an issue with line %d", __LINE__);
        return nullptr;
    }

    return tuple_obj;
}

//*** PARSE TUPLE ***//

//For J
double parseTupleJ(PyObject *tuple_obj) {

    // get the object corresponding to the coupling
    PyObject *J_obj = PyTuple_GetItem(tuple_obj, 0);
    //get the corresponding double
    double J = PyFloat_AsDouble(J_obj);

    // check for errors
    if (PyErr_Occurred()) {
        PyErr_Format(PyExc_ValueError, "DIMERS.cpp : There was an issue with line %d", __LINE__);
        return NAN;
    }

    return J;
}

// For M
int* parseTupleM(PyObject *tuple_obj, PyObject **PtrM_array) {
    // Get the M_object
    PyObject *M_obj = PyTuple_GetItem(tuple_obj, 1);
    // check for issues
    if(M_obj == nullptr){
        PyErr_Format(PyExc_ValueError, "DIMERS.cpp : There was an issue with line %d", __LINE__);
        return nullptr;
    }

    //interpret as numpy array
    *PtrM_array = PyArray_FROM_OTF(M_obj, NPY_INT32, NPY_IN_ARRAY); //obj (a_obj), typenum (int),  requirements (C-contiguous)
    if(*PtrM_array == nullptr) {
        Py_XDECREF(*PtrM_array); //decrease the allocation pile if not already null
        PyErr_Format(PyExc_ValueError, "DIMERS.cpp : There was an issue with line %d", __LINE__);
        return nullptr;
    }

    // check the dimensions
    int nDim = PyArray_NDIM(*PtrM_array);
    if(nDim!=3) {
        PyErr_Format(PyExc_ValueError, "DIMERS.cpp : There was an issue with line %d", __LINE__);
        return nullptr;
    }

    //get pointer to the data as C-type and return it
    int *M = (int*)PyArray_DATA(*PtrM_array);

    return M;
}

//*** GET INTERACTIONS ***//
vector<tuple<double, int*, int, int>> getInteractions(PyObject *list_obj, vector<PyObject*>& Marray_list, bool* ok) {
    *ok = false;
    int listsize = PyList_Size(list_obj);
    vector<tuple<double, int*, int, int>> interactions(listsize-1);

    for(int l = 1; l < listsize; l++) {
        /* obtain the list tuple */
        PyObject *tuple_obj = getTuple(list_obj, l);
        if(!tuple_obj) return interactions;

        /* parse the tuple */
        // for J
        double J = parseTupleJ(tuple_obj);
        if(std::isnan(J)) return interactions;
        // for M
        int *M = parseTupleM(tuple_obj, &Marray_list[l - 1]); // send the pointer to the pointer to the array
        if(!M) return interactions;
        //get lengths
        int npaths = (int)PyArray_DIM(Marray_list[l - 1], 1);
        int nnei = (int)PyArray_DIM(Marray_list[l - 1], 2);

        /* put everything in the vector that will be passed to Hamiltonian */
        tuple<double, int*, int, int> tuple_topass(J, M, npaths, nnei);
        interactions[l - 1] = tuple_topass;
    }

    *ok = true;
    return interactions;
}


//*** GET LIST ***//
/*tuple<double, int, vector<tuple<double, int*, int, int>>>  getList(list_obj) {
    //----------------------------//
    /* Interpret the list input  */
    //----------------------------//
    // check that the list_obj is indeed a list
/*    if(!PyList_Check(list_obj)) {
         PyErr_Format(PyExc_ValueError,
                         "DIMERS.cpp : Give mea list to work with, I'm line %d", \
                         __LINE__);
        return nullptr;
    }

    // first, get J1 which is the first element of the list
    double J1 = getJ1(list_obj);
    if (std::isnan(J1)) {
        return nullptr;
    }

    // if the list is longer than only one element
    int listsize = PyList_Size(list_obj);
    vector<tuple<double, int*, int, int>> interactions(listsize - 1);
    vector<PyObject*> Marray_list(listsize-1);

    bool ok;
    interactions = getInteractions(list_obj, Marray_list, &ok);
    if(ok == false) return nullptr;
     if(!PyList_Check(list_obj)) {
        PyErr_Format(PyExc_ValueError,
                         "DIMERS.cpp : Give mea list to work with, I'm line %d", \
                         __LINE__);
        return nullptr;
     }

    tuple<double, int, vector<tuple<double, int*, int, int>>> tuple_topass(J1, listsize, interactions);

    return tuple_topass;

}*/
