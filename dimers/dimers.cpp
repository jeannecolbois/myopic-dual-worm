#include <Python.h> // methods in the API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/arrayobject.h" // interact with numpy arrays
#include <cmath>

#include "hamiltonian.h"
//#include "manydualworms.h"
//#include "magneticdualworms.h"
#include "mcsevolve.h"
#include "magneticmcsevolve.h"
#include "updatespinstates.h"
#include "measupdate.h"
#include "ssfsevolve.h"



using namespace std;

//----------FUNCTIONS TO PARSE AN INTERACTIONS LIST ----------//
//*** GET J1 ***//
double getJ1(PyObject *list_obj) ;
//*** Get TUPLE ***//
PyObject* getTuple(PyObject *list_obj, int index);
//*** PARSE TUPLE ***//
//For J
double parseTupleJ(PyObject *tuple_obj);
// For M
int* parseTupleM(PyObject *tuple_obj, PyArrayObject **PtrM_array);
//*** GET INTERACTIONS ***//
vector<tuple<double, int*, int, int>> getInteractions(PyObject *list_obj, vector<PyArrayObject*>& Marray_list, bool* ok);

//----------FUNCTION TO PARSE STATES AND SPINSTATES --------//
tuple<int*, int, int> parseStates(PyArrayObject* states_array, PyObject* states_obj);
//----------FUNCTIONS TO PARSE 1D HELPER LISTS --------------//
tuple<int*, int> parseInteger1DArray(PyArrayObject* oned_array, PyObject* oned_pyobject);

//----------FUNCTIONS TO PARSE 2D HELPER LISTS --------------//
tuple<int*, int, int> parseInteger2DArray(PyArrayObject* twod_array, PyObject* twod_obj);

///////////////////INCLUDE SOME EXTRA MODULARISATION !!!!!!!!!!!!!!!!!!!!!!!!!!!!///////////////////////////////////



//-----------------------ACTUAL WRAPPING OF THE VARIOUS FUNCTIONS IN A PYTHON MODULE ---------------------------------------//
// This is strongly inspired from "Python modules in C" by Dan Foreman-Mackey : http://dan.iel.fm/posts/python-c-extensions/

/* Docstrings */
static char module_docstring[] =
        "This module provides an interface to work on the dimer models and make a worm update.";
static char hamiltonian_docstring[] =
        "Takes a list of (couplings, dimers related by this coupling) and a state and computes the energy.";
// static char manydualworms_docstring[] =
//         "Updates a dimers state and performs a myopic worm update which takes into account the interactions of the dimers. Returns the energy difference, as well as some statistics on the loops built";
static char mcsevolve_docstring[] =
	"From a set of states, performs a myopic worm update of the set of dimer states taking into account the dimers interactions and the temperature associated with each state. Returns 0 if success ";
static char magneticmcsevolve_docstring[] =
  	"From a set of states and spinstates, performs a myopic worm update of the set of dimer states taking into account the dimers interactions and the temperature associated with each state, accepts or rejects based on magnetic field. Returns 0 if success ";
static char ssfsevolve_docstring[] =
      	"From a set of states and spinstates, performs a single spin flip update of the set of dimer states taking into account the dimers interactions and the temperature associated with each state, accepts or rejects based on magnetic field and J1 interaction. Returns 0 if success ";
static char updatespinstates_docstring[] =
	"From a set of states, updates the spinstates given the connectivity ";
static char measupdates_docstring[] =
  	"From a set of states, updates the spinstates as a magnetic tip would ";

/* Python interfaces to the C++ functions */

// > Everything is a PyObject
// > module_method(pointer to the module instance in python, pointer to the tuple of arguments of the function)
static PyObject* dimers_hamiltonian(PyObject *self, PyObject *args);
//static PyObject* dimers_manydualworms(PyObject *self, PyObject *args);
static PyObject* dimers_mcsevolve(PyObject *self, PyObject *args);
static PyObject* dimers_magneticmcsevolve(PyObject *self, PyObject *args);
static PyObject* dimers_ssfsevolve(PyObject *self, PyObject *args);
static PyObject* dimers_updatespinstates(PyObject *self, PyObject *args);
static PyObject* dimers_measupdates(PyObject *self, PyObject *args);
/* Module specifications */
static PyMethodDef module_methods[] = {
    {"hamiltonian", dimers_hamiltonian, METH_VARARGS, hamiltonian_docstring},
    {"mcsevolve", dimers_mcsevolve, METH_VARARGS, mcsevolve_docstring},
    {"magneticmcsevolve", dimers_magneticmcsevolve, METH_VARARGS, magneticmcsevolve_docstring},
    {"ssfsevolve", dimers_ssfsevolve, METH_VARARGS, ssfsevolve_docstring},
    {"updatespinstates", dimers_updatespinstates, METH_VARARGS, updatespinstates_docstring},
    {"measupdates", dimers_measupdates, METH_VARARGS, measupdates_docstring},
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
    PyObject *list_obj;
    PyObject *state_obj;

    if(!PyArg_ParseTuple(args,"OO", &list_obj, &state_obj))
        return nullptr;

    //----------------------------//
    /* Interpret the list output */
    //----------------------------//
    // check that the list_obj is indeed a list
    if(!PyList_Check(list_obj)) {
        PyErr_Format(PyExc_ValueError,
                         "DIMERS.cpp : Give me a list to work with, I'm line %d", \
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
    vector<PyArrayObject*> Marray_list(listsize-1);

    bool ok;
    interactions = getInteractions(list_obj, Marray_list, &ok);
    if(ok == false) {
      for(auto marray : Marray_list) {
          Py_DECREF(marray); // idem
      }
      PyErr_Format(PyExc_ValueError, "DIMERS.cpp : There was an issue with line %d (NPY array?)", __LINE__);
      return nullptr;
    }
    //---------------------------------//
    /* Interpret the pointer to state */
    //---------------------------------//
    PyArrayObject* state_array = nullptr;
    tuple<int*, int> statetuple = parseInteger1DArray(state_array, state_obj);
    int* state = get<0>(statetuple);
    if(state == nullptr) {
        Py_XDECREF(state_array);
        PyErr_Format(PyExc_ValueError, "DIMERS.cpp : There was an issue with line %d (NPY array?)", __LINE__);
        return nullptr;
    }
    int statesize = get<1>(statetuple);

    //----------------------------------//
    /* Call the hamiltonian C-function */
    //----------------------------------//
    PyThreadState* threadState = PyEval_SaveThread();
    double energy = hamiltonian(J1, interactions, state, statesize);
    PyEval_RestoreThread(threadState);
    /* Clean up */
    if( state_array != nullptr){
      Py_DECREF(state_array); // decrement the reference
    }
    //else{
    //  PyErr_Format(PyExc_ValueError, "DIMERS.cpp : state_array is null (NPY array?)", __LINE__);
    //return nullptr;

    for(auto marray : Marray_list) {
        Py_DECREF(marray); // idem
    }

    /* Build the output */
    PyObject *ret = Py_BuildValue("d", energy);
    return ret;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

//****** MCSEVOLVE *****//
static PyObject* dimers_mcsevolve(PyObject *self, PyObject *args) {
    /* What we want to get from the arguments:
     * >> A list of couplings [J1, (J2, M2), (J3, M3), ...]
     * >> A table of state objects (numpy table) describing the current states of the systems (table of 1d numpy tables)
     * >> A table of temperatures (numpy)
     * >> A numpy table of energies
     * >> A numpy table of loops to save
     * >> A numpy table to save the number of missed updates
     * >> A numpy table of dimers connected to each dimer through an n site
     * >> A numpy table of dimers connected to each dimer through a v site
     * >> A numpy table of which dimers to count for each winding number
     * >> A number of iterations
     * >> A maximal number of loops occupancy */

     //------------------------------------------------------INPUT INTERPRETATION---------------------------------------------------------------------//
     PyObject *list_obj;
     PyObject  *states_obj, *betas_obj, *energies_obj, *failedupdates_obj, *d_nd_obj, *d_vd_obj, *d_wn_obj; //*saveloops_obj,
     int niterworm;
     int nmaxiter;
     int nthreads;
     // take the arguments as pointers + int
     if(!PyArg_ParseTuple(args,"OOOOOOOOiii", &list_obj, &states_obj, &betas_obj, &energies_obj, &failedupdates_obj,  &d_nd_obj, &d_vd_obj, &d_wn_obj, &niterworm, &nmaxiter, &nthreads))
	return nullptr;
    //----------------------------//
    /* Interpret the list input  */
    //----------------------------//
    // check that the list_obj is indeed a list
    if(!PyList_Check(list_obj)) {
        PyErr_Format(PyExc_ValueError,
                         "DIMERS.cpp : Give me a list to work with, I'm line %d", \
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
    vector<PyArrayObject*> Marray_list(listsize-1);

    bool ok;
    interactions = getInteractions(list_obj, Marray_list, &ok);
    if(ok == false) return nullptr;
     if(!PyList_Check(list_obj)) {
        PyErr_Format(PyExc_ValueError,
                         "DIMERS.cpp : Give me a list to work with, I'm line %d", \
                         __LINE__);
        return nullptr;
     }

    //-------------------------------//
    /* Interpret the table of states */
    //-------------------------------//

    PyArrayObject* states_array = nullptr;
    tuple<int*, int, int> statestuple = parseStates(states_array, states_obj);
    int* states = get<0>(statestuple);
    if(states == nullptr) {
        Py_XDECREF(states_array);
        PyErr_Format(PyExc_ValueError, "DIMERS.cpp : There was an issue with line %d (NPY array?)", __LINE__);
        return nullptr;
    }
    int nt_states = get<1>(statestuple);
    int statesize = get<2>(statestuple);

    //--------------------------------------//
    /* Interpret the table of temperatures */
    //--------------------------------------//

    // because we want to be able to update the state as well: NPY_ARRAY_INOUT_ARRAY and not NPY_ARRAY_IN_ARRAY
    PyArrayObject *betas_array = (PyArrayObject*) PyArray_FROM_OTF(betas_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if(betas_array == nullptr) {
        Py_XDECREF(betas_array);
        PyErr_Format(PyExc_ValueError, "DIMERS.cpp : There was an issue with line %d (NPY array?)", __LINE__);
        return nullptr;
    }

    int nbt = (int)PyArray_DIM(betas_array, 0);
    //check that state has the dimensions expected
    if(PyArray_NDIM(betas_array) != 1) {
        PyErr_Format(PyExc_ValueError, "DIMERS.cpp : There was an issue with line %d", __LINE__);
        return nullptr;
    }

    if(nt_states != nbt) {
        PyErr_Format(PyExc_ValueError, "DIMERS.cpp : Non-consistent number of temperatures and states, see line %d", __LINE__);
        return nullptr;
    }
    //get pointer as Ctype
    double *betas = (double*)PyArray_DATA(betas_array);

    //---------------------------------//
    /* Interpret the table of energies */
    //---------------------------------//

    PyArrayObject *energies_array = (PyArrayObject*) PyArray_FROM_OTF(energies_obj, NPY_DOUBLE, NPY_ARRAY_INOUT_ARRAY); //INOUT: we want to update the energies
    if(energies_array == nullptr) {
        Py_XDECREF(energies_array);
        PyErr_Format(PyExc_ValueError, "DIMERS.cpp : There was an issue with line %d (NPY array?)", __LINE__);
        return nullptr;
    }

    int nte = (int)PyArray_DIM(energies_array, 0);
    //check that state has the dimensions expected
    if(PyArray_NDIM(energies_array) != 1) {
        PyErr_Format(PyExc_ValueError, "DIMERS.cpp : There was an issue with line %d", __LINE__);
        return nullptr;
    }

    if(nte != nbt) {
        PyErr_Format(PyExc_ValueError, "DIMERS.cpp : Non-consistent number of energies and states, see line %d", __LINE__);
        return nullptr;
    }

    //get pointer as Ctype
    double *energies = (double*)PyArray_DATA(energies_array);


    //-----------------------------------------//
    /* Interpret the pointers to dimer tables */
    //-----------------------------------------//
    PyArrayObject* failedupdates_array = (PyArrayObject*) PyArray_FROM_OTF(failedupdates_obj, NPY_INT32, NPY_ARRAY_INOUT_ARRAY);
    if(failedupdates_array == nullptr) {
        Py_XDECREF(failedupdates_array);
        PyErr_Format(PyExc_ValueError, "DIMERS.cpp : There was an issue with line %d (NPY array?)", __LINE__);
        return nullptr;
    }
    //check that state has the dimensions expected
    if(PyArray_NDIM(failedupdates_array) != 1) {
        PyErr_Format(PyExc_ValueError, "DIMERS.cpp : There was an issue with line %d", __LINE__);
        return nullptr;
    }
    //get pointer as Ctype
    int *failedupdates = (int*)PyArray_DATA(failedupdates_array);


    PyArrayObject* d_nd_array = nullptr;
    tuple<int*, int, int> d_ndtuple = parseInteger2DArray(d_nd_array, d_nd_obj);
    int* d_nd = get<0>(d_ndtuple);
    if(d_nd == nullptr) {
        Py_XDECREF(d_nd_array);
        PyErr_Format(PyExc_ValueError, "DIMERS.cpp : There was an issue with line %d (NPY array?)", __LINE__);
        return nullptr;
    }
    int nnei_d_nd = get<2>(d_ndtuple);

    PyArrayObject* d_vd_array = nullptr;
    tuple<int*, int, int> d_vdtuple = parseInteger2DArray(d_vd_array, d_vd_obj);
    int* d_vd = get<0>(d_vdtuple);
    if(d_vd == nullptr) {
        Py_XDECREF(d_vd_array);
        PyErr_Format(PyExc_ValueError, "DIMERS.cpp : There was an issue with line %d (NPY array?)", __LINE__);
        return nullptr;
    }
    int nnei_d_vd = get<2>(d_vdtuple);

    //prevent any big issue (the algorithm works only for systems with two exit dimers)
    if(nnei_d_vd != 2) {
        Py_XDECREF(d_nd_array);
        Py_XDECREF(d_vd_array);
        PyErr_Format(PyExc_ValueError, "DIMERS.cpp : There was an issue with line %d", __LINE__);
        return nullptr;
    }

    //------------------------------------------------//
    /* Interpret the pointer to winding number table */
    //------------------------------------------------//
    // get the pointer as numpy array
    PyArrayObject* d_wn_array = nullptr;
    tuple<int*, int, int> d_wntuple = parseInteger2DArray(d_wn_array, d_wn_obj);
    int* d_wn = get<0>(d_wntuple);
    if(d_wn== nullptr) {
        Py_XDECREF(d_wn_array);
        PyErr_Format(PyExc_ValueError, "DIMERS.cpp : There was an issue with line %d (NPY array?)", __LINE__);
        return nullptr;
    }
    //------------------------------------------------------FUNCTION CALL---------------------------------------------------------------------//

    //------------------------------------//
    /* Call the manydualworms C function */
    //------------------------------------//
    PyThreadState* threadState = PyEval_SaveThread(); // release the GIL
    mcsevolve(J1, interactions, states, statesize, d_nd, nnei_d_nd, d_vd, nnei_d_vd, d_wn, betas, energies, failedupdates, nbt, nmaxiter, niterworm, nthreads); //saveloops = 0
    PyEval_RestoreThread(threadState); // claim the GIL

    // Clean up
    if( states_array != nullptr){
      Py_DECREF(states_array); // decrement the reference
    }
    if( betas_array != nullptr){
      Py_DECREF(betas_array); // decrement the reference
    }
    //Py_DECREF(saveloops_array);
    if( failedupdates_array != nullptr){
      Py_DECREF(failedupdates_array); // decrement the reference
    }
    if( d_nd_array != nullptr){
      Py_DECREF(d_nd_array); // decrement the reference
    }
    if( d_vd_array != nullptr){
      Py_DECREF(d_vd_array); // decrement the reference
    }
    if( d_wn_array != nullptr){
      Py_DECREF(d_wn_array); // decrement the reference
    }

    for(auto marray : Marray_list) { // clean up as well the array(s) created for the couplings in getInteractions
        Py_DECREF(marray);
    }
    //build output

    /* Build the output */
    PyObject *ret = Py_BuildValue("d", 0);
    return ret;
}
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

//****** MAGNETICMCSEVOLVE *****//
static PyObject* dimers_magneticmcsevolve(PyObject *self, PyObject *args) {
    /* What we want to get from the arguments:
     * >> A list of couplings [J1, (J2, M2), (J3, M3), ...]
     * >> A magnetic field
     * >> A table of state objects (numpy table) describing the current states of the systems (table of 1d numpy tables)
     * >> A table of spinstates object
     * >> A numpy table of dimers connected to each dimer through an n site
     * >> A numpy table of dimers connected to each dimer through a v site
     * >> A numpy table of which dimers to count for each winding number
     * >> A numpy table of spins for update
     * >> A numpy table of dimers for update
     * >> A table of temperatures (numpy)
     * >> A numpy table of energies
     * >> A numpy table to save the number of missed updates
     * >> A maximal loop size
     * >> A number of iterations
     * >> A number of cores */

     //------------------------------------------------------INPUT INTERPRETATION---------------------------------------------------------------------//
     PyObject *list_obj;
     double h;
     PyObject *states_obj, *spinstates_obj, *d_nd_obj, *d_vd_obj, *d_wn_obj;
     PyObject *sidlist_obj, *didlist_obj, *betas_obj, *energies_obj;
     PyObject *failedupdates_obj; //*saveloops_obj,
     int nmaxiter;
     int niterworm;
     int nthreads;
     // take the arguments as pointers + int
     if(!PyArg_ParseTuple(args,"OdOOOOOOOOOOiii", &list_obj, &h, &states_obj, &spinstates_obj, &d_nd_obj, &d_vd_obj, &d_wn_obj, &sidlist_obj, &didlist_obj, &betas_obj, &energies_obj, &failedupdates_obj, &nmaxiter, &niterworm, &nthreads))
	    return nullptr;
    //----------------------------//
    /* Interpret the list input  */
    //----------------------------//
    // check that the list_obj is indeed a list
    if(!PyList_Check(list_obj)) {
        PyErr_Format(PyExc_ValueError,
                         "DIMERS.cpp : Give me a list to work with, I'm line %d", \
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
    vector<PyArrayObject*> Marray_list(listsize-1);

    bool ok;
    interactions = getInteractions(list_obj, Marray_list, &ok);
    if(ok == false) return nullptr;
     if(!PyList_Check(list_obj)) {
        PyErr_Format(PyExc_ValueError,
                         "DIMERS.cpp : Give me a list to work with, I'm line %d", \
                         __LINE__);
        return nullptr;
     }

    //-------------------------------//
    /* Interpret the table of states */
    //-------------------------------//

    PyArrayObject* states_array = nullptr;
    tuple<int*, int, int> statestuple = parseStates(states_array, states_obj);
    int* states = get<0>(statestuple);
    if(states == nullptr) {
        Py_XDECREF(states_array);
        PyErr_Format(PyExc_ValueError, "DIMERS.cpp : There was an issue with line %d (NPY array?)", __LINE__);
        return nullptr;
    }
    int nt_states = get<1>(statestuple);
    int statesize = get<2>(statestuple);

    //-----------------------------------//
    /* Interpret the table of spinstates */
    //-----------------------------------//

    PyArrayObject* spinstates_array = nullptr;
    tuple<int*, int, int> spinstatestuple = parseStates(spinstates_array, spinstates_obj);
    int* spinstates = get<0>(spinstatestuple);
    if(spinstates == nullptr) {
        Py_XDECREF(spinstates_array);
        PyErr_Format(PyExc_ValueError, "DIMERS.cpp : There was an issue with line %d (NPY array?)", __LINE__);
        return nullptr;
    }//int nt_spinstates = get<1>(spinstatestuple);
    int spinstatesize = get<2>(spinstatestuple);
    //-----------------------------------------//
    /* Interpret the pointers to dimer tables */
    //-----------------------------------------//

    PyArrayObject* d_nd_array = nullptr;
    tuple<int*, int, int> d_ndtuple = parseInteger2DArray(d_nd_array, d_nd_obj);
    int* d_nd = get<0>(d_ndtuple);
    if(d_nd == nullptr) {
        Py_XDECREF(d_nd_array);
        PyErr_Format(PyExc_ValueError, "DIMERS.cpp : There was an issue with line %d (NPY array?)", __LINE__);
        return nullptr;
    }
    int nnei_d_nd = get<2>(d_ndtuple);

    PyArrayObject* d_vd_array = nullptr;
    tuple<int*, int, int> d_vdtuple = parseInteger2DArray(d_vd_array, d_vd_obj);
    int* d_vd = get<0>(d_vdtuple);
    if(d_vd == nullptr) {
        Py_XDECREF(d_vd_array);
        PyErr_Format(PyExc_ValueError, "DIMERS.cpp : There was an issue with line %d (NPY array?)", __LINE__);
        return nullptr;
    }
    int nnei_d_vd = get<2>(d_vdtuple);

    //prevent any big issue (the algorithm works only for systems with two exit dimers)
    if(nnei_d_vd != 2) {
        Py_XDECREF(d_nd_array);
        Py_XDECREF(d_vd_array);
        PyErr_Format(PyExc_ValueError, "DIMERS.cpp : There was an issue with line %d", __LINE__);
        return nullptr;
    }

    //------------------------------------------------//
    /* Interpret the pointer to winding number table */
    //------------------------------------------------//
    // get the pointer as numpy array
    PyArrayObject* d_wn_array = nullptr;
    tuple<int*, int, int> d_wntuple = parseInteger2DArray(d_wn_array, d_wn_obj);
    int* d_wn = get<0>(d_wntuple);
    if(d_wn== nullptr) {
        Py_XDECREF(d_wn_array);
        PyErr_Format(PyExc_ValueError, "DIMERS.cpp : There was an issue with line %d (NPY array?)", __LINE__);
        return nullptr;
    }
    //--------------------------------------//
    /* Interpret the table of sidlist */
    //--------------------------------------//
    PyArrayObject* sidlist_array = nullptr;
    tuple<int*, int> sidlisttuple = parseInteger1DArray(sidlist_array, sidlist_obj);
    int* sidlist = get<0>(sidlisttuple);
    if(sidlist == nullptr) {
        Py_XDECREF(sidlist_array);
        PyErr_Format(PyExc_ValueError, "DIMERS.cpp : There was an issue with line %d (NPY array?)", __LINE__);
        return nullptr;
    }
    //--------------------------------------//
    /* Interpret the table of didlist */
    //--------------------------------------//
    PyArrayObject* didlist_array = nullptr;
    tuple<int*, int> didlisttuple = parseInteger1DArray(didlist_array, didlist_obj);
    int* didlist = get<0>(didlisttuple);
    if(didlist == nullptr) {
        Py_XDECREF(didlist_array);
        PyErr_Format(PyExc_ValueError, "DIMERS.cpp : There was an issue with line %d (NPY array?)", __LINE__);
        return nullptr;
    }
    int nbit = get<1>(didlisttuple);

    //--------------------------------------//
    /* Interpret the table of temperatures */
    //--------------------------------------//

    // because we want to be able to update the state as well: NPY_ARRAY_INOUT_ARRAY and not NPY_ARRAY_IN_ARRAY
    PyArrayObject *betas_array = (PyArrayObject*) PyArray_FROM_OTF(betas_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if(betas_array == nullptr) {
        Py_XDECREF(betas_array);
        PyErr_Format(PyExc_ValueError, "DIMERS.cpp : There was an issue with line %d (NPY array?)", __LINE__);
        return nullptr;
    }

    int nbt = (int)PyArray_DIM(betas_array, 0);
    //check that state has the dimensions expected
    if(PyArray_NDIM(betas_array) != 1) {
        PyErr_Format(PyExc_ValueError, "DIMERS.cpp : There was an issue with line %d", __LINE__);
        return nullptr;
    }

    if(nt_states != nbt) {
        PyErr_Format(PyExc_ValueError, "DIMERS.cpp : Non-consistent number of temperatures and states, see line %d", __LINE__);
        return nullptr;
    }
    //get pointer as Ctype
    double *betas = (double*)PyArray_DATA(betas_array);

    //---------------------------------//
    /* Interpret the table of energies */
    //---------------------------------//

    PyArrayObject *energies_array = (PyArrayObject*) PyArray_FROM_OTF(energies_obj, NPY_DOUBLE, NPY_ARRAY_INOUT_ARRAY); //INOUT: we want to update the energies
    if(energies_array == nullptr) {
        Py_XDECREF(energies_array);
        PyErr_Format(PyExc_ValueError, "DIMERS.cpp : There was an issue with line %d (NPY array?)", __LINE__);
        return nullptr;
    }

    int nte = (int)PyArray_DIM(energies_array, 0);
    //check that state has the dimensions expected
    if(PyArray_NDIM(energies_array) != 1) {
        PyErr_Format(PyExc_ValueError, "DIMERS.cpp : There was an issue with line %d", __LINE__);
        return nullptr;
    }

    if(nte != nbt) {
        PyErr_Format(PyExc_ValueError, "DIMERS.cpp : Non-consistent number of energies and states, see line %d", __LINE__);
        return nullptr;
    }

    //get pointer as Ctype
    double *energies = (double*)PyArray_DATA(energies_array);

    //---------------------------------//
    /* Interpret the table of updates */
    //---------------------------------//

    PyArrayObject* failedupdates_array = (PyArrayObject*) PyArray_FROM_OTF(failedupdates_obj, NPY_INT32, NPY_ARRAY_INOUT_ARRAY);
    if(failedupdates_array == nullptr) {
        Py_XDECREF(failedupdates_array);
        PyErr_Format(PyExc_ValueError, "DIMERS.cpp : There was an issue with line %d (NPY array?)", __LINE__);
        return nullptr;
    }
    //check that state has the dimensions expected
    if(PyArray_NDIM(failedupdates_array) != 1) {
        PyErr_Format(PyExc_ValueError, "DIMERS.cpp : There was an issue with line %d", __LINE__);
        return nullptr;
    }
    //get pointer as Ctype
    int *failedupdates = (int*)PyArray_DATA(failedupdates_array);
    //------------------------------------------------------FUNCTION CALL---------------------------------------------------------------------//

    //------------------------------------//
    /* Call the magneticmcs C function */
    //------------------------------------//
    PyThreadState* threadState = PyEval_SaveThread(); // release the GIL
    magneticmcsevolve(J1, interactions, h, states, statesize, spinstates, spinstatesize,
      d_nd, nnei_d_nd, d_vd, nnei_d_vd, d_wn, sidlist, didlist, nbit, betas,
      energies, failedupdates, nbt, nmaxiter, niterworm, nthreads); //saveloops = 0
    PyEval_RestoreThread(threadState); // claim the GIL

    // Clean up
    if( states_array != nullptr){
      Py_DECREF(states_array); // decrement the reference
    }
    if( spinstates_array != nullptr){
      Py_DECREF(spinstates_array); // decrement the reference
    }
    //Py_DECREF(saveloops_array);

    if( d_nd_array != nullptr){
      Py_DECREF(d_nd_array); // decrement the reference
    }
    if( d_vd_array != nullptr){
      Py_DECREF(d_vd_array); // decrement the reference
    }
    if( d_wn_array != nullptr){
      Py_DECREF(d_wn_array); // decrement the reference
    }
    if( sidlist_array != nullptr){
      Py_DECREF(sidlist_array); // decrement the reference
    }
    if( didlist_array != nullptr){
      Py_DECREF(didlist_array); // decrement the reference
    }
    if( betas_array != nullptr){
      Py_DECREF(betas_array); // decrement the reference
    }
    if( energies_array != nullptr){
      Py_DECREF(energies_array); // decrement the reference
    }
    if( failedupdates_array != nullptr){
      Py_DECREF(failedupdates_array); // decrement the reference
    }


    for(auto marray : Marray_list) { // clean up as well the array(s) created for the couplings in getInteractions
        Py_DECREF(marray);
    }
    //build output

    /* Build the output */
    PyObject *ret = Py_BuildValue("d", 0);
    return ret;
}
//--------------------------------------------------------------------------//
////////////////////////////////////////////////////////////////////////////////

//****** ssfSEVOLVE *****//
static PyObject* dimers_ssfsevolve(PyObject *self, PyObject *args) {
    /* What we want to get from the arguments:
     * >> A J1 interaction
     * >> A magnetic field
     * >> A table of state objects (numpy table) describing the current states of the systems (table of 1d numpy tables)
     * >> A table of spinstates object
     * >> A table of spins to plaquette dimers correspondance
     * >> A table of temperatures (numpy)
     * >> A numpy table of energies
     * >> A numpy table to save the number of missed updates
     * >> A number of cores */

     //------------------------------------------------------INPUT INTERPRETATION---------------------------------------------------------------------//
     double J1;
     double h;
     PyObject *states_obj, *spinstates_obj;
     PyObject *s2p_obj;
     PyObject *betas_obj, *energies_obj;
     PyObject *failedupdates_obj; //*saveloops_obj,
     int nthreads;
     int iters;
     // take the arguments as pointers + int
     if(!PyArg_ParseTuple(args,"ddOOOOOOii", &J1, &h, &states_obj, &spinstates_obj, &s2p_obj, &betas_obj, &energies_obj, &failedupdates_obj, &nthreads, &iters))
	    return nullptr;

    //-------------------------------//
    /* Interpret the table of states */
    //-------------------------------//

    PyArrayObject* states_array = nullptr;
    tuple<int*, int, int> statestuple = parseStates(states_array, states_obj);
    int* states = get<0>(statestuple);
    if(states == nullptr) {
        Py_XDECREF(states_array);
        PyErr_Format(PyExc_ValueError, "DIMERS.cpp : There was an issue with line %d (NPY array?)", __LINE__);
        return nullptr;
    }
    int nt_states = get<1>(statestuple);
    int statesize = get<2>(statestuple);

    //-----------------------------------//
    /* Interpret the table of spinstates */
    //-----------------------------------//

    PyArrayObject* spinstates_array = nullptr;
    tuple<int*, int, int> spinstatestuple = parseStates(spinstates_array, spinstates_obj);
    int* spinstates = get<0>(spinstatestuple);
    if(spinstates == nullptr) {
        Py_XDECREF(spinstates_array);
        PyErr_Format(PyExc_ValueError, "DIMERS.cpp : There was an issue with line %d (NPY array?)", __LINE__);
        return nullptr;
    }//int nt_spinstates = get<1>(spinstatestuple);
    int spinstatesize = get<2>(spinstatestuple);

    //-----------------------------------------//
    /* Interpret the pointers to dimer tables */
    //-----------------------------------------//
    PyArrayObject* s2p_array = nullptr;
    tuple<int*, int, int> s2ptuple = parseInteger2DArray(s2p_array, s2p_obj);
    int* s2p = get<0>(s2ptuple);
    if(s2p == nullptr) {
        Py_XDECREF(s2p_array);
        PyErr_Format(PyExc_ValueError, "DIMERS.cpp : There was an issue with line %d (NPY array?)", __LINE__);
        return nullptr;
    }
    int nd = get<2>(s2ptuple);

    //--------------------------------------//
    /* Interpret the table of temperatures */
    //--------------------------------------//

    // because we want to be able to update the state as well: NPY_ARRAY_INOUT_ARRAY and not NPY_ARRAY_IN_ARRAY
    PyArrayObject *betas_array = (PyArrayObject*) PyArray_FROM_OTF(betas_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if(betas_array == nullptr) {
        Py_XDECREF(betas_array);
        PyErr_Format(PyExc_ValueError, "DIMERS.cpp : There was an issue with line %d (NPY array?)", __LINE__);
        return nullptr;
    }

    int nbt = (int)PyArray_DIM(betas_array, 0);
    //check that state has the dimensions expected
    if(PyArray_NDIM(betas_array) != 1) {
        PyErr_Format(PyExc_ValueError, "DIMERS.cpp : There was an issue with line %d", __LINE__);
        return nullptr;
    }

    if(nt_states != nbt) {
        PyErr_Format(PyExc_ValueError, "DIMERS.cpp : Non-consistent number of temperatures and states, see line %d", __LINE__);
        return nullptr;
    }
    //get pointer as Ctype
    double *betas = (double*)PyArray_DATA(betas_array);

    //---------------------------------//
    /* Interpret the table of energies */
    //---------------------------------//

    PyArrayObject *energies_array = (PyArrayObject*) PyArray_FROM_OTF(energies_obj, NPY_DOUBLE, NPY_ARRAY_INOUT_ARRAY); //INOUT: we want to update the energies
    if(energies_array == nullptr) {
        Py_XDECREF(energies_array);
        PyErr_Format(PyExc_ValueError, "DIMERS.cpp : There was an issue with line %d (NPY array?)", __LINE__);
        return nullptr;
    }

    int nte = (int)PyArray_DIM(energies_array, 0);
    //check that state has the dimensions expected
    if(PyArray_NDIM(energies_array) != 1) {
        PyErr_Format(PyExc_ValueError, "DIMERS.cpp : There was an issue with line %d", __LINE__);
        return nullptr;
    }

    if(nte != nbt) {
        PyErr_Format(PyExc_ValueError, "DIMERS.cpp : Non-consistent number of energies and states, see line %d", __LINE__);
        return nullptr;
    }

    //get pointer as Ctype
    double *energies = (double*)PyArray_DATA(energies_array);

    //---------------------------------//
    /* Interpret the table of updates */
    //---------------------------------//

    PyArrayObject* failedupdates_array = (PyArrayObject*) PyArray_FROM_OTF(failedupdates_obj, NPY_INT32, NPY_ARRAY_INOUT_ARRAY);
    if(failedupdates_array == nullptr) {
        Py_XDECREF(failedupdates_array);
        PyErr_Format(PyExc_ValueError, "DIMERS.cpp : There was an issue with line %d (NPY array?)", __LINE__);
        return nullptr;
    }
    //check that state has the dimensions expected
    if(PyArray_NDIM(failedupdates_array) != 1) {
        PyErr_Format(PyExc_ValueError, "DIMERS.cpp : There was an issue with line %d", __LINE__);
        return nullptr;
    }
    //get pointer as Ctype
    int *failedupdates = (int*)PyArray_DATA(failedupdates_array);
    //------------------------------------------------------FUNCTION CALL---------------------------------------------------------------------//

    //------------------------------------//
    /* Call the magneticmcs C function */
    //------------------------------------//
    PyThreadState* threadState = PyEval_SaveThread(); // release the GIL
    ssfsevolve(J1, h, states, statesize, spinstates, spinstatesize,
      s2p, nd, betas, energies, failedupdates, nbt, nthreads, iters); //saveloops = 0
    PyEval_RestoreThread(threadState); // claim the GIL

    // Clean up
    if( states_array != nullptr){
      Py_DECREF(states_array); // decrement the reference
    }
    if( spinstates_array != nullptr){
      Py_DECREF(spinstates_array); // decrement the reference
    }
    if( s2p_array != nullptr){
      Py_DECREF(s2p_array); // decrement the reference
    }
    if( betas_array != nullptr){
      Py_DECREF(betas_array); // decrement the reference
    }
    if( energies_array != nullptr){
      Py_DECREF(energies_array); // decrement the reference
    }
    if( failedupdates_array != nullptr){
      Py_DECREF(failedupdates_array); // decrement the reference
    }
    //build output

    /* Build the output */
    PyObject *ret = Py_BuildValue("d", 0);
    return ret;
}
//--------------------------------------------------------------------------//
/////////////////////////////////////////////////////////////////////////////
//******UPDATESPINSTATES *****//
static PyObject* dimers_updatespinstates(PyObject *self, PyObject *args) {
    /* What we want to get from the arguments:
     * >> A table of state objects (numpy table) describing the current states of the systems (table of 1d numpy tables)
     * >> A table of spinstate objects (numpy table) describing the current states of the systems (table of 1d numpy tables)
     * >> A numpy table of temperature indices
     * >> A numpy table of spin indices
     * >> A numpy table of dimers indices
     * >> A number of statistics
     * >> A number of threads

     */

     //------------------------------------------------------INPUT INTERPRETATION---------------------------------------------------------------------//
     PyObject  *states_obj, *spinstates_obj, *stat_temps_obj, *sidlist_obj, *didlist_obj; //*saveloops_obj,
     int nthreads;
     // take the arguments as pointers + int
     if(!PyArg_ParseTuple(args,"OOOOOi", &states_obj, &spinstates_obj, &stat_temps_obj, &sidlist_obj, &didlist_obj, &nthreads))
	   return nullptr;

    //-------------------------------//
    /* Interpret the table of states */
    //-------------------------------//

    PyArrayObject* states_array = nullptr;
    tuple<int*, int, int> statestuple = parseStates(states_array, states_obj);
    int* states = get<0>(statestuple);
    if(states == nullptr) {
        Py_XDECREF(states_array);
        PyErr_Format(PyExc_ValueError, "DIMERS.cpp : There was an issue with line %d (NPY array?)", __LINE__);
        return nullptr;
    }
    //int nt_states = get<1>(statestuple);
    int statesize = get<2>(statestuple);
    //-------------------------------//
    /* Interpret the table of spins */
    //-------------------------------//
    PyArrayObject* spinstates_array = nullptr;
    tuple<int*, int, int> spinstatestuple = parseStates(spinstates_array, spinstates_obj);
    int* spinstates = get<0>(spinstatestuple);
    if(spinstates == nullptr) {
        Py_XDECREF(spinstates_array);
        PyErr_Format(PyExc_ValueError, "DIMERS.cpp : There was an issue with line %d (NPY array?)", __LINE__);
        return nullptr;
    }//int nt_spinstates = get<1>(spinstatestuple);
    int spinstatesize = get<2>(spinstatestuple);

    //--------------------------------------//
    /* Interpret the table of temperatures */
    //--------------------------------------//
    PyArrayObject* stat_temps_array = nullptr;
    tuple<int*, int> stat_tempstuple = parseInteger1DArray(stat_temps_array, stat_temps_obj);
    int* stat_temps = get<0>(stat_tempstuple);
    if(stat_temps == nullptr) {
        Py_XDECREF(stat_temps_array);
        PyErr_Format(PyExc_ValueError, "DIMERS.cpp : There was an issue with line %d (NPY array?)", __LINE__);
        return nullptr;
    }
    int nbstat = get<1>(stat_tempstuple);

    //--------------------------------------//
    /* Interpret the table of sidlist */
    //--------------------------------------//
    PyArrayObject* sidlist_array = nullptr;
    tuple<int*, int> sidlisttuple = parseInteger1DArray(sidlist_array, sidlist_obj);
    int* sidlist = get<0>(sidlisttuple);
    if(sidlist == nullptr) {
        Py_XDECREF(sidlist_array);
        PyErr_Format(PyExc_ValueError, "DIMERS.cpp : There was an issue with line %d (NPY array?)", __LINE__);
        return nullptr;
    }
    //--------------------------------------//
    /* Interpret the table of didlist */
    //--------------------------------------//
    PyArrayObject* didlist_array = nullptr;
    tuple<int*, int> didlisttuple = parseInteger1DArray(didlist_array, didlist_obj);
    int* didlist = get<0>(didlisttuple);
    if(didlist == nullptr) {
        Py_XDECREF(didlist_array);
        PyErr_Format(PyExc_ValueError, "DIMERS.cpp : There was an issue with line %d (NPY array?)", __LINE__);
        return nullptr;
    }
    int nbit = get<1>(didlisttuple);
    //-----------------------------FUNCTION CALL---------------------------------------------------------------------//

    //------------------------------------//
    /* Call the updatespinstates C function */
    //------------------------------------//
    PyThreadState* threadState = PyEval_SaveThread(); // release the GIL
    updatespinstates(states, spinstates, stat_temps, sidlist, didlist, nbstat, statesize, spinstatesize, nthreads, nbit);
    PyEval_RestoreThread(threadState); // claim the GIL

    // Clean up
    if( states_array != nullptr){
      Py_DECREF(states_array); // decrement the reference
    }
    if( spinstates_array != nullptr){
      Py_DECREF(spinstates_array); // decrement the reference
    }
    //Py_DECREF(saveloops_array);

    if( sidlist_array != nullptr){
      Py_DECREF(sidlist_array); // decrement the reference
    }
    if( didlist_array != nullptr){
      Py_DECREF(didlist_array); // decrement the reference
    }
    if( stat_temps_array != nullptr){
      Py_DECREF(stat_temps_array); // decrement the reference
    }
      //build output

    /* Build the output */
    PyObject *ret = Py_BuildValue("d", 0);
    return ret;
}
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
static PyObject* dimers_measupdates(PyObject *self, PyObject *args) {
    /* What we want to get from the arguments:
     * >> A table of state objects (numpy table) describing the current states of the systems (table of 1d numpy tables)
     * >> A table of spinstate objects (numpy table) describing the current states of the systems (table of 1d numpy tables)
     * >> A numpy table of temperature indices
     * >> A numpy table of spin indices
     * >> A numpy table of dimers indices
     * >> A numpy table of nearest neighbour spins indices
     * >> A numpy table of dimers surrounding each spin
     * >> A number of threads
     * >> A double describint the flipping probability
     */

     //------------------------------------------------------INPUT INTERPRETATION---------------------------------------------------------------------//
     PyObject *states_obj, *spinstates_obj, *stat_temps_obj;
     PyObject *sidlist_obj, *didlist_obj, *nnspins_obj, *s2p_obj; //*saveloops_obj,
     int nthreads;
     double p;
     // take the arguments as pointers + int
     if(!PyArg_ParseTuple(args,"OOOOOOOid", &states_obj, &spinstates_obj, &stat_temps_obj, &sidlist_obj, &didlist_obj, &nnspins_obj, &s2p_obj, &nthreads, &p))
	   return nullptr;

    // //-------------------------------//
    // /* Interpret the table of states */
    // //-------------------------------//
    PyArrayObject* states_array = nullptr;
    tuple<int*, int, int> statestuple = parseStates(states_array, states_obj);
    int* states = get<0>(statestuple);
    if(states == nullptr) {
        Py_XDECREF(states_array);
        PyErr_Format(PyExc_ValueError, "DIMERS.cpp : There was an issue with line %d (NPY array?)", __LINE__);
        return nullptr;
    }
    //int nt_states = get<1>(statestuple);
    int statesize = get<2>(statestuple);
    //-------------------------------//
    /* Interpret the table of spins */
    //-------------------------------//
    PyArrayObject* spinstates_array = nullptr;
    tuple<int*, int, int> spinstatestuple = parseStates(spinstates_array, spinstates_obj);
    int* spinstates = get<0>(spinstatestuple);
    if(spinstates == nullptr) {
        Py_XDECREF(spinstates_array);
        PyErr_Format(PyExc_ValueError, "DIMERS.cpp : There was an issue with line %d (NPY array?)", __LINE__);
        return nullptr;
    }
    //int nt_spinstates = get<1>(spinstatestuple);
    int spinstatesize = get<2>(spinstatestuple);
    //--------------------------------------//
    /* Interpret the table of stats_temperatures*/
    //--------------------------------------//

    PyArrayObject* stat_temps_array = nullptr;
    tuple<int*, int> stat_tempstuple = parseInteger1DArray(stat_temps_array, stat_temps_obj);
    int* stat_temps = get<0>(stat_tempstuple);
    if(stat_temps == nullptr) {
        Py_XDECREF(stat_temps_array);
        PyErr_Format(PyExc_ValueError, "DIMERS.cpp : There was an issue with line %d (NPY array?)", __LINE__);
        return nullptr;
    }
    int nbstat = get<1>(stat_tempstuple);

    //--------------------------------------//
    /* Interpret the table of sidlist */
    //--------------------------------------//
    PyArrayObject* sidlist_array = nullptr;
    tuple<int*, int> sidlisttuple = parseInteger1DArray(sidlist_array, sidlist_obj);
    int* sidlist = get<0>(sidlisttuple);
    if(sidlist == nullptr) {
        Py_XDECREF(sidlist_array);
        PyErr_Format(PyExc_ValueError, "DIMERS.cpp : There was an issue with line %d (NPY array?)", __LINE__);
        return nullptr;
    }
    //--------------------------------------//
    /* Interpret the table of didlist */
    //--------------------------------------//
    PyArrayObject* didlist_array = nullptr;
    tuple<int*, int> didlisttuple = parseInteger1DArray(didlist_array, didlist_obj);
    int* didlist = get<0>(didlisttuple);
    if(didlist == nullptr) {
        Py_XDECREF(didlist_array);
        PyErr_Format(PyExc_ValueError, "DIMERS.cpp : There was an issue with line %d (NPY array?)", __LINE__);
        return nullptr;
    }
    int nbit = get<1>(didlisttuple);

    //--------------------------------------//
    /*      Interpret the table of nnspins   */
    //--------------------------------------//
    PyArrayObject* nnspins_array = nullptr;
    tuple<int*, int, int> nnspinstuple = parseInteger2DArray(nnspins_array, nnspins_obj);
    int* nnspins = get<0>(nnspinstuple);
    if(nnspins == nullptr) {
        Py_XDECREF(nnspins_array);
        PyErr_Format(PyExc_ValueError, "DIMERS.cpp : There was an issue with line %d (NPY array?)", __LINE__);
        return nullptr;
    }
    int nn = get<2>(nnspinstuple);
    //--------------------------------------//
    /*      Interpret the table of s2p   */
    //--------------------------------------//
    PyArrayObject* s2p_array = nullptr;
    tuple<int*, int, int> s2ptuple = parseInteger2DArray(s2p_array, s2p_obj);
    int* s2p = get<0>(s2ptuple);
    if(s2p == nullptr) {
        Py_XDECREF(s2p_array);
        PyErr_Format(PyExc_ValueError, "DIMERS.cpp : There was an issue with line %d (NPY array?)", __LINE__);
        return nullptr;
    }
    int ndims = get<2>(s2ptuple);
    //-----------------------------FUNCTION CALL---------------------------------------------------------------------//
    //------------------------------------//
    /* Call the updatespinstates C function */
    //------------------------------------//
    PyThreadState* threadState = PyEval_SaveThread(); // release the GIL
    measupdates(states, spinstates, stat_temps, sidlist, didlist, nnspins, s2p,
       nbstat, statesize, spinstatesize, nthreads, nbit, nn, ndims, p);
    PyEval_RestoreThread(threadState); // claim the GIL

    // Clean up
    if( states_array != nullptr){
      Py_DECREF(states_array); // decrement the reference
    }
    if( spinstates_array != nullptr){
      Py_DECREF(spinstates_array); // decrement the reference
    }
    //Py_DECREF(saveloops_array);

    if( sidlist_array != nullptr){
      Py_DECREF(sidlist_array); // decrement the reference
    }
    if( didlist_array != nullptr){
      Py_DECREF(didlist_array); // decrement the reference
    }
    if( stat_temps_array != nullptr){
      Py_DECREF(stat_temps_array); // decrement the reference
    }
    if( nnspins_array != nullptr){
      Py_DECREF(nnspins_array); // decrement the reference
    }
    if( s2p_array != nullptr){
      Py_DECREF(s2p_array); // decrement the reference
    }

      //build output

    /* Build the output */
    PyObject *ret = Py_BuildValue("d", 0);
    return ret;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

//****** MANYDUALWORMS *******//
// static PyObject* dimers_manydualworms(PyObject *self, PyObject *args) {
//     /* What we want to get from the arguments:
//      * >> A list of couplings [J1, (J2,M2), (J3,M3),...]
//      * >> A state object (numpy table) describing the current state of the system
//      * >> A numpy table of dimers connected to each dimer through an n site
//      * >> A numpy table of dimers connected to each dimer through a v site
//      * >> A numpy table of which dimers to count for each winding number
//      * >> A temperature */
//
//     PyObject *list_obj;
//     PyObject *state_obj, *d_nd_obj, *d_vd_obj, *d_wn_obj;
//     double beta;
//     int saveloops;
//     int nmaxiter;
//     int iterworm;
//
//     //take the five arguments as pointers to PyObjects + the temperature as a double
//     if(!PyArg_ParseTuple(args,"OOOOOdpii", &list_obj, &state_obj, &d_nd_obj, &d_vd_obj, &d_wn_obj, &beta, &saveloops, &nmaxiter, &iterworm))
//         return nullptr;
//
//     //----------------------------//
//     /* Interpret the list output */
//     //----------------------------//
//     // check that the list_obj is indeed a list
//
//     if(!PyList_Check(list_obj)) {
//         PyErr_Format(PyExc_ValueError,
//                          "DIMERS.cpp : Give mea list to work with, I'm line %d", __LINE__);
//         return nullptr;
//     }
//
//     // first, get J1 which is the first element of the list
//     double J1 = getJ1(list_obj);
//     if (std::isnan(J1)) {
//         return nullptr;
//     }
//
//     // if the list is longer than only one element
//     int listsize = PyList_Size(list_obj);
//     vector<tuple<double, int*, int, int>> interactions(listsize - 1);
//     vector<PyArrayObject*> Marray_list(listsize-1);
//
//     bool ok;
//     interactions = getInteractions(list_obj, Marray_list, &ok);
//     if(ok == false) return nullptr;
//
//     //---------------------------------//
//     /* Interpret the pointer to state */
//     //---------------------------------//
//
//     // because we want to be able to update the state as well: NPY_ARRAY_INOUT_ARRAY and not NPY_ARRAY_IN_ARRAY
//     PyArrayObject *state_array = (PyArrayObject*) PyArray_FROM_OTF(state_obj, NPY_INT32, NPY_ARRAY_INOUT_ARRAY);
//     if(state_array == nullptr) {
//         Py_XDECREF(state_array);
//         PyErr_Format(PyExc_ValueError, "DIMERS.cpp : There was an issue with line %d", __LINE__);
//         return nullptr;
//     }
//
//     int statesize = (int)PyArray_DIM(state_array, 0);
//     //check that state has the dimensions expected
//     if(PyArray_NDIM(state_array) != 1) {
//         PyErr_Format(PyExc_ValueError, "DIMERS.cpp : There was an issue with line %d", __LINE__);
//         return nullptr;
//     }
//
//     //get pointer as Ctype
//     int *state = (int*)PyArray_DATA(state_array);
//
//     //-----------------------------------------//
//     /* Interpret the pointers to dimer tables */
//     //-----------------------------------------//
//     PyArrayObject* d_nd_array = nullptr;
//     tuple<int*, int, int> d_ndtuple = parseInteger2DArray(d_nd_array, d_nd_obj);
//     int* d_nd = get<0>(d_ndtuple);
//     if(d_nd == nullptr) {
//         Py_XDECREF(d_nd_array);
//         PyErr_Format(PyExc_ValueError, "DIMERS.cpp : There was an issue with line %d (NPY array?)", __LINE__);
//         return nullptr;
//     }
//     int nnei_d_nd = get<2>(d_ndtuple);
//
//     PyArrayObject* d_vd_array = nullptr;
//     tuple<int*, int, int> d_vdtuple = parseInteger2DArray(d_vd_array, d_vd_obj);
//     int* d_vd = get<0>(d_vdtuple);
//     if(d_vd == nullptr) {
//         Py_XDECREF(d_vd_array);
//         PyErr_Format(PyExc_ValueError, "DIMERS.cpp : There was an issue with line %d (NPY array?)", __LINE__);
//         return nullptr;
//     }
//     int nnei_d_vd = get<2>(d_vdtuple);
//
//     //prevent any big issue (the algorithm works only for systems with two exit dimers)
//     if(nnei_d_vd != 2) {
//         Py_XDECREF(d_nd_array);
//         Py_XDECREF(d_vd_array);
//         PyErr_Format(PyExc_ValueError, "DIMERS.cpp : There was an issue with line %d", __LINE__);
//         return nullptr;
//     }
//
//     //------------------------------------------------//
//     /* Interpret the pointer to winding number table */
//     //------------------------------------------------//
//     // get the pointer as numpy array
//     PyArrayObject* d_wn_array = nullptr;
//     tuple<int*, int, int> d_wntuple = parseInteger2DArray(d_wn_array, d_wn_obj);
//     int* d_wn = get<0>(d_wntuple);
//     if(d_wn == nullptr) {
//         Py_XDECREF(d_wn_array);
//         PyErr_Format(PyExc_ValueError, "DIMERS.cpp : There was an issue with line %d (NPY array?)", __LINE__);
//         return nullptr;
//     }
//
//     //------------------------------------//
//     /* Call the manydualworms C function */
//     //------------------------------------//
//     PyThreadState* threadState = PyEval_SaveThread();
//     tuple<double, bool, vector<int>, vector<int>> resultworm = manydualworms(J1,
//       interactions, state, statesize, d_nd, nnei_d_nd, d_vd, nnei_d_vd, d_wn,
//       beta, saveloops, nmaxiter, iterworm);
//     PyEval_RestoreThread(threadState);
//     double deltaE = get<0>(resultworm);
//     vector<int> effective_update = get<2>(resultworm);
//     vector<int> looplengths = get<3>(resultworm);
//
//     // Clean up
//     Py_DECREF(state_array);
//     Py_DECREF(d_nd_array);
//     Py_DECREF(d_vd_array);
//     Py_DECREF(d_wn_array);
//     for(auto marray : Marray_list) { // clean up as well the array(s) created for the couplings in getInteractions
//         Py_DECREF(marray);
//     }
//     //build output
//
//     PyObject *effective_update_list = PyList_New(effective_update.size()); //list of dual bonds, i.e. list of int
//     for (unsigned int i = 0; i < effective_update.size(); i++) {
//         PyObject *dualbond = Py_BuildValue("i", effective_update[i]);
//         PyList_SetItem(effective_update_list, i, dualbond);
//     }
//
//     PyObject *looplengthslist = PyList_New(looplengths.size()); //list of length of loops, i.e. list of int
//     for (unsigned int i = 0; i < looplengths.size(); i++) {
//         PyObject *looplength = Py_BuildValue("i", looplengths[i]);
//         PyList_SetItem(looplengthslist, i, looplength);
//     }
//
//     PyObject *ret = Py_BuildValue("dOO", deltaE, effective_update_list, looplengthslist);
//     return ret;
// }
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////
//////////////             HELPER FUNCTIONS                   /////////////////
///////////////////////////////////////////////////////////////////////////////

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
int* parseTupleM(PyObject *tuple_obj, PyArrayObject **PtrM_array) {
    // Get the M_object
    PyObject *M_obj = PyTuple_GetItem(tuple_obj, 1);
    // check for issues
    if(M_obj == nullptr){
        PyErr_Format(PyExc_ValueError, "DIMERS.cpp : There was an issue with line %d", __LINE__);
        return nullptr;
    }

    //interpret as numpy array
    *PtrM_array = (PyArrayObject*) PyArray_FROM_OTF(M_obj, NPY_INT32, NPY_ARRAY_IN_ARRAY); //obj (a_obj), typenum (int),  requirements (C-contiguous)
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
vector<tuple<double, int*, int, int>> getInteractions(PyObject *list_obj, vector<PyArrayObject*>& Marray_list, bool* ok) {
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


//*** PARSE STATES ***//

tuple<int*, int, int> parseStates(PyArrayObject* states_array, PyObject* states_obj) {
  //-------------------------------//
  /* Interpret the table of states */
  //-------------------------------//

  // because we want to be able to update the state as well: NPY_ARRAY_INOUT_ARRAY and not NPY_ARRAY_IN_ARRAY
  states_array = (PyArrayObject*) PyArray_FROM_OTF(states_obj, NPY_INT32, NPY_ARRAY_INOUT_ARRAY);
  if(states_array == nullptr) {
      Py_XDECREF(states_array);
      PyErr_Format(PyExc_ValueError, "DIMERS.cpp : There was an issue with line %d (NPY array?)", __LINE__);
      return make_tuple(nullptr,0,0);
  }

  int nt_states = (int)PyArray_DIM(states_array, 0);
  int statesize = (int)PyArray_DIM(states_array, 1);
  //check that state has the dimensions expected
  if(PyArray_NDIM(states_array) != 2) {
      PyErr_Format(PyExc_ValueError, "DIMERS.cpp : There was an issue with line %d", __LINE__);
      return make_tuple(nullptr,0,0);
  }

  //get pointer as Ctype
  int *states = (int*)PyArray_DATA(states_array);

  tuple<int*, int, int> statestuple(states, nt_states, statesize);
  return statestuple;
}

//*** PARSE 1D INT ARRAY ***//

tuple<int*, int> parseInteger1DArray(PyArrayObject* oned_array, PyObject* oned_pyobject){
  //--------------------------------------//
  /* Interpret the one d table  */
  //--------------------------------------//
  oned_array = (PyArrayObject*) PyArray_FROM_OTF(oned_pyobject, NPY_INT32, NPY_ARRAY_IN_ARRAY);
  if(oned_array == nullptr) {
      Py_XDECREF(oned_array);
      PyErr_Format(PyExc_ValueError, "DIMERS.cpp : There was an issue with line %d (NPY array?)", __LINE__);
      return make_tuple(nullptr,0);
  }

  int nbs = (int)PyArray_DIM(oned_array, 0);
  //check that state has the dimensions expected
  if(PyArray_NDIM(oned_array) != 1) {
      PyErr_Format(PyExc_ValueError, "DIMERS.cpp : There was an issue with line %d", __LINE__);
      return make_tuple(nullptr,0);
  }
  //get pointer as Ctype
  int *onedres = (int*)PyArray_DATA(oned_array);
  if(onedres == nullptr) {
     PyErr_Format(PyExc_ValueError, "DIMERS.cpp : There was an issue with line %d (NPY array?)", __LINE__);
     return make_tuple(nullptr,0);
  }
  tuple<int*, int> onedtuple(onedres,nbs);
  return onedtuple;
}


//*** PARSE 2D INT ARRAY ***//

tuple<int*, int, int> parseInteger2DArray(PyArrayObject* twod_array, PyObject* twod_obj){
  twod_array = (PyArrayObject*) PyArray_FROM_OTF(twod_obj, NPY_INT32, NPY_ARRAY_IN_ARRAY);
  if(twod_array == nullptr) {
      return make_tuple(nullptr,0,0);
  }

  int ndims0 = (int)PyArray_DIM(twod_array, 0);
  int ndims = (int)PyArray_DIM(twod_array, 1);
  //check that state has the dimensions expected
  if(PyArray_NDIM(twod_array) != 2) {
      PyErr_Format(PyExc_ValueError, "DIMERS.cpp : There was an issue with line %d", __LINE__);
      return make_tuple(nullptr,0,0);
  }
  //get pointer as Ctype
  int *twodres = (int*)PyArray_DATA(twod_array);

  tuple<int*, int, int> twodtuple(twodres,ndims0, ndims);
  return twodtuple;
}
