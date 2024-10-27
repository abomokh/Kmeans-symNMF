#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "symnmf.h"

static PyObject *symnmf(PyObject *self, PyObject *args);
static PyObject *sym(PyObject *self, PyObject *args);
static PyObject *ddg(PyObject *self, PyObject *args);
static PyObject *norm(PyObject *self, PyObject *args);
static double **cnonvert_to_C_list(PyObject *list, int K, int d);
static PyObject *cnonvert_to_python_list(double **list, int K, int d);
static int get_2D_dimensions(PyObject* matrix,int *N, int *d);
static PyObject *symnmf_process(PyObject *self, PyObject *args, char *goal);

static PyMethodDef SymNMFMethods[] = {
    {"symnmf", (PyCFunction)symnmf, METH_VARARGS, "Perform SymNMF."},
    {"sym", (PyCFunction)sym, METH_VARARGS, "Compute the similarity matrix."},
    {"ddg", (PyCFunction)ddg, METH_VARARGS, "Compute the diagonal degree matrix."},
    {"norm", (PyCFunction)norm, METH_VARARGS, "Compute the normalized similarity matrix."},
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef symnmfmodule = {
    PyModuleDef_HEAD_INIT,
    "mysymnmfsp",
    NULL,
    -1,
    SymNMFMethods};

PyMODINIT_FUNC PyInit_mysymnmfsp(void)
{
    PyObject *m;
    m = PyModule_Create(&symnmfmodule);
    if (!m)
    {
        return NULL;
    }
    return m;
}

/**
 * Performs SymNMF using the given factor matrix H and weight matrix W.
 *
 * @param self A reference to the module object.
 * @param args The Python arguments as a tuple.
 * @return A Python list representing the updated factor matrix H.
 */
static PyObject *symnmf(PyObject *self, PyObject *args){
    PyObject *P_H, *P_W, *P_result;
    int N, K;

    /* Convert Python objects to C types */
    if(!PyArg_ParseTuple(args, "OOii", &P_H, &P_W, &N, &K)) {
      return NULL;
    }

    double **H, **W, **result;
    H = cnonvert_to_C_list(P_H, N, K);
    W = cnonvert_to_C_list(P_W, N, N);
 
    result = symnmf_imp(H, W, N, K);
    P_result = cnonvert_to_python_list(result, N, K);

    free2DArray(H, N);
    free2DArray(W, N);

    return P_result;
}

/**
 * Computes the similarity matrix for the given data points.
 *
 * @param self A reference to the module object.
 * @param args The Python arguments as a tuple.
 * @return A Python list representing the similarity matrix.
 */
static PyObject *sym(PyObject *self, PyObject *args){
    return symnmf_process(self, args, "sym");
}

/**
 * Computes the normalized similarity matrix for the given data points.
 *
 * @param self A reference to the module object.
 * @param args The Python arguments as a tuple.
 * @return A Python list representing the normalized similarity matrix.
 */
static PyObject *norm(PyObject *self, PyObject *args){
    return symnmf_process(self, args, "norm");
}

/**
 * Computes the diagonal degree matrix for the given data points.
 *
 * @param self A reference to the module object.
 * @param args The Python arguments as a tuple.
 * @return A Python list representing the diagonal degree matrix.
 */
static PyObject *ddg(PyObject *self, PyObject *args){
    return symnmf_process(self, args, "ddg");
}

/**
 * Converts a Python list to a C 2D array.
 *
 * @param list The Python list.
 * @param K The number of rows in the C array.
 * @param d The number of columns in the C array.
 * @return A pointer to the C 2D array.
 */
static double **cnonvert_to_C_list(PyObject *list, int K, int d) {
  if (1 == 1){
  fflush(stdout);
  }
  PyObject *row;
  PyObject *item;
  int j;
  double obj;
  double **array;
  array = (double **)malloc(K * sizeof(double *));
  if (array == NULL) {
    fflush(stdout);
    return NULL;
  }
  fflush(stdout);
  int i;
  for (i = 0; i < K; i++){
    array[i] = (double*)malloc(d * sizeof(double));

    row = PyList_GetItem(list, i);
    if (!PyList_Check(row)) {
      fflush(stdout);
    }
    else {
      fflush(stdout);
    }
    for (j = 0; j < d; j++){
      fflush(stdout);
      item = PyList_GetItem(row, j);
      obj = PyFloat_AsDouble(item);
      array[i][j] = obj;
    }
    fflush(stdout);
  }
  return array;
}

/**
 * Converts a C 2D array to a Python list.
 *
 * @param list The C 2D array.
 * @param dim1 The number of rows in the C array.
 * @param dim2 The number of columns in the C array.
 * @return A Python list representing the C 2D array.
 */
static PyObject *cnonvert_to_python_list(double **list, int dim1, int dim2){
    PyObject *ret, *pyList, *pyValue;
    ret = PyList_New(0);
    int i, j;
    for (i = 0; i < dim1; i++){
        pyList = PyList_New(0);
        for (j = 0; j < dim2; j++){

            pyValue = PyFloat_FromDouble(list[i][j]);
            PyList_Append(pyList, pyValue);
            Py_DECREF(pyValue);
        }
        PyList_Append(ret, pyList);
        Py_DECREF(pyList);
    }
    return Py_BuildValue("O", ret);
}

/**
 * Processes the given goal (sym, ddg, norm) for the provided data points.
 *
 * @param self A reference to the module object.
 * @param args The Python arguments as a tuple.
 * @param goal The goal to be processed.
 * @return A Python list representing the result of the processed goal.
 */
static PyObject *symnmf_process(PyObject *self, PyObject *args, char *goal){
    PyObject *P_datapoints, *P_result;

    /* Convert Python objects to C types */
    if(!PyArg_ParseTuple(args, "O", &P_datapoints)) {
        return NULL;
    }

    int N, d;
    N = 0;
    d = 0;
    get_2D_dimensions(P_datapoints, &N, &d);
    double **datapoints, **result;
    datapoints = cnonvert_to_C_list(P_datapoints, N, d);
    double **A;
    A = calloc2D(N, N);
    result = calloc2D(N, N);
    sym_imp(A, datapoints, N, d);
    if (strcmp(goal, "ddg") == 0)
    {
        double **D;
        D = calloc2D(N, N);
        ddg_imp(D, A, N);
        copyMatrix(result, D, N, N);
        free2DArray(D, N);
    }

    else if (strcmp(goal, "norm") == 0)
    {
        double **D, **W;
        D = calloc2D(N, N);
        W = calloc2D(N, N);
        ddg_imp(D, A, N);
        norm_imp(W, A, D, N);
        copyMatrix(result, W, N, N);
        free2DArray(D, N);
        free2DArray(W, N);
    }
    else /*goal == sym*/
    {
        copyMatrix(result, A, N, N);
    }
    P_result = cnonvert_to_python_list(result, N, N);
    
    free2DArray(A, N);
    free2DArray(datapoints, N);

    return P_result;
}

// Function to get matrix dimensions

/**
 * Gets the dimensions of a 2D Python list.
 *
 * @param matrix The Python list.
 * @param N Pointer to store the number of rows.
 * @param d Pointer to store the number of columns.
 * @return 0 on success, non-zero on failure.
 */
static int get_2D_dimensions(PyObject* matrix, int *N, int *d) {

    // Check if the input is a list
    if (!PyList_Check(matrix)) {
        PyErr_SetString(PyExc_TypeError, "Input should be a list of lists");
        return 0;
    }

    // Get number of rows
    Py_ssize_t rows = PyList_Size(matrix);

    if (rows == 0) {
        // Return (0, 0) for an empty matrix
        *N = 0;
        *d = 0;
        return 0;
    }

    // Get the first row to determine the number of columns
    PyObject* first_row = PyList_GetItem(matrix, 0);

    // Check if the first row is a list
    if (!PyList_Check(first_row)) {
        PyErr_SetString(PyExc_TypeError, "Input should be a list of lists");
        return 0;
    }

    // Get number of columns from the first row
    Py_ssize_t cols = PyList_Size(first_row);

    // Check if all rows have the same number of columns
    for (Py_ssize_t i = 1; i < rows; ++i) {
        PyObject* row = PyList_GetItem(matrix, i);
        if (!PyList_Check(row) || PyList_Size(row) != cols) {
            PyErr_SetString(PyExc_ValueError, "All rows must have the same number of columns");
            return 0;
        }
    }
    
    // Return the dimensions as a list (rows, cols)
    *N = rows;
    *d = cols;
    return 0;
}