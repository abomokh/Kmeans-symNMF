#define _GNU_SOURCE
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#define eps 0.0001
#define max_iter 300

double **symnmf_imp(double **H, double **W, int N, int K);
double **update_H(double **H, double **W, int N, int K);
double frobeniusNorm(double **M, int len1, int len2);
double **sym_imp(double **A, double **datapoints, int N, int d);
double **ddg_imp(double **D, double **A, int N);
double **norm_imp(double **W, double **A, double **D, int N);
double sumVictor(double *victor, int len);
void transpose(double **M_T, double **M, int len1, int len2);
double VictorProduct(double *p, double *q, int len);
void MatrixProduct(double **M, double **M1, double **M2, int len1, int len2, int len3);
double distance(double *p, double *q, int len);
void free2DArray(double **arr, int dim1);
double **calloc2D(int dim1, int dim2);
void read_load(int N, int d, double **victors, FILE *file);
void printMatrix(double **M, int dim1, int dim2);
void copyMatrix(double **M1, double **M2, int len1, int len2);
void subtractMatrix(double **M, double **M1, double **M2, int len1, int len2);

/**
 * Performs Symmetric Non-negative Matrix Factorization (SymNMF).
 *
 * @param H Initial factor matrix H.
 * @param W Weight matrix W.
 * @param N Number of data points.
 * @param K Number of clusters.
 * @return Updated factor matrix H.
 */
double **symnmf_imp(double **H, double **W, int N, int K)
{
    int iter;
    double **old_H;
    double **copy_H;
    double **def;

    iter = 1;
    old_H = calloc2D(N, K);
    copy_H = calloc2D(N, K);
    def = calloc2D(N, K);
    while (iter <= max_iter)
    {
        copyMatrix(copy_H, H, N, K);
        copyMatrix(old_H, H, N, K);
        update_H(copy_H, W, N, K);
        copyMatrix(H, copy_H, N, K);
        subtractMatrix(def, H, old_H, N, K);
        if (frobeniusNorm(def, N, K) < eps)
        {
            break;
        }
        iter++;
    }

    /*free*/
    free2DArray(old_H, N);
    free2DArray(copy_H, N);
    free2DArray(def, N);
    return H;
}

/**
 * Updates the factor matrix H.
 *
 * @param H Initial factor matrix H.
 * @param W Weight matrix W.
 * @param N Number of data points.
 * @param K Number of clusters.
 * @return Updated factor matrix H.
 */
double **update_H(double **H, double **W, int N, int K)
{
    int i, j;
    double beta;
    double **H_T;
    double **WH;
    double **HH_T;
    double **HHH;

    beta = 0.5;
    H_T = calloc2D(K, N);
    transpose(H_T, H, N, K);
    WH = calloc2D(N, K);
    MatrixProduct(WH, W, H, N, N, K);
    HH_T = calloc2D(N, N);
    MatrixProduct(HH_T, H, H_T, N, K, N);
    HHH = calloc2D(N, K);
    MatrixProduct(HHH, HH_T, H, N, N, K);

    for (i = 0; i < N; i++)
    {
        for (j = 0; j < K; j++)
        {
            H[i][j] *= ((1 - beta) + (beta * WH[i][j] / HHH[i][j]));
        }
    }
    /*free*/
    free2DArray(H_T, K);
    free2DArray(WH, N);
    free2DArray(HH_T, N);
    free2DArray(HHH, N);

    return H;
}

/**
 * Calculates the Frobenius norm of a matrix.
 *
 * @param M The matrix.
 * @param len1 Number of rows.
 * @param len2 Number of columns.
 * @return Frobenius norm of the matrix.
 */
double frobeniusNorm(double **M, int len1, int len2)
{
    int i, j;
    double res;
    res = 0;
    for (i = 0; i < len1; i++)
    {
        for (j = 0; j < len2; j++)
        {
            res += pow(M[i][j], 2);
        }
    }
    return res;
}

/* get empty A and fill it according to the instruction file*/
/**
 * Computes the similarity matrix A from data points.
 *
 * @param A Empty similarity matrix to be filled.
 * @param datapoints Data points.
 * @param N Number of data points.
 * @param d Dimensionality of the data points.
 * @return Similarity matrix A.
 */
double **sym_imp(double **A, double **datapoints, int N, int d)
{
    int i, j;

    /*calculate the entires of A*/
    for (i = 0; i < N; i++)
    {
        for (j = 0; j < N; j++)
        {
            if (i == j)
            {
                A[i][j] = 0;
            }
            else
            {
                A[i][j] = exp(pow(distance(datapoints[i], datapoints[j], d), 2) * (-0.5));
            }
        }
    }
    return A;
}

/**
 * Computes the Degree Diagonal matrix D from similarity matrix A.
 *
 * @param D Empty degree matrix to be filled.
 * @param A Similarity matrix.
 * @param N Number of data points.
 * @return Degree matrix D.
 */
double **ddg_imp(double **D, double **A, int N)
{
    int i, j;

    /*calculate D*/
    for (i = 0; i < N; i++)
    {
        for (j = 0; j < N; j++)
        {
            if (i == j)
            {
                D[i][j] = sumVictor(A[i], N);
            }
            else
            {
                D[i][j] = 0;
            }
        }
    }

    return D;
}

/**
 * Normalizes the similarity matrix W using the degree matrix D.
 *
 * @param W Empty normalized matrix to be filled.
 * @param A Similarity matrix.
 * @param D Degree matrix.
 * @param N Number of data points.
 * @return Normalized matrix W.\n
 * @note this fuction changes the values of D to D^(-0.5)
 */
double **norm_imp(double **W, double **A, double **D, int N)
{
    int i;
    double **DA;

    /*calculate D^(-0.5)*/
    for (i = 0; i < N; i++)
    {
        D[i][i] = pow(D[i][i], -0.5);
    }

    /*allocate temp matrix */
    DA = calloc2D(N, N);

    /*calculate W*/
    MatrixProduct(DA, D, A, N, N, N);
    MatrixProduct(W, DA, D, N, N, N);

    free2DArray(DA, N);

    return W;
}

/**
 * Sums the elements of a vector.
 *
 * @param victor The vector.
 * @param len Length of the vector.
 * @return Sum of the elements.
 */
double sumVictor(double *victor, int len)
{
    int i;
    double sum = 0;
    for (i = 0; i < len; i++)
    {
        sum += victor[i];
    }
    return sum;
}

/**
 * Transposes a matrix.
 *
 * @param M_T Transposed matrix.
 * @param M Original matrix.
 * @param len1 Number of rows in the original matrix.
 * @param len2 Number of columns in the original matrix.
 */
void transpose(double **M_T, double **M, int len1, int len2)
{
    /* M: len1*len2 , M_T: len2*len1 */
    int i, j;
    for (i = 0; i < len1; i++)
    {
        for (j = 0; j < len2; j++)
        {
            M_T[j][i] = M[i][j];
        }
    }
}

/**
 * Computes the dot product of two vectors.
 *
 * @param p First vector.
 * @param q Second vector.
 * @param len Length of the vectors.
 * @return Dot product of the vectors.
 */
double VictorProduct(double *p, double *q, int len)
{
    int i;
    double result = 0;
    for (i = 0; i < len; i++)
    {
        result += p[i] * q[i];
    }
    return result;
}

/* M1: len1*len2 , M2: len2*len3 */
/**
 * Multiplies two matrices.
 *
 * @param M Result matrix.
 * @param M1 First matrix.
 * @param M2 Second matrix.
 * @param len1 Number of rows in the first matrix.
 * @param len2 Number of columns in the first matrix and rows in the second matrix.
 * @param len3 Number of columns in the second matrix.
 */
void MatrixProduct(double **M, double **M1, double **M2, int len1, int len2, int len3)
{
    double **res;
    double **M2_T;
    int i, j;

    /*allocate space for result*/
    res = calloc2D(len1, len3);

    /*allocate space for M2_transpose*/
    M2_T = calloc2D(len3, len2);

    /*calculate M2_T*/
    transpose(M2_T, M2, len2, len3);

    /*calculate res = M1*M2*/
    for (i = 0; i < len1; i++)
    {
        for (j = 0; j < len3; j++)
        {
            res[i][j] = VictorProduct(M1[i], M2_T[j], len2);
        }
    }
    /*M = res*/
    copyMatrix(M, res, len1, len3);
    /*free*/
    free2DArray(res, len1);
    free2DArray(M2_T, len3);
}

/**
 * Calculates the Euclidean distance between two vectors.
 *
 * @param p First vector.
 * @param q Second vector.
 * @param len Length of the vectors.
 * @return Euclidean distance between p and q.
 */
double distance(double *p, double *q, int len)
{
    int i;
    double sum = 0.0;
    for (i = 0; i < len; ++i)
    {
        sum += pow(p[i] - q[i], 2);
    }
    return sqrt(sum);
}

/**
 * Frees a 2D array.
 *
 * @param arr The 2D array.
 * @param dim1 Number of rows.
 */
void free2DArray(double **arr, int dim1)
{
    int i;
    for (i = 0; i < dim1; i++)
    {
        free(arr[i]);
    }
    free(arr);
}

/**
 * Allocates memory for a 2D array.
 *
 * @param dim1 Number of rows.
 * @param dim2 Number of columns.
 * @return Allocated 2D array.
 */
double **calloc2D(int dim1, int dim2)
{
    int i;
    double **array2D;
    array2D = calloc(dim1, sizeof(double *));
    for (i = 0; i < dim1; i++)
    {
        array2D[i] = calloc(dim2, sizeof(double));
    }
    return array2D;
}

/**
 * Reads data points from a file and loads them into a 2D array.
 *
 * @param N Number of data points.
 * @param d Dimensionality of the data points.
 * @param victors 2D array to store the data points.
 * @param file File pointer to read from.
 */
void read_load(int N, int d, double **victors, FILE *file)
{
    int i, j;
    for (i = 0; i < N; i++)
    {
        for (j = 0; j < d; j++)
        {
            if (fscanf(file, "%lf,", &victors[i][j]) != 1)
            {
                printf("An Error Has Occurred");
                exit(0);
            }
        }
    }
}

/**
 * Prints a matrix.
 *
 * @param M The matrix.
 * @param dim1 Number of rows.
 * @param dim2 Number of columns.
 */
void printMatrix(double **M, int dim1, int dim2)
{
    int i, j;
    for (i = 0; i < dim1; i++)
    {
        for (j = 0; j < dim2 - 1; j++)
        {
            printf("%.4f,", M[i][j]);
        }
        printf("%.4f\n", M[i][j]);
    }
}

/**
 * Copies the contents of one matrix to another.
 *
 * @param M1 Destination matrix.
 * @param M2 Source matrix.
 * @param len1 Number of rows.
 * @param len2 Number of columns.
 */
void copyMatrix(double **M1, double **M2, int len1, int len2)
{
    int i, j;
    for (i = 0; i < len1; i++)
    {
        for (j = 0; j < len2; j++)
        {
            M1[i][j] = M2[i][j];
        }
    }
}

/**
 * Subtracts one matrix from another.
 *
 * @param M Result matrix.
 * @param M1 Minuend matrix.
 * @param M2 Subtrahend matrix.
 * @param len1 Number of rows.
 * @param len2 Number of columns.
 */
void subtractMatrix(double **M, double **M1, double **M2, int len1, int len2)
{
    int i, j;
    for (i = 0; i < len1; i++)
    {
        for (j = 0; j < len2; j++)
        {
            M[i][j] = M1[i][j] - M2[i][j];
        }
    }
}

int main(int argc, char const *argv[])
{
    int i;
    const char *file_name;
    FILE *file;
    int N, d;
    char buffer[1024];
    const char *goal;
    double **dataPoints;
    double **A;

    i = argc; /*just to make use of argc*/
    i = 0;

    goal = argv[1];
    file_name = argv[2];

    N = 0;
    d = 1;
    file = fopen(file_name, "r");
    if (file == NULL)
    {
        perror("An Error Has Occurred");
        exit(1);
    }

    while (fgets(buffer, sizeof(buffer), file))
    {
        if (N == 0)
        {
            while (buffer[i] != '\0')
            {
                if (buffer[i] == ',')
                    d++;
                i++;
            }
        }
        N++;
    }

    /* allocate space for dataPoints*/
    dataPoints = calloc2D(N, d);

    /*fill the dataPoints*/
    rewind(file);
    read_load(N, d, dataPoints, file);
    fclose(file);

    /* print the output depending on the goal */
    A = calloc2D(N, N);
    sym_imp(A, dataPoints, N, d);

    if (strcmp(goal, "sym") == 0)
    {
        printMatrix(A, N, N);
    }

    else if (strcmp(goal, "ddg") == 0)
    {
        double **D;
        D = calloc2D(N, N);
        ddg_imp(D, A, N);
        printMatrix(D, N, N);
        free2DArray(D, N);
    }

    else if (strcmp(goal, "norm") == 0)
    {
        double **D, **W;
        D = calloc2D(N, N);
        W = calloc2D(N, N);
        ddg_imp(D, A, N);
        norm_imp(W, A, D, N);
        printMatrix(W, N, N);
        free2DArray(D, N);
        free2DArray(W, N);
    }

    /*free*/
    free2DArray(A, N);
    free2DArray(dataPoints, N);
    return 0;
}
