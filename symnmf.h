# ifndef SYM_NMF
# define SYM_NMF

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

# endif