#include <cstdlib>
#include <ctime>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <string>
#include <sstream>
#include <fstream>
#include <opencv2/opencv.hpp>

#ifdef _OPENACC
#include <openacc.h>
#endif

#define PI 3.1415926535897

using namespace cv;
using namespace std;

void Ltrans(double* X, double* P1, double* P2, int m, int n);
void Lforward(double* P1, double* P2, double* X, int m, int n);
void denoise_bound_init(double* Xobs, double lambda, int idx, double* X_den, double* P1, double* P2, int m, int n, int MAXITER, int test_device);
void deblur_tv_fista(double* Bobs, double* P, double* X_out, int center[2], double lambda, int m, int n, int km, int kn, int MAXITER);
void padPSF(double* P, double* Pbig, int m, int n, int km, int kn);
void dctshift(double* Ps, double* PSF, int m, int n, int center[2]);
void matrix_multiply(double* mat_a, double* mat_b, double* mat_c, int dim);
void dct2(double* macroblock, double* dct2, int m, int n);
void dct2_ocv(double* macroblock, double* dct2, int m, int n);
void idct2(double* dct2, double* idct2, int m, int n);
void idct2_ocv(double* dct2, double* idct2, int m, int n);
double Lipschitz_const(double* mat, int m ,int n);
void load_img();
void load_psf(double* P);

int main(int argc, char** argv) {

    load_img();

    return 0;
}

void load_psf(double* P) {
    std::string file_name = "psf.txt";
    std::string line;
    std::ifstream infile(file_name);
    int idx = 0;
    double tmp;

    while (getline(infile, line)) {
        std::istringstream iss(line);
        iss >> tmp;
        P[idx++] = tmp;
    }
}

void load_img() {
    Mat blur_img = cv::imread("test_pic_1.bmp", 0);
    Mat blur_img_double;

    int m = blur_img.rows;
    int n = blur_img.cols;
    Mat res_img_double(m, n, CV_64F);
    Mat res_img(m, n, CV_8UC1);
    int km = 15;
    int kn = 15;
    int MAXITER = 100;
    int center[2] = {8, 8};
    double lambda = 0.0001;

    double* Bobs = (double*)malloc(m * n * sizeof(double));
    double* X_out = (double*)malloc(m * n * sizeof(double));
    double* P = (double*)malloc(km * kn * sizeof(double));

    load_psf(P);

    for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			Bobs[i * n + j] = (double)blur_img.at<uchar>(i, j) / 255.0;
            X_out[i * n + j] = 2.0 * (double)rand() / RAND_MAX;
		}
	}

    deblur_tv_fista(Bobs, P, X_out, center, lambda, m, n, km, kn, MAXITER);

    for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
            res_img_double.at<double>(i, j) = X_out[i * n + j];
		}
	}

    res_img_double.convertTo(res_img, CV_8UC1, 255.0);

    imwrite("res.bmp", res_img);

    free(Bobs);
    free(X_out);
    free(P);
}

void deblur_tv_fista(double* Bobs, double* P, double* X_out, int center[2], double lambda, int m, int n, int km, int kn, int MAXITER) {
    double L, t_new, t_old;

    double* Pbig = (double*)malloc(m * n * sizeof(double));
    double* e1 = (double*)malloc(m * n * sizeof(double));
    double* Ps = (double*)malloc(m * n * sizeof(double));
    double* tmp1 = (double*)malloc(m * n * sizeof(double));
    double* tmp2 = (double*)malloc(m * n * sizeof(double));
    double* tmp3 = (double*)malloc(m * n * sizeof(double));
    double* Sbig = (double*)malloc(m * n * sizeof(double));
    double* Btrans = (double*)malloc(m * n * sizeof(double));
    double* X_iter = (double*)malloc(m * n * sizeof(double));
    double* Y = (double*)malloc(m * n * sizeof(double));
    double* X_old = (double*)malloc(m * n * sizeof(double));
    double* D = (double*)malloc(m * n * sizeof(double));
    double* P1 = (double*)malloc((m - 1) * n * sizeof(double));
	double* P2 = (double*)malloc(m * (n - 1) * sizeof(double));
    double* Z_iter = (double*)malloc(m * n * sizeof(double));

    padPSF(P, Pbig, m, n, km, kn);

    for (int i = 0; i < m * n; i++) {
        e1[i] = 0.0;
    }
    e1[0] = 1.0;
    dctshift(Ps, Pbig, m, n, center);
    dct2(Ps, tmp1, m, n);
    dct2(e1, tmp2, m, n);
    for (int i = 0; i < m * n; i++) {
        Sbig[i] = tmp1[i] / tmp2[i];
    }

    dct2(Bobs, Btrans, m, n);

    L = 2.0 * Lipschitz_const(Sbig, m, n);


    for (int i = 0; i < m * n; i++) {
        X_iter[i] = Bobs[i];
        Y[i] = X_iter[i];
    }
    t_new = 1.0;

    // main loop
    for (int idx = 0; idx < MAXITER; idx++) {
        t_old = t_new;printf("outer iter: %d\n", idx);
        dct2(Y, tmp1, m, n);

        for (int i = 0; i < m * n; i++) {
            X_old[i] = X_iter[i];
            tmp1[i] = Sbig[i] * (Sbig[i] * tmp1[i] - Btrans[i]);
        }

        idct2(tmp1, tmp2, m, n);

        for (int i = 0 ; i < m * n; i++) {
            Y[i] -= (2.0 / L) * tmp2[i];
        }


        denoise_bound_init(Y, 2.0 * lambda / L, idx, Z_iter, P1, P2, m, n, MAXITER, 0);

        for (int i = 0; i < m * n; i++) {
            X_iter[i] = Z_iter[i];
        }

        t_new = (1.0 + sqrt(1.0 + 4.0 * t_old * t_old)) / 2.0;

        for (int i = 0; i < m * n; i++) {
            Y[i] = X_iter[i] + t_old / t_new * (Z_iter[i] - X_iter[i]) + (t_old - 1.0) / t_new * (X_iter[i] - X_old[i]);
        }
    }

    for (int i = 0; i < m * n; i++) {
        X_out[i] = X_iter[i];
    }

    free(Pbig);
    free(e1);
    free(Ps);
    free(tmp1);
    free(tmp2);
    free(tmp3);
    free(Sbig);
    free(Btrans);
    free(X_iter);
    free(Y);
    free(X_old);
    free(D);
    free(P1);
    free(P2);
    free(Z_iter);

}

double Lipschitz_const(double* mat, int m ,int n) {
    double tmp;
    tmp = fabs(mat[0]);
    double L = tmp * tmp;

    for (int i = 0; i < m * n; i++) {
        tmp = fabs(mat[i]);
        tmp = tmp * tmp;
        L = (tmp >= L)? tmp : L;
    }

    return L;
}

void dctshift(double* Ps, double* PSF, int m, int n, int center[2]) {
    int ii = center[0];
    int jj = center[1];
    int kk = fmin(fmin(ii - 1, m - ii), fmin(jj - 1, n - jj));

    double* PP_old = (double*)malloc((2 * kk + 1) * (2 * kk + 1) * sizeof(double));
    double* PP = (double*)malloc((2 * kk + 1) * (2 * kk + 1) * sizeof(double));
    double* Z1 = (double*)malloc((2 * kk + 1) * (2 * kk + 1) * sizeof(double));
    double* Z2 = (double*)malloc((2 * kk + 1) * (2 * kk + 1) * sizeof(double));
    double* Z1_trans = (double*)malloc((2 * kk + 1) * (2 * kk + 1) * sizeof(double));
    double* Z2_trans = (double*)malloc((2 * kk + 1) * (2 * kk + 1) * sizeof(double));
    double* temp1 = (double*)malloc((2 * kk + 1) * (2 * kk + 1) * sizeof(double));
    double* temp2 = (double*)malloc((2 * kk + 1) * (2 * kk + 1) * sizeof(double));
    double* temp3 = (double*)malloc((2 * kk + 1) * (2 * kk + 1) * sizeof(double));
    double* temp4 = (double*)malloc((2 * kk + 1) * (2 * kk + 1) * sizeof(double));
    double* temp5 = (double*)malloc((2 * kk + 1) * (2 * kk + 1) * sizeof(double));
    double* temp6 = (double*)malloc((2 * kk + 1) * (2 * kk + 1) * sizeof(double));


    for (int i = 0; i < m * n; i++) {
        Ps[i] = 0.0;
    }

    for (int i = 0; i < (2 * kk + 1) * (2 * kk + 1); i++) {
        Z1[i] = 0.0;
        Z2[i] = 0.0;
        Z1_trans[i] = 0.0;
        Z2_trans[i] = 0.0;
        PP[i] = 0.0;
    }

    for (int i = ii - kk - 1; i < ii + kk; i++) {
        for (int j = jj - kk - 1; j < jj + kk; j++) {
            PP_old[(i - ii + kk + 1) * (2 * kk + 1) + (j - jj + kk + 1)] = PSF[i * n + j];
        }
    }

    for (int i = 0; i < kk + 1; i++) {
        Z1[i * (2 * kk + 1) + i + kk] = 1.0;
        Z1_trans[(i + kk) * (2 * kk + 1) + i] = 1.0;
        if (i < kk) {
            Z2[i * (2 * kk + 1) + i + kk + 1] = 1.0;
            Z2_trans[(i + kk + 1) * (2 * kk + 1) + i] = 1.0;
        }
    }

    for (int i = 0; i < (2 * kk + 1); i++) {
        for (int j = 0; j < (2 * kk + 1); j++) {
            temp1[i * (2 * kk + 1) + j] = 0.0;
            temp4[i * (2 * kk + 1) + j] = 0.0;
            for (int k = 0; k < (2 * kk + 1); k++) {
                temp1[i * (2 * kk + 1) + j] += Z1[i * (2 * kk + 1) + k] * PP_old[k * (2 * kk + 1) + j];
                temp4[i * (2 * kk + 1) + j] += Z2[i * (2 * kk + 1) + k] * PP_old[k * (2 * kk + 1) + j];
            }
        }
    }

    for (int i = 0; i < (2 * kk + 1); i++) {
        for (int j = 0; j < (2 * kk + 1); j++) {
            temp2[i * (2 * kk + 1) + j] = 0.0;
            temp3[i * (2 * kk + 1) + j] = 0.0;
            temp5[i * (2 * kk + 1) + j] = 0.0;
            temp6[i * (2 * kk + 1) + j] = 0.0;
            for (int k = 0; k < (2 * kk + 1); k++) {
                temp2[i * (2 * kk + 1) + j] += temp1[i * (2 * kk + 1) + k] * Z1_trans[k * (2 * kk + 1) + j];
                temp3[i * (2 * kk + 1) + j] += temp1[i * (2 * kk + 1) + k] * Z2_trans[k * (2 * kk + 1) + j];
                temp5[i * (2 * kk + 1) + j] += temp4[i * (2 * kk + 1) + k] * Z1_trans[k * (2 * kk + 1) + j];
                temp6[i * (2 * kk + 1) + j] += temp4[i * (2 * kk + 1) + k] * Z2_trans[k * (2 * kk + 1) + j];
            }
        }
    }


    for (int i = 0; i < (2 * kk + 1) * (2 * kk + 1); i++) {
        PP[i] += temp2[i] + temp3[i] + temp5[i] + temp6[i];
    }

    for (int i = 0; i < 2 * kk + 1; i++) {
        for (int j = 0; j < 2 * kk + 1; j++) {
            Ps[i * n + j] = PP[i * (2 * kk + 1) + j];
        }
    }

    free(PP_old);
    free(PP);
    free(Z1);
    free(Z2);
    free(Z1_trans);
    free(Z2_trans);
    free(temp1);
    free(temp2);
    free(temp3);
    free(temp4);
    free(temp5);
    free(temp6);
}

void matrix_multiply(double* mat_a, double* mat_b, double* mat_c, int dim) {
    int i, j, k;

    for (i = 0; i < dim; i++) {
        for (j = 0; j < dim; j++) {
            mat_c[i * dim + j] = 0.0;
            for (k = 0; k < dim; k++) {
                mat_c[i * dim + j] += mat_a[i * dim + k] * mat_b[k * dim + j];
            }
        }
    }
}

void padPSF(double* P, double* Pbig, int m, int n, int km, int kn) {
    memset(Pbig, 0.0, m * n * sizeof(double));

    for (int i = 0; i < km; i++) {
        for (int j = 0; j < kn; j++) {
            Pbig[i * n + j] = P[i * kn + j];
        }
    }
}

void denoise_bound_init(double* Xobs, double lambda, int idx, double* X_den, double* P1, double* P2, int m, int n, int MAXITER, int test_device) {
    double* A = (double*)malloc(m * n * sizeof(double));
	double* R1 = (double*)malloc((m - 1) * n * sizeof(double));
	double* R2 = (double*)malloc(m * (n - 1) * sizeof(double));
	double* Dold = (double*)malloc(m * n * sizeof(double));
	double* Pold1 = (double*)malloc((m - 1) * n * sizeof(double));
	double* Pold2 = (double*)malloc(m * (n - 1) * sizeof(double));
	double* Q1 = (double*)malloc((m - 1) * n * sizeof(double));
	double* Q2 = (double*)malloc(m * (n - 1) * sizeof(double));
	double* temp_vec1 = (double*)malloc(m * n * sizeof(double));
    double* D = (double*)malloc(m * n * sizeof(double));

	double tk = 1.0;
	double tkp1 = 1.0;
	int count = 0;
	int ii = 0;
	double temp1 = 0.0;
	double temp2 = 0.0;
	double re = 0.0;

    if (idx == 0) {

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (i < m - 1) {
                    P1[i * n + j] = 0.0;
                    R1[i * n + j] = 0.0;
                }
                if (j < n - 1) {
                    P2[i * (n - 1) + j] = 0.0;
                    R2[i * (n - 1) + j] = 0.0;
                }
            }
        }
    }

	memset(D, 0.0, m * n * sizeof(double));

	//clock_t startclock, stopclock;
	//double time;

	//startclock = clock();
#pragma acc data copy(Xobs[0 : m * n], A[0 : m * n], P1[0 : (m - 1) * n], P2[0 : m * (n - 1)], \
						R1[0 : (m - 1) * n], R2[0 : m * (n - 1)], D[0 : m * n], \
						Dold[0 : m * n], Pold1[0 : (m - 1) * n], Pold2[0 : m * (n - 1)], Q1[0 : (m - 1) * n], \
						Q2[0 : m * (n - 1)], temp_vec1[0 : m * n]) if(test_device)
{
	while ((ii < MAXITER) && (count < 5)) {
		ii++;

        #pragma acc parallel loop gang
		for (int i = 0; i < m; i++) {
            #pragma acc loop vector
			for (int j = 0; j < n; j++) {
				Dold[i * n + j] = D[i * n + j];

                if (i < m - 1) {
                    Pold1[i * n + j] = P1[i * n + j];
                }
                if (j < n - 1) {
                    Pold2[i * (n - 1) + j] = P2[i * (n - 1) + j];
                }
			}
		}

		tk = tkp1;
		temp1 = 0.0;
		temp2 = 0.0;

		Lforward(R1, R2, temp_vec1, m, n);

        #pragma acc parallel loop gang worker vector
        for (int i = 0; i < m * n; i++) {
            D[i] = Xobs[i] - lambda * temp_vec1[i];
        }
		Ltrans(D, Q1, Q2, m, n);

        #pragma acc parallel loop collapse(2) gang worker vector
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
                if (i < m - 1) {
                    P1[i * n + j] = R1[i * n + j] + 1.0 / (8.0 * lambda) * Q1[i * n + j];
                }
                if (j < n - 1) {
                    P2[i * (n - 1) + j] = R2[i * (n - 1) + j] + 1.0 / (8.0 * lambda) * Q2[i * (n - 1) + j];
                }

				if (i == m - 1 && j == n - 1) {
					A[i * n + j] = 0.0;
				} else if (i == m - 1 && j != n - 1) {
					A[i * n + j] = P2[i * (n - 1) + j] * P2[i * (n - 1) + j];
				} else if (i != m - 1 && j == n - 1) {
					A[i * n + j] = P1[i * n + j] * P1[i * n + j];
				} else {
					A[i * n + j] = P1[i * n + j] * P1[i * n + j] + P2[i * (n - 1) + j] * P2[i * (n - 1) + j];
				}
				if (A[i * n + j] > 1.0) {
					A[i * n + j] = sqrt(A[i * n + j]);
				} else {
					A[i * n + j] = 1.0;
				}
			}
		}

		tkp1 = (1.0 + sqrt(1.0 + 4.0 * tk * tk)) / 2.0;

        #pragma acc parallel loop gang reduction(+: temp1, temp2)
		for (int i = 0; i < m; i++) {
            #pragma acc loop vector reduction(+: temp1, temp2)
			for (int j = 0; j < n; j++) {
                if (i < m - 1) {
                    P1[i * n + j] /= A[i * n + j];
    				R1[i * n + j] = P1[i * n + j] + (tk - 1.0) / tkp1 * (P1[i * n + j] - Pold1[i * n + j]);
                }
                if (j < n - 1) {
                    P2[i * (n - 1) + j] /= A[i * n + j];
    				R2[i * (n - 1) + j] = P2[i * (n - 1) + j] + (tk - 1.0) / tkp1 * (P2[i * (n - 1) + j] - Pold2[i * (n - 1) + j]);
                }

				temp1 += (D[i * n + j] - Dold[i * n + j]) * (D[i * n + j] - Dold[i * n + j]);
				temp2 += D[i * n + j] * D[i * n + j];
			}
		}

		re = sqrt(temp1) / sqrt(temp2);
		if (re < 1e-4) {
			count++;
		} else {
			count = 0;
		}

	}
}

	//stopclock = clock();
	//time = ((double)(stopclock - startclock)) / CLOCKS_PER_SEC;
	// if (test_device) {
	// 	printf("GPU time: %.2f s\n", time);
	// } else {
	// 	printf("CPU time: %.2f s\n", time);
	// }

    memcpy(X_den, D, m * n * sizeof(double));

	free(A);
	free(R1);
    free(R2);
	free(Dold);
    free(Pold1);
	free(Pold2);
	free(Q1);
	free(Q2);
    free(temp_vec1);
    free(D);
}

void Ltrans(double* X, double* P1, double* P2, int m, int n) {
#pragma acc kernels present(X, P1, P2) if(acc_is_present(X, m * n * sizeof(double)))
{
    #pragma acc loop independent
    for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
            if (i < m - 1) {
                P1[i * n + j] = X[i * n + j] - X[(i + 1) * n + j];
            }
            if (j < n - 1) {
                P2[i * (n - 1) + j] = X[i * n + j] - X[i * n + j + 1];
            }
		}
	}
}
}

void Lforward(double* P1, double* P2, double* X, int m, int n) {

#pragma acc kernels present(P1, P2, X) if(acc_is_present(X, m * n * sizeof(double)))
{
    #pragma acc loop independent
	for (int i = 0; i < m - 1; i++) {
		for (int j = 0; j < n; j++) {
			X[i * n + j] = P1[i * n + j];
		}
	}
    #pragma acc loop independent
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n - 1; j++) {
			X[i * n + j] += P2[i * (n - 1) + j];
		}
	}
    #pragma acc loop independent
	for (int i = 1; i < m; i++) {
		for (int j = 0; j < n; j++) {
			X[i * n + j] -= P1[(i - 1) * n + j];
		}
	}
    #pragma acc loop independent
	for (int i = 0; i < m; i++) {
		for (int j = 1; j < n; j++) {
			X[i * n + j] -= P2[i * (n - 1) + j - 1];
		}
	}
}
}

void dct2(double* macroblock, double* dct2, int m, int n) {
    double* x = (double*)malloc(m * n * sizeof(double));
    double* y = (double*)malloc(m * n * sizeof(double));

    double AlphaP, AlphaQ, sum, tmp;

    int i, j, k;

    for (i = 0; i < m; i++) {
        for (j = 0; j < n; j++) {
            x[i * n + j] = (double)i + 1.0;
            y[i * n + j] = (double)j + 1.0;
        }
    }
#pragma omp parallel for private(i, j, sum, AlphaP, AlphaQ, tmp) num_threads(4)
    for (i = 0; i < m; i++) {
        for (j = 0; j < n; j++) {
            if (i == 0) {
                AlphaP = sqrt(1.0 / (double)m);
            } else {
                AlphaP = sqrt(2.0 / (double)m);
            }

            if (j == 0) {
                AlphaQ = sqrt(1.0 / (double)n);
            } else {
                AlphaQ = sqrt(2.0 / (double)n);
            }

            sum = 0.0;
            for (k = 0; k < m * n; k++) {
                tmp = macroblock[k]
                        * cos((PI * (2.0 * x[k] - 1.0) * (double)i) / (2.0 * (double)m))
                        * cos((PI * (2.0 * y[k] - 1.0) * (double)j) / (2.0 * (double)n));
                sum += tmp;
            }

            dct2[i * n + j] = AlphaP * AlphaQ * sum;
        }
    }

    free(x);
    free(y);
}

void idct2(double* dct2, double* idct2, int m, int n) {
    double* x = (double*)malloc(m * n * sizeof(double));
    double* y = (double*)malloc(m * n * sizeof(double));

    double AlphaP, AlphaQ, sum, tmp;

    int i, j, ii, jj;

    for (i = 0; i < m; i++) {
        for (j = 0; j < n; j++) {
            x[i * n + j] = (double)i + 1.0;
            y[i * n + j] = (double)j + 1.0;
        }
    }
#pragma omp parallel for private(i, j, sum, AlphaP, AlphaQ, ii, jj, tmp) num_threads(4)
    for (i = 0; i < m; i++) {
        for (j = 0; j < n; j++) {
            sum = 0.0;
            for (ii = 0; ii < m; ii++) {
                for (jj = 0; jj < n; jj++) {
                    if (ii == 0) {
                        AlphaP = sqrt(1.0 / (double)m);
                    } else {
                        AlphaP = sqrt(2.0 / (double)m);
                    }

                    if (jj == 0) {
                        AlphaQ = sqrt(1.0 / (double)n);
                    } else {
                        AlphaQ = sqrt(2.0 / (double)n);
                    }

                    tmp = dct2[ii * n + jj]
                        * cos((PI * (2.0 * x[i * n + j] - 1.0) * (double)ii) / (2.0 * (double)m))
                        * cos((PI * (2.0 * y[i * n + j] - 1.0) * (double)jj) / (2.0 * (double)n))
                        * AlphaP
                        * AlphaQ;
                    sum += tmp;
                }
            }

            idct2[i * n + j] = sum;
        }
    }

    free(x);
    free(y);
}

void dct2_ocv(double* macroblock, double* dct2, int m, int n) {
    Mat macroblock_m(m, n, CV_64F);
    Mat dct2_m(m, n, CV_64F);

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            macroblock_m.at<double>(i, j) = macroblock[i * n + j];
        }
    }

    cv::dct(macroblock_m, dct2_m);

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            dct2[i * n + j] = dct2_m.at<double>(i, j);
        }
    }
}

void idct2_ocv(double* dct2, double* idct2, int m, int n) {
    Mat dct2_m(m, n, CV_64F);
    Mat idct2_m(m, n, CV_64F);

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            dct2_m.at<double>(i, j) = dct2[i * n + j];
        }
    }

    cv::dct(dct2_m, idct2_m, DCT_INVERSE);

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            idct2[i * n + j] = idct2_m.at<double>(i, j);
        }
    }
}
