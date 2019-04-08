#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cstdio>
#include <vector>
#include <chrono>
#include <assert.h>
#include <cusparse.h>
#include <cublas_v2.h>

using namespace std;

const int n = 10000;
const int n_feature = 128;
const double p = .01;
const int  max_e = (int)(n * n * p * 3);
size_t f_size = n * n_feature * sizeof(float);

float f1[n * n_feature], f2[n * n_feature];
float *d_f1, *d_f2;
int indices[max_e], values[max_e];
int *d_indices, *d_values;
float *d_a;
int tote(0);

cublasHandle_t cublasH = NULL;
cusparseHandle_t cusparseH = NULL;
cudaStream_t stream = NULL;
cusparseMatDescr_t descrA = NULL;

void prepareData() {
    srand(time(0));
    const int k = 1e6;
    tote = 0;
    for (int i = 0; i < n; ++i) {
        indices[i] = tote;
        for (int j = 0; j < n; ++j) {
            if (rand() % k < p * k) {
                values[tote++] = j;
            }
        }
        for (int j = 0; j < n_feature; ++j) {
            f1[i * n_feature + j] = rand() % k / (float)k;
            f2[i * n_feature + j] = 0;
        }
    }
    indices[n] = tote;
    printf("Graph edges %d\n", tote);


    cudaMalloc(&d_indices, (n + 1)* sizeof(int));
    cudaMemcpy(d_indices, indices, (n + 1)* sizeof(int), cudaMemcpyHostToDevice);
    cudaMalloc(&d_values, tote * sizeof(int));
    cudaMemcpy(d_values, values, tote * sizeof(int), cudaMemcpyHostToDevice);

    float *a = new float[tote];
    for (int i = 0; i < tote; ++i) {
        a[i] = 1.;
    }
    cudaMalloc(&d_a, tote * sizeof(float));
    cudaMemcpy(d_a, a, tote * sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc(&d_f1, f_size);
    cudaMemcpy(d_f1, f1, f_size, cudaMemcpyHostToDevice);
    cudaMalloc(&d_f2, f_size);
    cudaMemset(d_f2, 0, f_size);


    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    auto cublasStat = cublasCreate(&cublasH);
    assert(CUBLAS_STATUS_SUCCESS == cublasStat);

    cublasStat = cublasSetStream(cublasH, stream);
    assert(CUBLAS_STATUS_SUCCESS == cublasStat);

    auto cusparseStat = cusparseCreate(&cusparseH);
    assert(CUSPARSE_STATUS_SUCCESS == cusparseStat);

    cusparseStat = cusparseSetStream(cusparseH, stream);
    assert(CUSPARSE_STATUS_SUCCESS == cusparseStat);

    /* step 2: configuration of matrix A */
    cusparseStat = cusparseCreateMatDescr(&descrA);
    assert(CUSPARSE_STATUS_SUCCESS == cusparseStat);

    cusparseSetMatIndexBase(descrA,CUSPARSE_INDEX_BASE_ZERO);
    cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL );

    cudaDeviceSynchronize();
}

inline double getDuration(std::chrono::time_point<std::chrono::system_clock> a,
                std::chrono::time_point<std::chrono::system_clock> b) {
    return  std::chrono::duration<double>(b - a).count();
}


#define timestamp(__var__) auto __var__ = std::chrono::system_clock::now();


double runOnce() {
    float alpha = 1.;
    float beta = 0.;
    timestamp(t1);
    for (int i = 0; i < 10; ++i) {
        cusparseScsrmm(cusparseH, CUSPARSE_OPERATION_NON_TRANSPOSE,
                n, n_feature, n, tote, &alpha,
                descrA, d_a, d_indices, d_values,
                d_f1, n, &beta, d_f2, n);
    }
    cudaStreamSynchronize(stream);
    timestamp(t2);
    cudaMemcpy(f2, d_f2, f_size, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    float sum(0);
    for (int i = 0; i < n; ++i) {
        for (int k = 0; k < n_feature; ++k) {
            sum += f2[i * n_feature + k];
        }
    }
    fprintf(stderr, "F20 %f Check sum %f\n", f2[0], sum);
    return getDuration(t1, t2);
}

int main() {
    prepareData();
    double total_time = 0;
    int times = 10;
    fprintf(stderr, "Ready\n");
    for (int i = 0; i < times; ++i) {
        total_time += runOnce();
    }
    fprintf(stderr, "Avg time %.9lf s\n", total_time / times);
}
