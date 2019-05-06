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

const int max_n = 22000;
const int max_e = 670003;
const int feature_0 = 1433;
const int feature_1 = 16;
const int feature_2 = 7;

int n, m;
int *gptr, *gidx;
float *gval;
float *f0, *f1, *f2, *wei;

cublasHandle_t cublasH = NULL;
cusparseHandle_t cusparseH = NULL;
cudaStream_t stream = NULL;
cusparseMatDescr_t descrA = NULL;

float* createCudaMatrixRandom(int n) {
    const int k = 1e6;
    float* d;
    d = new float[n];
#pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        d[i] = (float)(rand() % k) / k;
    }
    float* p_d;
    cudaMalloc(&p_d, n * sizeof(float));
    cudaMemcpy(p_d, d, n * sizeof(float), cudaMemcpyHostToDevice);
    return p_d;
}

template <class T>
T* createCudaMatrixCopy(T* d, int n) {
    T* p_d;
    cudaMalloc(&p_d, n * sizeof(T));
    cudaMemcpy(p_d, d, n * sizeof(T), cudaMemcpyHostToDevice);
    return p_d;
}

void prepareData() {
    srand(time(0));
    FILE *fin(fopen("graph.in", "r"));
    fscanf(fin, "%d%d", &n, &m);
    static int indptr[max_e], indices[max_e];
    static float values[max_e];
    for (int i = 0; i < n; ++i) {
        fscanf(fin, "%d", indptr + i);
    }
    for (int i = 0; i < m; ++i) {
        fscanf(fin, "%d", indices + i);
    }
    for (int i = 0; i < m; ++i) {
        fscanf(fin, "%f", values + i);
    }
    fclose(fin);
    gptr = createCudaMatrixCopy(indptr, n);
    gidx = createCudaMatrixCopy(indices, m);
    gval = createCudaMatrixCopy(values, m);
    printf("Graph edges %d\n", m);

    f0 = createCudaMatrixRandom(n * feature_0);
    f1 = createCudaMatrixRandom(n * feature_1);
    f2 = createCudaMatrixRandom(n * feature_1);
    wei = createCudaMatrixRandom(feature_0 * feature_1);

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

__global__
void graph_add(int n_feature, int n, int* indices, int* values, float* f1, float* f2) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float l_f2[128];
    if (i < n) {
        for (int k = 0; k < n_feature; ++k) {
            l_f2[k] = 0;
        }
        for (int j = indices[i]; j < indices[i + 1]; ++j) {
            for (int k = 0; k < n_feature; ++k) {
                l_f2[k] += f1[values[j] * n_feature + k];
            }
        }
        for (int k = 0; k < n_feature; ++k) {
            f2[i * n_feature + k] = l_f2[k];
        }
    }
}

void runit() {
    static float alpha = 1.;
    static float beta = 0.;
    auto gemmStat = cublasSgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N,
            n, feature_1, feature_0, &alpha,
            f0, n,
            wei, feature_0,
            &beta, f1, n);
    graph_add<<<400, 64, 0, stream>>>(feature_1, n, gptr, gidx, f1, f2);
    assert(gemmStat == CUBLAS_STATUS_SUCCESS);
    cudaStreamSynchronize(stream);
}

double runOnce() {
    timestamp(t1);
    runit();
    timestamp(t2);
    float* curf = new float[n * feature_1];
    cudaMemcpy(curf, f2, sizeof(float) * n * feature_1, cudaMemcpyDeviceToHost);
    float csum(0);
    for (int i = 0; i < n * feature_1; ++i) {
        csum += curf[i];
    }
    fprintf(stderr, "Chksum = %.3f, time = %lf s\n", csum, getDuration(t1, t2));
    delete [] curf;
    return getDuration(t1, t2);
}

extern "C" {
    void prepare() {
        prepareData();
    }
    void run() {
        runit();
    }
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
