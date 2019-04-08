#include <cstdlib>
#include <ctime>
#include <cstdio>
#include <vector>
#include <chrono>

using namespace std;

const int n = 10000;
const int n_feature = 128;
const double p = .1;
const int  max_e = (int)(n * n * p * 3);
size_t f_size = n * n_feature * sizeof(float);

vector<int> adjacent_matrix[n];
float f1[n * n_feature], f2[n * n_feature];
float *d_f1, *d_f2;
int indices[max_e], values[max_e];
int *d_indices, *d_values;

void prepareData() {
    srand(time(0));
    const int k = 1e6;
    int tote(0);
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
    cudaMalloc(&d_f1, f_size);
    cudaMemcpy(d_f1, f1, f_size, cudaMemcpyHostToDevice);
    cudaMalloc(&d_f2, f_size);
    cudaMemset(d_f2, 0, f_size);
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
    float l_f2[::n_feature];
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


double runOnce() {
    timestamp(t1);
    graph_add<<<128, 128>>>(n_feature, n, d_indices, d_values, d_f1, d_f2);
    cudaDeviceSynchronize();
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
