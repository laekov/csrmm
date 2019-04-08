#include <cstdlib>
#include <ctime>
#include <cstdio>
#include <vector>
#include <chrono>

using namespace std;

const int n = 10000;
const int n_feature = 128;
const double p = .1;

vector<int> adjacent_matrix[n];
float f1[n][n_feature], f2[n][n_feature];

void prepareData() {
    srand(time(0));
    const int k = 1e6;
    size_t sz = 0;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (rand() % k < p * k) {
                adjacent_matrix[i].push_back(j);
            }
        }
        sz += adjacent_matrix[i].size();
        for (int j = 0; j < n_feature; ++j) {
            f1[i][j] = rand() % k / (float)k;
            f2[i][j] = 0;
        }
    }
    printf("Graph edges %lu\n", sz);
}


inline double getDuration(std::chrono::time_point<std::chrono::system_clock> a,
                std::chrono::time_point<std::chrono::system_clock> b) {
    return  std::chrono::duration<double>(b - a).count();
}


#define timestamp(__var__) auto __var__ = std::chrono::system_clock::now();


double runOnce() {
    timestamp(t1);
#pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        for (auto j : adjacent_matrix[i]) {
            for (int k = 0; k < n_feature; ++k) {
                f2[i][k] += f1[j][k];
            }
        }
    }
    timestamp(t2);
    double sum = 0.;
    for (int i = 0; i < n; ++i) {
        for (int k = 0; k < n_feature; ++k) {
            sum += f2[i][k];
        }
    }
    fprintf(stderr, "Check sum %lf\n", sum);
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
