#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N (10 * 1000 * 1000)

void benchmark_cpu(float* A, float* B, float* C) {
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    for (int i = 0; i < N; i += 8) {
        __m256 a = _mm256_loadu_ps(&A[i]);
        __m256 b = _mm256_loadu_ps(&B[i]);
        __m256 c = _mm256_add_ps(a, b);
        _mm256_storeu_ps(&C[i], c);
    }

    clock_gettime(CLOCK_MONOTONIC, &end);
    double t = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    printf("CPU (AVX2) time: %.6f s\n", t);
}

int main() {
    float *A = aligned_alloc(32, sizeof(float) * N);
    float *B = aligned_alloc(32, sizeof(float) * N);
    float *C = aligned_alloc(32, sizeof(float) * N);

    for (int i = 0; i < N; i++) {
        A[i] = i * 0.5f;
        B[i] = i * 0.25f;
    }

    benchmark_cpu(A, B, C); // ou benchmark_gpu()

    free(A); free(B); free(C);
    return 0;
}