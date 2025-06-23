#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N (10 * 1000 * 1000)
void benchmark_cpu(float* A, float* B, float* C, int iterations) {
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    for (int iter = 0; iter < iterations; iter++) {
        for (int i = 0; i < N; i += 8) {
            __m256 a = _mm256_loadu_ps(&A[i]);
            __m256 b = _mm256_loadu_ps(&B[i]);
            __m256 c = _mm256_loadu_ps(&C[i]);
            __m256 mul = _mm256_mul_ps(a, b);
            c = _mm256_add_ps(c, mul);
            _mm256_storeu_ps(&C[i], c);
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &end);
    double t = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

    double macs = (double)N * iterations;
    double gmacs_per_sec = macs / (t * 1e9);

    printf("CPU (AVX2) MAC time (%d iters): %.6f s\n", iterations, t);
    printf("CPU throughput: %.2f GMAC/s\n", gmacs_per_sec);
}


int main() {
    float *A = aligned_alloc(32, sizeof(float) * N);
    float *B = aligned_alloc(32, sizeof(float) * N);
    float *C = aligned_alloc(32, sizeof(float) * N);

    for (int i = 0; i < N; i++) {
        A[i] = i * 0.5f;
        B[i] = i * 0.25f;
        C[i] = 0.0f;
    }

    int iterations = 1000;

    benchmark_cpu(A, B, C, iterations); // ou benchmark_gpu()

    free(A); free(B); free(C);
    return 0;
}