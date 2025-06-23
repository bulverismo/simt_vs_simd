#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define MAX_SOURCE_SIZE 4096
#define N (10 * 1000 * 1000)

void benchmark_gpu(float* A, float* B, float* C, int iterations) {
    char *source_str;
    size_t source_size;
    FILE* fp = fopen("simt_gpu.cl", "r");
    if (!fp) {
        fprintf(stderr, "Erro ao abrir o arquivo hello.cl\n");
    }
    source_str = (char*)malloc(MAX_SOURCE_SIZE);
    source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
    fclose(fp);

    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;
    cl_mem bufA, bufB, bufC;
    cl_int ret;

    ret = clGetPlatformIDs(1, &platform, NULL);
    ret = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &ret);
    queue = clCreateCommandQueueWithProperties(context, device, 0, &ret);

    bufA = clCreateBuffer(context, CL_MEM_READ_ONLY, N * sizeof(float), NULL, &ret);
    bufB = clCreateBuffer(context, CL_MEM_READ_ONLY, N * sizeof(float), NULL, &ret);
    bufC = clCreateBuffer(context, CL_MEM_WRITE_ONLY, N * sizeof(float), NULL, &ret);
    if (!bufA || !bufB || !bufC) {
        fprintf(stderr, "Erro ao criar buffers.\n");
        exit(1);
    }
    ret = clEnqueueWriteBuffer(queue, bufA, CL_TRUE, 0, N * sizeof(float), A, 0, NULL, NULL);
    ret = clEnqueueWriteBuffer(queue, bufB, CL_TRUE, 0, N * sizeof(float), B, 0, NULL, NULL);

    program = clCreateProgramWithSource(context, 1, (const char**)&source_str, &source_size, &ret);
    ret = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if (ret != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char *log = malloc(log_size);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        fprintf(stderr, "Erro ao compilar o kernel:\n%s\n", log);
        free(log);
        exit(1);
    }
    
    kernel = clCreateKernel(program, "mac_kernel", &ret);
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufA);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufB);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufC);
    clSetKernelArg(kernel, 3, sizeof(int), &iterations);

    size_t global_size = N;

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    ret = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size, NULL, 0, NULL, NULL);
    clFinish(queue);
    clock_gettime(CLOCK_MONOTONIC, &end);
    
    ret = clEnqueueReadBuffer(queue, bufC, CL_TRUE, 0, N * sizeof(float), C, 0, NULL, NULL);

    double t = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    double macs = (double)N * iterations;
    double gmacs_per_sec = macs / (t * 1e9);

    printf("GPU (OpenCL) MAC time (%d iters): %.6f s\n", iterations, t);
    printf("GPU throughput: %.2f GMAC/s\n", gmacs_per_sec);

    t = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    printf("GPU (OpenCL) time: %.6f s\n", t);

    clReleaseMemObject(bufA);
    clReleaseMemObject(bufB);
    clReleaseMemObject(bufC);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
}
int main() {
    float *A = aligned_alloc(32, sizeof(float) * N);
    float *B = aligned_alloc(32, sizeof(float) * N);
    float *C = aligned_alloc(32, sizeof(float) * N);

    for (int i = 0; i < N; i++) {
        A[i] = i * 0.5f;
        B[i] = i * 0.25f;
    }


    int iterations = 1000;
    benchmark_gpu(A, B, C, iterations); // ou benchmark_gpu()

    free(A); free(B); free(C);
    return 0;
}
