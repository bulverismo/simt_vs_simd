__kernel void mac_kernel(__global float* A, __global float* B, __global float* C, int iterations) {
    int i = get_global_id(0);
    float c = C[i];
    float a = A[i];
    float b = B[i];

    for (int iter = 0; iter < iterations; iter++) {
        c += a * b;
    }
    C[i] = c;
}