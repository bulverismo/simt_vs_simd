all:
	gcc -g -O2 -mavx2 simd_cpu.c -o simd_cpu
	gcc -g -O2 simt_gpu.c -lOpenCL -o simt_gpu

run:
	./simd_cpu && ./simt_gpu
