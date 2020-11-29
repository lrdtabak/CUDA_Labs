import numpy as np
import time
import random
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
from pycuda.curandom import rand as curandom
from pycuda import driver as dr
from pycuda.compiler import SourceModule

N = 10000000

kernel = SourceModule(
    """
    __global__ void foundpi(double *x, double *y, int *points_in_circle, const int N){
        int j = blockIdx.x * blockDim.x + threadIdx.x;
        int c = 0;
        for (int i = j; i < N; i += gridDim.x * blockDim.x) {
            if (x[i]*x[i] + y[i]*y[i] <= 1) {
            c+=1;
            }
        }
        atomicAdd(points_in_circle, c);
    }
    """)


def main():
    print(N)
    start_time = time.time()
    points_in_circle = 0
    x = np.zeros((N, 1))
    y = np.zeros((N, 1))
    for i in range(N):
        x[i] = random.uniform(-1, 1)
        y[i] = random.uniform(-1, 1)

    for i in range(N):
        if x[i] ** 2 + y[i] ** 2 <= 1:
            points_in_circle = points_in_circle + 1

    pi = 4 * points_in_circle / N
    end_time = time.time()
    t = end_time - start_time
    err = np.abs(np.pi - pi)
    print(pi)
    print('Cpu time: ', t, '. Error: ', err)

    start_time = time.time()
    gpu_points_in_circle = gpuarray.zeros((1,), dtype=np.int32)
    gpu_points_in_circle = gpu_points_in_circle.get()

    gpu_x = curandom((N,), dtype=np.double).get().astype(np.double)
    gpu_y = curandom((N,), dtype=np.double).get().astype(np.double)
    pi_calc = kernel.get_function("foundpi")
    pi_calc(dr.In(gpu_x), dr.In(gpu_y), dr.Out(gpu_points_in_circle), np.int32(N), block=(128, 1, 1),
            grid=(int(N / (128 ** 2)), 1))
    dr.Context.synchronize()

    gpu_pi = 4 * gpu_points_in_circle[0] / N
    end_time = time.time()
    gpu_t = end_time - start_time
    gpu_err = np.abs(np.pi - gpu_pi)
    print(gpu_pi)
    print('Gpu time: ', gpu_t, '. Error: ', gpu_err)

    print('T(cpu)/T(gpu): ')
    print(t / gpu_t)


if __name__ == '__main__':
    main()