import numpy as np
import time
import pycuda.autoinit
from pycuda.compiler import SourceModule

N = 128
BLOCK_SIZE = 16
GRID_SIZE = N // BLOCK_SIZE

kernel = SourceModule(
    """
__global__ void mult(int n,float *a, float *b, float *c)
{
    if((blockDim.y*blockIdx.y + threadIdx.y <n) && (blockDim.x*blockIdx.x + threadIdx.x < n)){
    float gpu_c = 0;
    for(int i=0; i<n; i++){
    float gpu_a = a[(blockDim.y*blockIdx.y + threadIdx.y)*n +i];
    float gpu_b = b[i*n +blockDim.x*blockIdx.x + threadIdx.x];
    gpu_c += gpu_a * gpu_b;
    }
    c[(blockDim.y*blockIdx.y + threadIdx.y) * n + blockDim.x*blockIdx.x + threadIdx.x] = gpu_c;
    }
}
"""
)


def main():
    t1 = time.time()
    A = np.random.rand(N, N).astype(np.float32)
    B = np.random.rand(N, N).astype(np.float32)
    total = A.dot(B)
    t2 = time.time()
    t = t2 - t1
    print('cpu time:')
    print(t)

    t1 = time.time()
    mult = kernel.get_function("mult")
    a_gpu = gpuarray.to_gpu(A)
    b_gpu = gpuarray.to_gpu(B)
    c_gpu = gpuarray.empty((N, N), np.float32)
    mult(np.uint32(N), a_gpu, b_gpu, c_gpu, grid=(GRID_SIZE, GRID_SIZE, 1), block=(BLOCK_SIZE, BLOCK_SIZE, 1))
    t2 = time.time()
    t = t2 - t1
    print('gpu time:')
    print(t)

    # print('test:')
    # print(total, c_gpu)


if __name__ == "__main__":
    main()
