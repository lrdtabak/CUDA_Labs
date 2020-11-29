import numpy as np
import time
from PIL import Image
from pycuda import driver as dr
from pycuda.compiler import SourceModule

blocksize = 32
filt = 5
displacement = filt//2

kernel = SourceModule(
    """
       __global__ void s_and_p(unsigned char* pixels, unsigned char* filtered, int* size){
        const int blocksize = 32;
        const int filt = 5;
        int x, y, index;
        int width = size[0];
        int j = blockIdx.x * blockDim.x + threadIdx.x;
    	int i = blockIdx.y * blockDim.y + threadIdx.y;
        __shared__ int local[blocksize][blocksize];
        int arr[filt*filt];
        local[threadIdx.y][threadIdx.x] = pixels[i * width + j];
        __syncthreads ();
        for (int k = 0; k < filt; k++){
            x = max(0, min(threadIdx.y + k - (int)(filt/2), blocksize - 1));
            for (int l = 0; l < filt; l++){
                index = k * filt + l;
                y = max(0, min(threadIdx.x + l - (int)(filt/2), blocksize - 1));
                arr[index] = local[x][y];
            }
        }
        __syncthreads ();
        for (int k = 0; k < filt*filt; k++){
            for (int l = k + 1; l < filt*filt; l++){
                if (arr[k] > arr[l]){
                    unsigned char temp = arr[k];
                    arr[k] = arr[l];
                    arr[l] = temp;
                }
            }
        }
        filtered[i * width + j] = arr[int(filt*filt / 2)];
    }
    """
)

def main():
    print(filt)
    file = "ffile.bmp"
    img = Image.open(file)
    pix = img.load()
    width = img.size[0]
    height = img.size[1]
    pixels = np.zeros((width, height), dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            pixels[i, j] = pix[j, i]

    start_time = time.time()
    new = np.zeros_like(pixels)
    for i in range(height):
        for j in range(width):
            grid = np.zeros(filt ** 2)
            for k in range(filt):
                x = max(0, min(i + k - displacement, height - 1))
                index = k * filt
                for l in range(filt):
                    y = max(0, min(j + l - displacement, width - 1))
                    grid[index + l] = pixels[x, y]
            grid.sort()
            new[i, j] = grid[filt ** 2 // 2]
    end_time = time.time()
    cpu_t = end_time - start_time
    print('Cpu time:')
    print(cpu_t)
    new_img = Image.fromarray(new.astype('uint8'), mode='L')
    new_img.save("After_cpu", format="BMP")


    start_time_gpu = time.time()
    new = np.zeros_like(pixels)
    size = np.array([width, height])
    f = kernel.get_function("s_and_p")
    f(dr.In(pixels), dr.Out(new), dr.In(size), block=(blocksize, blocksize, 1), grid=(width // blocksize, height // blocksize))
    dr.Context.synchronize()
    end_time_gpu = time.time()
    gpu_t = end_time_gpu - start_time_gpu
    print('Gpu time: ')
    print(gpu_t)
    new_img = Image.fromarray(new.astype('uint8'), mode='L')
    new_img.save("After_gpu", format="BMP")


    print("T(cpu)/T(gpu): ")
    print(cpu_t/gpu_t)


if __name__ == '__main__':
    main()


