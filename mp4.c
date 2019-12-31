#include <wb.h>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "CUDA error: ", cudaGetErrorString(err));              \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      return -1;                                                          \
    }                                                                     \
  } while (0)

//@@ Define any useful program-wide constants here
#define TILE_WIDTH 3
#define MASK_SIZE 3
#define MASK_RADIUS 1
//@@ Define constant memory for device kernel here
__constant__ float deviceKernel[MASK_SIZE][MASK_SIZE][MASK_SIZE];

__global__ void conv3d(float *input, float *output, const int z_size,
                       const int y_size, const int x_size) {
  //@@ Insert kernel code here
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int tz = threadIdx.z;
  int col_o = blockIdx.x * TILE_WIDTH + tx;
  int row_o = blockIdx.y * TILE_WIDTH + ty;
  int hight_o = blockIdx.z * TILE_WIDTH + tz;
  int col_i = col_o - MASK_RADIUS;
  int row_i = row_o - MASK_RADIUS;
  int hight_i = hight_o - MASK_RADIUS;
  __shared__ float N_ds[TILE_WIDTH + MASK_SIZE-1][TILE_WIDTH + MASK_SIZE-1][TILE_WIDTH + MASK_SIZE-1];
  if ((row_i >= 0) && (row_i < y_size) && (col_i >= 0) && (col_i < x_size) && (hight_i >= 0) && (hight_i < z_size)) { 
    N_ds[tz][ty][tx] = input[hight_i * y_size * x_size + row_i * x_size + col_i];
  } else {
    N_ds[tz][ty][tx] = 0.0f;
  }
  __syncthreads();
  
  float res = 0.0f;
  if(tz < TILE_WIDTH && ty < TILE_WIDTH && tx < TILE_WIDTH){
    for(int i = 0; i < MASK_SIZE; i++) { 
      for(int j = 0; j < MASK_SIZE; j++) {
        for (int k = 0; k < MASK_SIZE; k++ ) {
          res += deviceKernel[i][j][k] * N_ds[i+tz][j+ty][k+tx];  
        }
      }
    }
    if(row_o < y_size && col_o < x_size && hight_o < z_size) {
      output[hight_o * y_size * x_size + row_o * x_size + col_o] = res; 
    }
  }
}

int main(int argc, char *argv[]) {
  wbArg_t args;
  int z_size;
  int y_size;
  int x_size;
  int inputLength, kernelLength;
  float *hostInput;
  float *hostKernel;
  float *hostOutput;
  float *deviceInput;
  float *deviceOutput;

  args = wbArg_read(argc, argv);

  // Import data
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
  hostKernel = (float *)wbImport(wbArg_getInputFile(args, 1), &kernelLength);
  hostOutput = (float *)malloc(inputLength * sizeof(float));

  // First three elements are the input dimensions
  z_size = hostInput[0];
  y_size = hostInput[1];
  x_size = hostInput[2];
  wbLog(TRACE, "The input size is ", z_size, "x", y_size, "x", x_size);
  assert(z_size * y_size * x_size == inputLength - 3);
  assert(kernelLength == 27);

  wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

  wbTime_start(GPU, "Doing GPU memory allocation");
  //@@ Allocate GPU memory here
  // Recall that inputLength is 3 elements longer than the input data
  // because the first  three elements were the dimensions
  cudaMalloc((void**) &deviceInput,  z_size * y_size * x_size * sizeof(float));
  cudaMalloc((void**) &deviceOutput, z_size * y_size * x_size * sizeof(float));
  wbTime_stop(GPU, "Doing GPU memory allocation");

  wbTime_start(Copy, "Copying data to the GPU");
  //@@ Copy input and kernel to GPU here
  // Recall that the first three elements of hostInput are dimensions and
  // do not need to be copied to the gpu
  cudaMemcpy(deviceInput, hostInput+3, z_size * y_size * x_size * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(deviceKernel, hostKernel, kernelLength * sizeof(float), 0, cudaMemcpyHostToDevice);
  wbTime_stop(Copy, "Copying data to the GPU");

  wbTime_start(Compute, "Doing the computation on the GPU");
  //@@ Initialize grid and block dimensions here
  dim3 dimGrid(ceil(x_size/(1.0*TILE_WIDTH)), ceil(y_size/(1.0*TILE_WIDTH)), ceil(z_size/(1.0*TILE_WIDTH)));
  dim3 dimBlock(TILE_WIDTH + MASK_SIZE-1, TILE_WIDTH + MASK_SIZE-1, TILE_WIDTH + MASK_SIZE-1);
  
  //@@ Launch the GPU kernel here
  conv3d<<<dimGrid, dimBlock>>>(
      deviceInput, deviceOutput,
      z_size, y_size, x_size
    );
  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Doing the computation on the GPU");

  wbTime_start(Copy, "Copying data from the GPU");
  //@@ Copy the device memory back to the host here
  // Recall that the first three elements of the output are the dimensions
  // and should not be set here (they are set below)
  cudaMemcpy(hostOutput + 3, deviceOutput, z_size * y_size * x_size * sizeof(float), cudaMemcpyDeviceToHost);
  wbTime_stop(Copy, "Copying data from the GPU");

  wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

  // Set the output dimensions for correctness checking
  hostOutput[0] = z_size;
  hostOutput[1] = y_size;
  hostOutput[2] = x_size;
  wbSolution(args, hostOutput, inputLength);

  // Free device memory
  cudaFree(deviceInput);
  cudaFree(deviceOutput);

  // Free host memory
  free(hostInput);
  free(hostOutput);
  return 0;
}
