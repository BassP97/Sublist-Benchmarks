#include <algorithm>
#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <vector>

__device__ int binary_search(const int *superList, int superList_size,
                             int target) {
  int left = 0;
  int right = superList_size - 1;
  while (left <= right) {
    int mid = left + (right - left) / 2;
    if (superList[mid] == target) {
      return mid;
    } else if (superList[mid] < target) {
      left = mid + 1;
    } else {
      right = mid - 1;
    }
  }
  return -1;
}

__global__ void find_indices(const int *superList, int superList_size,
                             const int *subList, int subList_size,
                             int *output) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < subList_size) {
    output[idx] = binary_search(superList, superList_size, subList[idx]);
  }
}

int main() {
  std::vector<int> superList(500000000);
  std::iota(superList.begin(), superList.end(), 1);

  std::vector<int> subList(10000);
  for (int i = 0; i < 10000; ++i) {
    subList[i] = rand() % 500000000 + 1;
  }

  int superList_size = superList.size();
  int subList_size = subList.size();

  int *d_superList;
  int *d_subList;
  int *d_output;
  int *h_output = new int[subList_size];

  cudaMalloc(&d_superList, superList_size * sizeof(int));
  cudaMalloc(&d_subList, subList_size * sizeof(int));
  cudaMalloc(&d_output, subList_size * sizeof(int));

  cudaMemcpy(d_superList, superList.data(), superList_size * sizeof(int),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_subList, subList.data(), subList_size * sizeof(int),
             cudaMemcpyHostToDevice);

  int blockSize = 256;
  int numBlocks = (subList_size + blockSize - 1) / blockSize;

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);

  find_indices<<<numBlocks, blockSize>>>(d_superList, superList_size, d_subList,
                                         subList_size, d_output);

  cudaEventRecord(stop);

  cudaEventSynchronize(stop);

  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);

  cudaMemcpy(h_output, d_output, subList_size * sizeof(int),
             cudaMemcpyDeviceToHost);

  std::cout << "Kernel execution time: " << milliseconds << " ms" << std::endl;

  cudaFree(d_superList);
  cudaFree(d_subList);
  cudaFree(d_output);
  delete[] h_output;

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return 0;
}