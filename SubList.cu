#include <algorithm>
#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <vector>

__device__ __forceinline__ int binary_search(const int *superList,
                                             const int superList_size,
                                             const int target) {
  int left = 0;
  int right = superList_size - 1;

  while (left <= right) {
    int mid = left + (right - left) / 2;

    if (superList[mid] == target) {
      return mid;
    }
    if (superList[mid] < target) {
      left = mid + 1;
    } else {
      right = mid - 1;
    }
  }
  return -1;
}

__global__ void find_indices(const int *__restrict__ superList,
                             const int superList_size,
                             const int *__restrict__ subList,
                             const int subList_size, int *__restrict__ output,
                             const int *__restrict__ original_indices) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < subList_size) {
    output[original_indices[idx]] =
        binary_search(superList, superList_size, subList[idx]);
  }
}

int main() {
  int superList_size = 500000000;
  int subList_size = 100000;

  std::vector<int> superList(superList_size);
  std::iota(superList.begin(), superList.end(), 1);

  std::vector<int> subList(subList_size);
  for (int i = 0; i < subList_size; ++i) {
    subList[i] = rand() % superList_size + 1;
  }
  std::cout << "Sublist size: " << subList.size() << std::endl;

  std::vector<int> indices(subList_size);
  std::iota(indices.begin(), indices.end(), 0);
  std::sort(indices.begin(), indices.end(),
            [&subList](int a, int b) { return subList[a] < subList[b]; });

  std::vector<int> sorted_subList(subList_size);
  for (int i = 0; i < subList_size; ++i) {
    sorted_subList[i] = subList[indices[i]];
  }

  int *d_superList, *d_subList, *d_output, *d_indices;
  int *h_output;
  cudaMallocHost(&h_output, subList_size * sizeof(int));

  cudaMalloc(&d_superList, superList_size * sizeof(int));
  cudaMalloc(&d_subList, subList_size * sizeof(int));
  cudaMalloc(&d_output, subList_size * sizeof(int));
  cudaMalloc(&d_indices, subList_size * sizeof(int));

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaMemcpy(d_superList, superList.data(), superList_size * sizeof(int),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_subList, sorted_subList.data(), subList_size * sizeof(int),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_indices, indices.data(), subList_size * sizeof(int),
             cudaMemcpyHostToDevice);

  int blockSize = 512;
  int numBlocks = (subList_size + blockSize - 1) / blockSize;

  cudaEventRecord(start);

  find_indices<<<numBlocks, blockSize>>>(d_superList, superList_size, d_subList,
                                         subList_size, d_output, d_indices);

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);

  cudaMemcpy(h_output, d_output, subList_size * sizeof(int),
             cudaMemcpyDeviceToHost);

  std::cout << "Kernel execution time: " << milliseconds << " ms" << std::endl;

  for (int i = 0; i < subList_size; ++i) {
    if (superList[h_output[i]] != subList[i]) {
      std::cout << "Mismatch at index " << h_output[i] << std::endl;
      std::cout << "Expected: " << superList[h_output[i]]
                << ", got: " << h_output[i] << std::endl;
      break;
    }
  }

  cudaFree(d_superList);
  cudaFree(d_subList);
  cudaFree(d_output);
  cudaFree(d_indices);
  cudaFreeHost(h_output);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return 0;
}