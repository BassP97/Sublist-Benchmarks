#include <algorithm>
#include <cstring>
#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <vector>

__device__ __forceinline__ int interpolation_search(const int *superList,
                                                    const int superList_size,
                                                    const int target) {
  int left = 0;
  int right = superList_size - 1;

  while (left <= right && target >= superList[left] &&
         target <= superList[right]) {
    if (left == right) {
      if (superList[left] == target)
        return left;
      return -1;
    }

    int pos =
        left + ((double)(right - left) / (superList[right] - superList[left]) *
                (target - superList[left]));

    if (superList[pos] == target) {
      return pos;
    }
    if (superList[pos] < target) {
      left = pos + 1;
    } else {
      right = pos - 1;
    }
  }
  return -1;
}

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
                             const int *__restrict__ original_indices,
                             bool use_interpolation) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < subList_size) {
    if (use_interpolation) {
      output[original_indices[idx]] =
          interpolation_search(superList, superList_size, subList[idx]);
    } else {
      output[original_indices[idx]] =
          binary_search(superList, superList_size, subList[idx]);
    }
  }
}

int main(int argc, char *argv[]) {
  if (argc != 4) {
    std::cerr << "Usage: " << argv[0]
              << " <search_type> <superList_size> <subList_size> " << std::endl;
    std::cerr << "search_type: 0 for binary search, 1 for interpolation search"
              << std::endl;
    return 1;
  }

  bool use_interpolation = std::stoi(argv[1]);
  int superList_size = std::stoi(argv[2]);
  int subList_size = std::stoi(argv[3]);

  std::vector<int> superList(superList_size);
  std::iota(superList.begin(), superList.end(), 1);

  std::vector<int> subList(subList_size);
  for (int i = 0; i < subList_size; ++i) {
    subList[i] = rand() % superList_size + 1;
  }

  std::vector<int> indices(subList_size);
  std::iota(indices.begin(), indices.end(), 0);
  std::sort(indices.begin(), indices.end(),
            [&subList](int a, int b) { return subList[a] < subList[b]; });

  std::vector<int> sorted_subList(subList_size);
  for (int i = 0; i < subList_size; ++i) {
    sorted_subList[i] = subList[indices[i]];
  }
  // std::cout << "Done with setup. Running the kernel" << std::endl;

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
                                         subList_size, d_output, d_indices,
                                         use_interpolation);

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