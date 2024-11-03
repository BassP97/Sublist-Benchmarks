#!/bin/bash

nvcc -std=c++11 -o SubList SubList.cu
superList_sizes=(1000000 10000000 100000000 500000000)
subList_sizes=(1000 10000 100000 1000000)


for superList_size in "${superList_sizes[@]}"; do
  for subList_size in "${subList_sizes[@]}"; do
    if [ $subList_size -le $superList_size ]; then
      echo "Running with superList_size=$superList_size and subList_size=$subList_size"
      echo "binary search timing:"
      ./SubList 0 $superList_size $subList_size
      echo "Interpolation search timing:"
      ./SubList 1 $superList_size $subList_size
    fi
  done
done