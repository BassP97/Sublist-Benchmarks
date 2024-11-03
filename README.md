# Sublist-Benchmarks

Given two arrays, superList and subList, both sorted, where all the elements in sublist are contained in superList, find the _indices_ of all the elements in subList in superList

This is a simple cuda program that does just that - for each element in subList, it performs a binary or interpolation search over superList for the element, returning the index at which the element's found. It doesn't use any fancy performance tricks (no shared memory, no texture memory, etc...) but performs kinda ok. To compile the program, run:

`nvcc -std=c++11 -o SubList SubList.cu`

To run the benchmark suite, run `./benchmark.sh`; it'll compile the program automatically.

The performance seems to be bound by the sublist size, which kinda makes sense since it spins off one thread per sublist element. I suspect it scales well w/ the number of SMs available, but I can't test that without shelling out for a few gpu enabled ec2 instances, which I don't wanna do.
