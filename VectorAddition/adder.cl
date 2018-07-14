/*
 * Add two vectors
 * @param inputA first vector
 * @param inputB second vector
 * @param output stores result
 */
__kernel void vecadd(__global int *inputA, __global int *inputB, __global int *output) {
  int gid = get_global_id(0);


  output[gid] = inputA[gid] + inputB[gid];
}
