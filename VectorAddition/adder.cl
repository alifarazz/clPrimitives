/**
 * Add two vectors
 * @param output stores result
 * @param inputA first vector
 * @param inputA second vector
 */
__kernel addVector(__global float output,
		   __global float inputA,
		   __global float inputB)
{
  int gid = get_global_id(0);

  output[gid] = inputA[gid] + inputB(gid);
}
