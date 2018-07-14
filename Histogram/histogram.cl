#define HIS_BINS 256

__kernel void histogram(__global int *data, int numData,
                        __global int *histogram) {
  __local int localHistogram[HIS_BINS];
  int lid = get_local_id(0);
  int gid = get_global_id(0);

  int lsize = get_local_size(0);
  int gsize = get_global_size(0);

  /* Initialize the histogram */
  for (int i = lid; i < HIS_BINS; i += lsize)
    localHistogram[i] = 0;

  /* Wait until all work-items within work-group have completed their store */
  barrier(CLK_LOCAL_MEM_FENCE);

  /* Compute local histogram */
  for (int i = gid; i < numData; i += gsize)
    atomic_add(&localHistogram[data[i]], 1);

  barrier(CLK_LOCAL_MEM_FENCE);

  /* Write the local histogram out to the global histogram */
  for (int i = lid; i < HIS_BINS; i += lsize)
    atomic_add(&histogram[i], localHistogram[i]);
}
