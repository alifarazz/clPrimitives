__kernel void
transpose_parallel_per_row
(
 __global const float *src,
 __global       float *trgt,
 int n
 )
{
    int gidx = get_global_id(0);

    for (int i = 0; i < n; ++i)
        trgt[gidx * n + i] = src[i * n + gidx];
        //trgt[i * n + gidx] = src[gidx * n + i];
}

__kernel void
transpose_parallel_per_element
(__global const float *src,
 __global       float *trgt
 )
{
    int2 gid = (int2)(get_global_id(0), get_global_id(1));
    int N    = get_global_size(0);
    
    trgt[N * gid.y + gid.x] = src[N * gid.x + gid.y];
        //trgt[i * n + gidx] = src[gidx * n + i];
}

__kernel void
transpose_parallel_per_element_tiled
(__global const float *src,
 __global       float *trgt,
 __local        float *tile
 )
{
    int2 gid         = (int2)(get_global_id(0), get_global_id(1));
    int2 lid         = (int2)(get_local_id(0), get_local_id(1));
    /* int2 group_id    = (int2)(get_group_id(0), get_group_id(1)); */
    int2 global_size = (int2)(get_global_size(0), get_global_size(1));
    int2 local_size = (int2)(get_local_size(0), get_local_size(1));
    /* int2 num_groups  = (int2)(get_num_groups(0), get_num_groups(1)); */

    tile[lid.y * local_size.x + lid.x] = src[gid.y * global_size.x + gid.x];
    barrier(CLK_LOCAL_MEM_FENCE);
    trgt[gid.x * global_size.y + gid.y] = tile[lid.y * local_size.x + lid.x];
}
    /* int2 globalId = (int2)(get_global_id(0), get_global_id(1)); */
    /* int2 localId = (int2)(get_local_id(0), get_local_id(1)); */
    /* int2 groupId = (int2)(get_group_id (0), get_group_id (1)); */
    /* int2 globalSize = (int2)(get_global_size(0), get_global_size(1)); */
    /* int2 locallSize = (int2)(get_local_size(0), get_local_size(1)); */
    /* int2 numberOfGrp = (int2)(get_num_groups (0), get_num_groups (1)); */
