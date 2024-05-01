#ifdef __CLION_IDE__
    #include "clion_defines.cl"
#endif

#line 6

#define TILE_SIZE 16

__kernel void matrix_transpose(__global const float* a, __global float* a_t, unsigned int m, unsigned int k) {
    int col = get_global_id(0);
    int row = get_global_id(1);

    int local_col = get_local_id(0);
    int local_row = get_local_id(1);

    __local float tile[TILE_SIZE][TILE_SIZE + 1];

    if (row < m && col < k) {
        tile[local_row][local_col] = a[row * k + col];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    unsigned int res_row = col - local_col + local_row;
    unsigned int res_col = row - local_row + local_col;
    if (res_row < k && res_col < m) {
        a_t[res_row * m + res_col] = tile[local_col][local_row];
    }
}