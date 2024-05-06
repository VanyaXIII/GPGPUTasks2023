#ifdef __CLION_IDE__
    #include "clion_defines.cl"
#endif

#line 6

#define TILE_SIZE 16
#define WPT 8
#define RTS 2


__kernel void matrix_multiplication_naive(__global const float* a, __global const float* b, __global float* c,
                                          unsigned int m, unsigned int k, unsigned int n) {
    int col = get_global_id(0);
    int row = get_global_id(1);

    if (row >= m || col >= n) {
        return;
    }

    float acc = .0;
    for (int i = 0; i < k; ++i) {
        acc += a[row * k + i] * b[i * n + col];
    }
    c[row * n + col] = acc;
}

__kernel void matrix_multiplication_local(__global const float* a, __global const float* b, __global float* c,

                                          unsigned int m, unsigned int k, unsigned int n) {
    int col = get_global_id(0);
    int row = get_global_id(1);
    int local_col = get_local_id(0);
    int local_row = get_local_id(1);

    __local float tile_a[TILE_SIZE][TILE_SIZE + 1];
    __local float tile_b[TILE_SIZE][TILE_SIZE + 1];

    float acc = .0;
    int num_tiles = k / TILE_SIZE;
    for (int t = 0; t <= num_tiles; ++t) {
        int tiled_row = TILE_SIZE * t + local_col;
        int tiled_col = TILE_SIZE * t + local_row;

        int ind = row * k + tiled_row;
        tile_a[local_row][local_col] = (ind < m * k) ? a[ind] : 0;

        ind = tiled_col * n + col;
        tile_b[local_row][local_col] = (ind < k * n) ? b[ind] : 0;
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int i = 0; i < TILE_SIZE; ++i) {
            acc += tile_a[local_row][i] * tile_b[i][local_col];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (row < m && col < n) {
        c[row * n + col] = acc;
    }
}

__kernel void matrix_multiplication_wpt(__global const float* a, __global const float* b, __global float* c,
                                        unsigned int m, unsigned int k, unsigned int n) {
    int local_col = get_local_id(0);
    int local_row = get_local_id(1);
    int col = TILE_SIZE * get_group_id(0) + local_col;
    int row = TILE_SIZE * get_group_id(1) + local_row;

    __local float tile_a[TILE_SIZE][TILE_SIZE + 1];
    __local float tile_b[TILE_SIZE][TILE_SIZE + 1];

    float acc[WPT];
    for (int w = 0; w < WPT; ++w) {
        acc[w] = .0;
    }
    int num_tiles = k / TILE_SIZE;
    for (int t = 0; t <= num_tiles; ++t) {

        for (int w = 0; w < WPT; ++w) {
            int tiled_row = TILE_SIZE * t + local_col;
            int tiled_col = TILE_SIZE * t + local_row;
            int ind = (row + w * RTS) * k + tiled_row;
            tile_a[local_row + w * RTS][local_col] = (ind < m * k) ? a[ind] : 0;
            ind = (tiled_col + w * RTS) * n + col;
            tile_b[local_row + w * RTS][local_col] = (ind < k * n) ? b[ind] : 0;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int i = 0; i < TILE_SIZE; ++i) {
            for (int w = 0; w < WPT; ++w) {
                acc[w] += tile_a[local_row + w * RTS][i] * tile_b[i][local_col];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for (int w = 0; w < WPT; ++w) {
        if (row + w * RTS < m && col < n) {
            c[(row + w * RTS) * n + col] = acc[w];
        }
    }
}