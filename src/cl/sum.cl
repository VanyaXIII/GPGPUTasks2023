#ifdef __CLION_IDE__
    #include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#define VALUES_PER_WORKITEM 64
#define WORKGROUP_SIZE 128

__kernel void sum_gpu_1(__global const unsigned int* xs, unsigned int n, __global unsigned int* res) {
    int id = get_global_id(0);
    if (id >= n) {
        return;
    }

    atomic_add(res, xs[id]);
}

__kernel void sum_gpu_2(__global const unsigned int* xs, unsigned int n, __global unsigned int* res) {
    int globalId = get_global_id(0);
    if (globalId >= n) {
        return;
    }

    unsigned int sum = 0;
    for (int i = 0; i < VALUES_PER_WORKITEM; ++i) {
        int ind = globalId * VALUES_PER_WORKITEM + i;
        if (ind >= n) {
            break;
        }
        sum += xs[ind];
    }

    atomic_add(res, sum);
}

__kernel void sum_gpu_3(__global const unsigned int* xs, unsigned int n, __global unsigned int* res) {
    int localId = get_local_id(0);
    int groupId = get_group_id(0);
    int groupSize = get_local_size(0);

    unsigned int sum = 0;
    for (int i = 0; i < VALUES_PER_WORKITEM; ++i) {
        int ind = groupId * groupSize * VALUES_PER_WORKITEM + i * groupSize + localId;
        if (ind >= n) {
            break;
        }
        sum += xs[ind];
    }

    atomic_add(res, sum);
}

__kernel void sum_gpu_4(__global const unsigned int* xs, unsigned int n, __global unsigned int* res) {
    int localId = get_local_id(0);
    int globalId = get_global_id(0);

    __local int local_xs[WORKGROUP_SIZE];
    local_xs[localId] = globalId < n ? xs[globalId] : 0;

    barrier(CLK_LOCAL_MEM_FENCE);

    if (localId != 0) {
        return;
    }

    unsigned int sum = 0;
    for (int i = 0; i < WORKGROUP_SIZE; ++i) {
        sum += local_xs[i];
    }

    atomic_add(res, sum);
}

__kernel void sum_gpu_5(__global const unsigned int* xs, unsigned int n, __global unsigned int* res) {
    int localId = get_local_id(0);
    int globalId = get_global_id(0);

    __local int local_xs[WORKGROUP_SIZE];
    local_xs[localId] = globalId < n ? xs[globalId] : 0;

    barrier(CLK_LOCAL_MEM_FENCE);

    for (int nValues = WORKGROUP_SIZE; nValues > 1; nValues /= 2) {
        if (2 * localId < nValues) {
            local_xs[localId] = local_xs[localId] + local_xs[localId + nValues / 2];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (localId == 0) {
        res[get_group_id(0)] = local_xs[0];
    }
}