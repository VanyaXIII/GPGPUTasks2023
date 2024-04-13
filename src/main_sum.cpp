#include "cl/sum_cl.h"
#include "libgpu/context.h"
#include "libgpu/shared_device_buffer.h"
#include <libutils/fast_random.h>
#include <libutils/misc.h>
#include <libutils/timer.h>


constexpr int kBenchmarkingIters = 10;
constexpr int kValuesPerWorkItem = 64;
constexpr int kWorkGroupSize = 128;

template<typename T>
void raiseFail(const T& a, const T& b, std::string message, std::string filename, int line) {
    if (a != b) {
        std::cerr << message << " But " << a << " != " << b << ", " << filename << ":" << line << std::endl;
        throw std::runtime_error(message);
    }
}

#define EXPECT_THE_SAME(a, b, message) raiseFail(a, b, message, __FILE__, __LINE__)

void runGPUSum(const std::string& func_name, const std::string& method_name, unsigned int work_size,
               unsigned int array_size) {
    ocl::Kernel kernel(sum_kernel, sum_kernel_length, func_name);

    std::vector<unsigned int> as(array_size, 0);
    FastRandom r(42);
    unsigned int reference_sum = 0;
    for (int i = 0; i < array_size; ++i) {
        as[i] = (unsigned int) r.next(0, std::numeric_limits<unsigned int>::max() / array_size);
        reference_sum += as[i];
    }

    gpu::gpu_mem_32u sum_gpu;
    gpu::gpu_mem_32u as_gpu;
    sum_gpu.resizeN(1);
    as_gpu.resizeN(array_size);
    as_gpu.writeN(as.data(), array_size);
    kernel.compile();

    timer t;
    for (int iter = 0; iter < kBenchmarkingIters; ++iter) {
        unsigned int sum = 0;
        sum_gpu.writeN(&sum, 1);
        kernel.exec(gpu::WorkSize(kWorkGroupSize, work_size), as_gpu, array_size, sum_gpu);
        sum_gpu.readN(&sum, 1);
        EXPECT_THE_SAME(reference_sum, sum, "GPU result should be consistent!");
        t.nextLap();
    }
    std::cout << "\nGPU " << method_name << ":     " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
    std::cout << "GPU " << method_name << ":     " << (array_size / 1000.0 / 1000.0) / t.lapAvg() << " millions/s"
              << std::endl;
}

int main(int argc, char** argv) {

    unsigned int reference_sum = 0;
    unsigned int n = 100 * 1000 * 1000;
    std::vector<unsigned int> as(n, 0);
    FastRandom r(42);
    for (int i = 0; i < n; ++i) {
        as[i] = (unsigned int) r.next(0, std::numeric_limits<unsigned int>::max() / n);
        reference_sum += as[i];
    }

    {
        timer t;
        for (int iter = 0; iter < kBenchmarkingIters; ++iter) {
            unsigned int sum = 0;
            for (int i = 0; i < n; ++i) {
                sum += as[i];
            }
            EXPECT_THE_SAME(reference_sum, sum, "CPU result should be consistent!");
            t.nextLap();
        }
        std::cout << "CPU:     " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "CPU:     " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;
    }

    {
        timer t;
        for (int iter = 0; iter < kBenchmarkingIters; ++iter) {
            unsigned int sum = 0;
#pragma omp parallel for reduction(+ : sum)
            for (int i = 0; i < n; ++i) {
                sum += as[i];
            }
            EXPECT_THE_SAME(reference_sum, sum, "CPU OpenMP result should be consistent!");
            t.nextLap();
        }
        std::cout << "CPU OMP: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "CPU OMP: " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;
    }

    {
        // TODO: implement on OpenCL
        // gpu::Device device = gpu::chooseGPUDevice(argc, argv);
        gpu::Device device = gpu::chooseGPUDevice(argc, argv);

        gpu::Context context;
        context.init(device.device_id_opencl);
        context.activate();

        runGPUSum("sum_gpu_1", "baseline", n, n);
        runGPUSum("sum_gpu_2", "simple_for", n / kValuesPerWorkItem, n);
        runGPUSum("sum_gpu_3", "coalesced_for", n / kValuesPerWorkItem, n);
        runGPUSum("sum_gpu_4", "local_mem", n, n);

        {
            ocl::Kernel tree_sum_kernel(sum_kernel, sum_kernel_length, "sum_gpu_5");
            tree_sum_kernel.compile();

            gpu::gpu_mem_32u as_gpu;
            gpu::gpu_mem_32u bs_gpu;

            as_gpu.resizeN(n);
            bs_gpu.resizeN(n);
            as_gpu.writeN(as.data(), n);
            bs_gpu.writeN(as.data(), n);

            timer t;
            for (int iter = 0; iter < kBenchmarkingIters; ++iter) {
                unsigned int res = 0;
                t.reset();
                as_gpu.writeN(as.data(), n);
                bs_gpu.writeN(as.data(), n);
                t.start();
                for (int current_n = n; current_n > 1; current_n = (current_n + kWorkGroupSize - 1) / kWorkGroupSize) {
                    tree_sum_kernel.exec(gpu::WorkSize(kWorkGroupSize, (current_n + kWorkGroupSize - 1) /
                                                                               kWorkGroupSize * kWorkGroupSize),
                                         as_gpu, current_n, bs_gpu);
                    std::swap(as_gpu, bs_gpu);
                }
                as_gpu.readN(&res, 1);
                EXPECT_THE_SAME(reference_sum, res, "GPU result should be constistent!");
                t.nextLap();
            }
            std::cout << "\nGPU tree: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
            std::cout << "GPU tree: " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;
        }
    }
    return 0;
}
