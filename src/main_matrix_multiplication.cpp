#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <libutils/fast_random.h>
#include <libutils/misc.h>
#include <libutils/timer.h>

#include "cl/matrix_multiplication_cl.h"

#include <iostream>
#include <stdexcept>
#include <vector>


static constexpr unsigned int kWPT = 8;
static constexpr unsigned int kWorkGroupSize = 16;

std::ostream& bold_on(std::ostream& os) {
    return os << "\e[1m";
}

std::ostream& bold_off(std::ostream& os) {
    return os << "\e[0m";
}

template<typename T>
T round_up(T first, T second) {
    return (first - 1 + second) / second;
}

template<typename T>
T fit(T first, T second) {
    return round_up(first, second) * second;
}

int main(int argc, char** argv) {
    gpu::Device device = gpu::chooseGPUDevice(argc, argv);

    gpu::Context context;
    context.init(device.device_id_opencl);
    context.activate();

    int benchmarkingIters = 10; // TODO пока тестируетесь удобно выставить единицу
    unsigned int M = 3;
    unsigned int K = 10;
    unsigned int N = 3;
    const size_t gflops =
            ((size_t) M * K * N * 2) / (1000 * 1000 * 1000);// умножить на два, т.к. операция сложения и умножения

    std::vector<float> as(M * K, 0);
    std::vector<float> bs(K * N, 0);
    std::vector<float> cs(M * N, 0);

    FastRandom r(M + K + N);
    for (unsigned int i = 0; i < as.size(); ++i) {
        as[i] = r.nextf();
    }
    for (unsigned int i = 0; i < bs.size(); ++i) {
        bs[i] = r.nextf();
    }
    std::cout << "Data generated for M=" << M << ", K=" << K << ", N=" << N << std::endl;

    {
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            for (int j = 0; j < M; ++j) {
                for (int i = 0; i < N; ++i) {
                    float sum = 0.0f;
                    for (int k = 0; k < K; ++k) {
                        sum += as.data()[j * K + k] * bs.data()[k * N + i];
                    }
                    cs.data()[j * N + i] = sum;
                }
            }
            t.nextLap();
        }
        std::cout << "CPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "CPU: " << gflops / t.lapAvg() << " GFlops\n" << std::endl;
    }

    const std::vector<float> cs_cpu_reference = cs;


    gpu::gpu_mem_32f as_gpu, bs_gpu, cs_gpu;
    as_gpu.resizeN(M * K);
    bs_gpu.resizeN(K * N);
    cs_gpu.resizeN(M * N);

    as_gpu.writeN(as.data(), M * K);
    bs_gpu.writeN(bs.data(), K * N);

    ocl::Kernel kernel1(matrix_multiplication, matrix_multiplication_length, "matrix_multiplication_naive");
    kernel1.compile();

    ocl::Kernel kernel2(matrix_multiplication, matrix_multiplication_length, "matrix_multiplication_local");
    kernel2.compile();

    ocl::Kernel kernel3(matrix_multiplication, matrix_multiplication_length, "matrix_multiplication_wpt");
    kernel3.compile();

    std::vector<std::tuple<std::string, ocl::Kernel*, gpu::WorkSize*>> programs;

    gpu::WorkSize ws1(kWorkGroupSize, kWorkGroupSize, fit(N, kWorkGroupSize), fit(M, kWorkGroupSize));
    gpu::WorkSize ws2(kWorkGroupSize, kWorkGroupSize, fit(N, kWorkGroupSize), fit(M, kWorkGroupSize));
    gpu::WorkSize ws3(kWorkGroupSize, round_up(kWorkGroupSize, kWPT), M, round_up(K, kWPT));

    programs.emplace_back("naive matrix multiplication", &kernel1, &ws1);
    programs.emplace_back("matrix multiplication with local mem", &kernel2, &ws2);
    programs.emplace_back("matrix multiplication with wpt optimization", &kernel3, &ws3);


    for (const auto& [name, kernel, ws] : programs) {
        bold_on(std::cout);
        std::cout << "Testing " << name << '\n';
        bold_off(std::cout);
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            // TODO
            kernel->exec(*ws, as_gpu, bs_gpu, cs_gpu, M, K, N);

            t.nextLap();
        }
        std::cout << "GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU: " << gflops / t.lapAvg() << " GFlops" << std::endl;

        cs_gpu.readN(cs.data(), M * N);

        // Проверяем корректность результатов
        double diff_sum = 0;
        for (int i = 0; i < M * N; ++i) {
            double a = cs[i];
            double b = cs_cpu_reference[i];
            if (a != 0.0 || b != 0.0) {
                double diff = fabs(a - b) / std::max(fabs(a), fabs(b));
                diff_sum += diff;
            }
        }

        double diff_avg = diff_sum / (M * N);
        std::cout << "Average difference: " << diff_avg * 100.0 << "%\n" << std::endl;
        if (diff_avg > 0.01) {
            std::cerr << "Too big difference!" << std::endl;
            return 1;
        }
    }

    return 0;
}
