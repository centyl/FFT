#pragma once

#include <array>
#include <algorithm>

namespace OptimizedFFT {

    namespace {
        const auto maxSupportedSamples = 4096;

        auto temp = std::vector<std::complex<double>>(maxSupportedSamples / 2);

        auto factors = std::array<std::complex<double>, maxSupportedSamples / 2> {};

        template<int N>
        constexpr std::complex<double> getFactor(int k) {
            return factors[k * maxSupportedSamples / N];
        }

    }

    template<int N, int M>
    constexpr void fft(std::span<std::complex<double>> data) {
        if constexpr (N == 4) {
             auto a1 = data[0] + data[2 * M / N];
             auto b1 = data[M / N] + data[3 * M / N];

             auto a2 = data[0] - data[2 * M / N];
             auto b2 = data[M / N] - data[3 * M / N];

             data[0] = a1 + b1;
             data[2 * M / N] = a1 - b1;

             data[M / N] = a2 + getFactor<N>(1) * b2;
             data[3 * M / N] = a2 - getFactor<N>(1) * b2;
         }
         else {
             fft<N / 2, M>(data);
             fft<N / 2, M>(data.subspan(M / N));

             for (auto k = 0; k < N / 2; k++) {
                 auto a = data[2 * k * M / N];
                 auto f_b = getFactor<N>(k) * data[2 * k * M / N + M / N];
                 data[k * M / N] = a + f_b;
                 temp[k] = a - f_b;
             }

             for (auto i = 0; i < N / 2; i++)
                 data[M / 2 + i * M / N] = temp[i];
         }
    }

    void initialize() {
        auto startNum = std::complex<double> { 1,0 };
        factors[0] = startNum;
        std::generate(std::begin(factors) + 1, std::end(factors), [&startNum] { return startNum *= exp(std::complex<double>{0, std::numbers::pi / maxSupportedSamples * (-2)}); });
    }

}