#pragma once

#include <vector>
#include <complex>
#include <span>
#include <ranges>
#include <numbers>

namespace BasicFFT {

    namespace {
        template <bool Inverse = false>
        std::vector<std::complex<double>> fft_loop(std::span<std::complex<double>> data, int N, int step, int offset) {
            if (N == 2) {
                return {
                    data[offset] + data[offset + step],
                    data[offset] - data[offset + step]
                };
            }
         
            auto even = fft_loop<Inverse>(data, N / 2, step * 2, offset);
            auto odd = fft_loop<Inverse>(data, N / 2, step * 2, offset + step);
            
            auto res = std::vector<std::complex<double>>(N);
            for (auto k = 0; k < N / 2; k++) {
                auto a = even[k];
                auto b = odd[k];
                auto factor = exp(std::complex<double>{0, std::numbers::pi * k / N * (Inverse ? 2 : -2) });

                res[k] = a + factor * b;
                res[k + N / 2] = a - factor * b;
            }
            return res;
        }
    }

    std::vector<std::complex<double>> fft(std::span<std::complex<double>> data) {
        return fft_loop(data, data.size(), 1, 0);
    }

    auto ifft(std::span<std::complex<double>> data) {
        return fft_loop<true>(data, data.size(), 1, 0)
            | std::views::transform([N=(double)data.size()](auto&& x) -> std::complex<double> { return x / N; });
    }

}