#include <iostream>
#include <chrono>
#include <format>

#include "BasicFFT.h"
#include "OptimizedFFT.h"

using namespace std;

void run_test() {
    const auto N = 1024;
    const auto S = 1000;
    cout << format("Testing 1D FFT of {} samples", N) << endl;
    auto a = array<complex<double>, N> {};
    auto t = ranges::iota_view(0, N)
        | views::transform([](auto&& x) -> complex<double> { return sin(2 * numbers::pi * x / 100.0); });
    ranges::copy(t, begin(a));
    auto time_1 = 0ll;
    for (auto i = 0; i < S; i++) {
        auto time_begin = chrono::high_resolution_clock::now();
        auto res = BasicFFT::fft(a);
        auto time_end = chrono::high_resolution_clock::now();
        time_1 += chrono::duration_cast<chrono::microseconds>(time_end - time_begin).count();
    }
    cout << "Basic algorithm" << endl;
    cout << format("{} FFTs completed in {} ms, average {} microseconds", S, time_1 / 1000.0, time_1 / S) << endl;
    
    auto a2 = array<complex<double>, N>{};
    auto time_2 = 0ll;
    for (auto i = 0; i < S; i++) {
        ranges::copy(t, begin(a));
        auto time_begin = chrono::high_resolution_clock::now();
        OptimizedFFT::fft<N,N>(a);
        auto time_end = chrono::high_resolution_clock::now();
        time_2 += chrono::duration_cast<chrono::microseconds>(time_end - time_begin).count();
    }
    cout << "Optimized algorithm" << endl;
    cout << format("{} FFTs completed in {} ms, average {} microseconds", S, time_2 / 1000.0, time_2 / S) << endl;

}

int main() {
    OptimizedFFT::initialize();
    run_test();
    return 0;
}