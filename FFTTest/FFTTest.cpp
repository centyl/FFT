#include "pch.h"
#include "CppUnitTest.h"

#include <random>

#include "BasicFFT.h"
#include "OptimizedFFT.h"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace FFTTest {
    using namespace std;

    auto close_enough = [](complex<double> a, complex<double> b) -> bool {
        return abs(a.real() - b.real()) < 0.005
            && abs(a.imag() - b.imag()) < 0.005; };

    TEST_CLASS(FFTTest) {
    public:

    TEST_METHOD(BasicFFTResultIsCorrect) {
        auto input = vector<complex<double>> { 0,1,2,3 };

        auto result = BasicFFT::fft(input);

        auto expected = vector<complex<double>> { {6,0}, {-2,2},{-2,0},{-2,-2 } };

        Assert::IsTrue(ranges::equal(result, expected, close_enough));
    }

    TEST_METHOD(OptimizedResultEqualToBasic) {
        auto rd = random_device {};
        auto gen = mt19937 { rd() };
        auto dist = uniform_real_distribution<>(0.0, 100.0);

        const auto size = 512;

        auto input = vector<complex<double>> { };

        auto input_2 = array<complex<double>, size>{};

        for (auto i = 0; i < size; i++) {
            input_2[i] = dist(gen);
            input.push_back(input_2[i]);
        }

        auto result = BasicFFT::fft(input);

        OptimizedFFT::initialize();
        OptimizedFFT::fft<size, size>(input_2);

        Assert::IsTrue(ranges::equal(result, input_2, close_enough));
    }

    TEST_METHOD(BasicIFFTWorks) {
        auto input = vector<complex<double>> { 0,1,2,3 };

        auto fftResult = BasicFFT::fft(input);

        Assert::IsTrue(ranges::equal(input, BasicFFT::ifft(fftResult), close_enough));
    }

    };
}
