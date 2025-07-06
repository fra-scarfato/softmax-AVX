#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <limits>      
#include <hpc_helpers.hpp>
#include <avx_mathfun.h>

inline float hsum_sse3(__m128 v) {
	/* duplicate the odd-indexed elements of v and store in shuf 
	v:[a,b,c,d] -> shuf:[b,b,d,d] */
	__m128 shuf = _mm_movehdup_ps(v);	
	__m128 maxs = _mm_add_ps(v, shuf);
	/* move the upper 2 values of maxs into the lower 2 values of shuf 
	maxs:[a,b,c,d] -> shuf:[c,d,_,_] */
	shuf = _mm_movehl_ps(shuf, maxs);
	// sum the lower 2 values of maxs and shuf: [a+c,b+d,c,d]
	maxs = _mm_add_ss(maxs, shuf);
	// return the lower value of maxs
	return _mm_cvtss_f32(maxs);
}

// Return the sum of vector containing 8 floats
inline float hsum_avx(__m256 v) {
	// extract the lower half (lower 4 floats)
	__m128 lo = _mm256_castps256_ps128(v);
	// extract the higher half (upper 4 floats)
	__m128 hi = _mm256_extractf128_ps(v, 1);
	// sum the lower and the higher part
	lo = _mm_add_ps(lo, hi);
	return hsum_sse3(lo);
}

inline float hmax_sse3(__m128 v) {
	// duplicate the odd-indexed elements of v and store in shuf 
    __m128 shuf = _mm_movehdup_ps(v);
    __m128 maxs = _mm_max_ps(v, shuf);
	// move the upper 2 values of maxs into the lower 2 values of shuf 
    shuf = _mm_movehl_ps(shuf, maxs);
    // compute max between the lower 2 values of maxs and shuf and store it in the lower value of maxs
	maxs = _mm_max_ss(maxs, shuf);
	// return the lower value of maxs
    return _mm_cvtss_f32(maxs);
}

inline float hmax_avx(__m256 v) {
	// extract the lower half (lower 4 floats)
    __m128 lo = _mm256_castps256_ps128(v);
	// extract the higher half (upper 4 floats)
    __m128 hi = _mm256_extractf128_ps(v, 1);
	// compute the maximum values between lo and hi and return a vector (4 floats)
    lo = _mm_max_ps(lo, hi);
    return hmax_sse3(lo);
}

float compute_max_avx(const float *input, size_t K, size_t loop_bound) {
	size_t i;
	// 256-bit vector of -infty
	__m256 max_vec = _mm256_set1_ps(-INFINITY);

	// compute the max processing 8 elements at once
	for (i = 0; i < loop_bound; i += 8) {
		// load input in the 256-bit vector (8 floats)
		__m256 vec_avx = _mm256_loadu_ps(&input[i]);
		// compute the maximum values between max_vec and vec_avx (8 results)
		max_vec = _mm256_max_ps(max_vec, vec_avx);
	}

	// compute max between the 8 values of the vector
	float max_val = hmax_avx(max_vec);

	// remaining elements (if any)
	for (; i < K; i++) {
        max_val = std::max(max_val, input[i]);
    }

	return max_val;
}

void softmax_avx(const float *input, float *output, size_t K) {
	// index for the loop to handle cases in which the length is not multiple of 8
	size_t i;
	// loop bound in case K is not multiple of 8
	size_t loop_bound = K - (K % 8);
	
	float max_val = compute_max_avx(input, K, loop_bound);

	// 256-bit vector in which every elements is max_val
	__m256 maxs = _mm256_set1_ps(max_val);
	// 256-bit vector of zeros
	__m256 sum_avx = _mm256_setzero_ps();

	for (i = 0; i < loop_bound; i += 8) {
		// load input in the 256-bit vector (8 floats)
		__m256 vec_avx = _mm256_loadu_ps(&input[i]);
		// normalize with (input - max) 
		__m256 temp = _mm256_sub_ps(vec_avx, maxs);
		// exponential of (input-max)
		__m256 exp_avx = exp256_ps(temp);
		// store the exponential values in the output vector
		_mm256_storeu_ps(&output[i], exp_avx);
		// sum the exponential
		sum_avx = _mm256_add_ps(sum_avx, exp_avx);
	}

	// compute the sum between the 8 values of the vector
	float sum = hsum_avx(sum_avx);

	// remaining elements (if any)
	for (; i < K; i++) {
        float exp_val = std::exp(input[i] - max_val);
        output[i] = exp_val;
        sum += exp_val;
    }

	// 256-bit vector in which every elements is the sum computed previously
	__m256 sums = _mm256_set1_ps(sum);

	for (i = 0; i < loop_bound; i += 8) {
		// load exponential values in the 256-bit vector (8 floats)
		__m256 output_avx = _mm256_loadu_ps(&output[i]);
		// compute (exp / sum)
		__m256 temp = _mm256_div_ps(output_avx, sums);
		// store the results in output
		_mm256_storeu_ps(&output[i], temp);
	}

	// remaining elements (if any)
	for (; i < K; i++) {
        output[i] /= sum;
    }
}

std::vector<float> generate_random_input(size_t K, float min = -1.0f, float max = 1.0f) {
    std::vector<float> input(K);
    //std::random_device rd;
    //std::mt19937 gen(rd());
	std::mt19937 gen(5489); // fixed seed for reproducible results
    std::uniform_real_distribution<float> dis(min, max);
    for (size_t i = 0; i < K; ++i) {
        input[i] = dis(gen);
    }
    return input;
}

void printResult(std::vector<float> &v, size_t K) {
	for(size_t i=0; i<K; ++i) {
		std::fprintf(stderr, "%f\n",v[i]);
	}
}


int main(int argc, char *argv[]) {
	if (argc == 1) {
		std::printf("use: %s K [1]\n", argv[0]);
		return 0;		
	}
	size_t K=0;
	if (argc >= 2) {
		K = std::stol(argv[1]);
	}
	bool print=false;
	if (argc == 3) {
		print=true;
	}	
	std::vector<float> input=generate_random_input(K);
	std::vector<float> output(K);

	TIMERSTART(softime_avx);
	softmax_avx(input.data(), output.data(), K);
	TIMERSTOP(softime_avx);
	
	// print the results on the standard output
	if (print) {
		printResult(output, K);
	}
}

