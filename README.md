# Softmax Function Optimization with AVX

This project focuses on optimizing a standard **softmax function** written in C++. Starting from a baseline scalar implementation, two optimization approaches were developed and compared:

1.  **Manual Vectorization:** Using **AVX (Advanced Vector Extensions)** intrinsics to explicitly leverage SIMD (Single Instruction, Multiple Data) processing.
2.  **Auto-Vectorization:** Guiding the compiler to automatically generate vectorized code through specific compiler flags and pragmas.

The goal is to analyze the performance gains from each method and compare the development effort required.

***

## Implementations

Three versions of the softmax function were used in this analysis.

### 1. Baseline (Scalar) Version

This is the initial implementation provided. It calculates the softmax function in a straightforward, element-by-element manner.

**Algorithm:**
1.  Find the maximum value in the input vector (`logits`). This is done for numerical stability to prevent overflow when calculating the exponential.
2.  Calculate the sum of the exponentials of each element after subtracting the maximum value.
3.  Divide the exponential of each element by the calculated sum to get the final probability distribution.

This version is easy to read but computationally inefficient as it does not take advantage of modern CPU vector capabilities.

### 2. Manual AVX Vectorization

This version was manually rewritten to process 8 floating-point numbers (FP32) simultaneously using 256-bit AVX registers. It uses the provided `exp256_ps` function for vectorized exponential calculations.

**Vectorized Algorithm:**
1.  **Find Max Logit:** The input array is processed in chunks of 8. A vector of maximum values (`max_v`) is updated iteratively using the `_mm256_max_ps` intrinsic. After the loop, a horizontal reduction is performed on `max_v` to find the single scalar maximum value.
2.  **Subtract and Exponentiate:** The code iterates through the array again in chunks of 8. In each chunk:
    * The scalar max value is broadcasted into a vector (`_mm256_set1_ps`).
    * The max vector is subtracted from the logit vector (`_mm256_sub_ps`).
    * The exponential of the resulting vector is calculated using the provided `exp256_ps` function.
3.  **Sum Exponentials:** The results from `exp256_ps` are accumulated into a sum vector using `_mm256_add_ps`. After the loop, a horizontal add is performed on the sum vector to get the final scalar sum.
4.  **Final Division:** The code iterates one last time. In each chunk, the exponentiated vector (recalculated or stored from the previous step) is divided by the total sum using `_mm256_div_ps`.
5.  **Tail Handling:** If the input size is not a multiple of 8, the remaining elements are processed using the original scalar logic.

This implementation leverages **Fused Multiply-Add (FMA)** implicitly where applicable in the math functions and explicitly through AVX intrinsics, significantly reducing the number of instructions and clock cycles required.

### 3. Compiler Auto-Vectorization

This version uses the baseline scalar code but relies on the compiler's optimization capabilities. To enable effective auto-vectorization, the following steps were taken:

* **Compiler Flags:** The code was compiled with flags designed to enable aggressive optimizations and instruct the compiler to target the AVX instruction set.
    * `-O3`: Enables a high level of optimization.
    * `-mavx`: Allows the compiler to use AVX instructions.
    * `-ffast-math`: Relaxes strict IEEE compliance, which can allow for more aggressive floating-point optimizations like reordering operations.
* **Code Pragmas:** A `#pragma GCC ivdep` (loop vectorization dependency-ignore) or similar pragma was added before the critical loops. This pragma asserts to the compiler that there are no loop-carried dependencies, giving it confidence to vectorize the code.

This approach requires minimal code changes but cedes control over the final instruction selection to the compiler.

***

## How to Build and Run

The project can be compiled and executed using the provided `Makefile` or by running the commands manually.

**Using Makefile:**
```bash
# Compile all versions and the benchmark
make

# Run the benchmark
./softmax_benchmark
