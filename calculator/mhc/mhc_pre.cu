// MIT License
//
// Copyright (c) 2026 CUI Xin (崔 欣)
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "mhc_pre.h"

#include <cmath>
#include <cstddef>

namespace cuinfer::mhc {
namespace {

constexpr int kMaxThreads = 512;

template <bool UsePtx>
__global__ void preKernel(const float *dot_mix, const float *square_sum,
                          const __nv_bfloat16 *residual, const float *scale,
                          const float *base, float *post_mix, float *comb_mix,
                          __nv_bfloat16 *layer_input, int tokens, int hidden,
                          PreOptions options, bool compute_square_sum) {
  const int token = blockIdx.x;
  const std::size_t residual_offset =
      static_cast<std::size_t>(token) * kHeads * hidden;
  __shared__ float partials[kMaxThreads];
  __shared__ float pre_mix[kHeads];

  if (compute_square_sum) {
    float sum = 0.0f;
    const std::size_t count = static_cast<std::size_t>(kHeads) * hidden;
    for (std::size_t i = threadIdx.x; i < count; i += blockDim.x) {
      const float value =
          detail::Arithmetic<UsePtx>::fromBf16(residual[residual_offset + i]);
      sum = detail::Arithmetic<UsePtx>::multiplyAdd(value, value, sum);
    }
    partials[threadIdx.x] = sum;
    __syncthreads();
    for (int step = blockDim.x / 2; step > 0; step /= 2) {
      if (threadIdx.x < step)
        partials[threadIdx.x] = detail::Arithmetic<UsePtx>::add(
            partials[threadIdx.x], partials[threadIdx.x + step]);
      __syncthreads();
    }
  }

  if (threadIdx.x == 0) {
    float logits[kMixValues] = {};
    float sum = compute_square_sum ? partials[0] : 0.0f;
    for (int split = 0; split < options.splits; ++split) {
      const std::size_t index =
          static_cast<std::size_t>(split) * tokens + token;
      if (!compute_square_sum)
        sum = detail::Arithmetic<UsePtx>::add(sum, square_sum[index]);
      for (int i = 0; i < kMixValues; ++i)
        logits[i] = detail::Arithmetic<UsePtx>::add(
            logits[i], dot_mix[index * kMixValues + i]);
    }
    const float rstd = detail::Arithmetic<UsePtx>::reciprocalSqrt(
        detail::Arithmetic<UsePtx>::add(
            detail::Arithmetic<UsePtx>::divide(sum, options.k),
            options.rms_eps));
    for (int head = 0; head < kHeads; ++head) {
      const float pre_value = logits[head] * rstd * scale[0] + base[head];
      const float post_value =
          logits[kHeads + head] * rstd * scale[1] + base[kHeads + head];
      pre_mix[head] = detail::Arithmetic<UsePtx>::add(
          detail::Arithmetic<UsePtx>::divide(1.0f, 1.0f + expf(-pre_value)),
          options.pre_eps);
      post_mix[static_cast<std::size_t>(token) * kHeads + head] =
          detail::Arithmetic<UsePtx>::divide(options.post_multiplier,
                                             1.0f + expf(-post_value));
    }

    float matrix[kHeads][kHeads];
    for (int row = 0; row < kHeads; ++row) {
      float max_value = -INFINITY;
      for (int col = 0; col < kHeads; ++col) {
        const int i = 2 * kHeads + row * kHeads + col;
        matrix[row][col] = logits[i] * rstd * scale[2] + base[i];
        max_value = fmaxf(max_value, matrix[row][col]);
      }
      float row_sum = 0.0f;
      for (int col = 0; col < kHeads; ++col) {
        matrix[row][col] = expf(matrix[row][col] - max_value);
        row_sum = detail::Arithmetic<UsePtx>::add(row_sum, matrix[row][col]);
      }
      for (int col = 0; col < kHeads; ++col)
        matrix[row][col] = detail::Arithmetic<UsePtx>::add(
            detail::Arithmetic<UsePtx>::divide(matrix[row][col], row_sum),
            options.sinkhorn_eps);
    }
    for (int iteration = 0; iteration < options.sinkhorn_repeat; ++iteration) {
      if (iteration > 0) {
        for (int row = 0; row < kHeads; ++row) {
          float sum = options.sinkhorn_eps;
          for (int col = 0; col < kHeads; ++col)
            sum = detail::Arithmetic<UsePtx>::add(sum, matrix[row][col]);
          for (int col = 0; col < kHeads; ++col)
            matrix[row][col] =
                detail::Arithmetic<UsePtx>::divide(matrix[row][col], sum);
        }
      }
      for (int col = 0; col < kHeads; ++col) {
        float sum = options.sinkhorn_eps;
        for (int row = 0; row < kHeads; ++row)
          sum = detail::Arithmetic<UsePtx>::add(sum, matrix[row][col]);
        for (int row = 0; row < kHeads; ++row)
          matrix[row][col] =
              detail::Arithmetic<UsePtx>::divide(matrix[row][col], sum);
      }
    }
    for (int row = 0; row < kHeads; ++row)
      for (int col = 0; col < kHeads; ++col)
        comb_mix[static_cast<std::size_t>(token) * kHeads * kHeads +
                 row * kHeads + col] = matrix[row][col];
  }
  __syncthreads();

  for (std::size_t h = threadIdx.x; h < static_cast<std::size_t>(hidden);
       h += blockDim.x) {
    float value = 0.0f;
    for (int head = 0; head < kHeads; ++head)
      value = detail::Arithmetic<UsePtx>::multiplyAdd(
          pre_mix[head],
          detail::Arithmetic<UsePtx>::fromBf16(
              residual[residual_offset +
                       static_cast<std::size_t>(head) * hidden + h]),
          value);
    layer_input[static_cast<std::size_t>(token) * hidden + h] =
        detail::Arithmetic<UsePtx>::toBf16(value);
  }
}

template <bool UsePtx>
cudaError_t launchPre(const float *dot_mix, const float *square_sum,
                      const __nv_bfloat16 *residual, const float *scale,
                      const float *base, float *post_mix, float *comb_mix,
                      __nv_bfloat16 *layer_input, int tokens, int hidden,
                      PreOptions options, bool compute_square_sum,
                      cudaStream_t stream) {
  if (tokens < 0 || hidden <= 0 || options.k <= 0 ||
      (options.splits != 1 && options.splits != 2 && options.splits != 4 &&
       options.splits != 8 && options.splits != 16) ||
      (compute_square_sum && options.splits != 1) ||
      options.sinkhorn_repeat < 1 || options.block_size < 0 ||
      !(options.rms_eps > 0.0f) || !std::isfinite(options.rms_eps) ||
      !(options.pre_eps > 0.0f) || !std::isfinite(options.pre_eps) ||
      !(options.sinkhorn_eps > 0.0f) || !std::isfinite(options.sinkhorn_eps) ||
      !std::isfinite(options.post_multiplier))
    return cudaErrorInvalidValue;
  if (tokens == 0)
    return cudaSuccess;
  if (!dot_mix || (!compute_square_sum && !square_sum) || !residual || !scale ||
      !base || !post_mix || !comb_mix || !layer_input)
    return cudaErrorInvalidValue;
  const int threads = options.block_size >= 512   ? 512
                      : options.block_size >= 256 ? 256
                                                  : 128;
  preKernel<UsePtx><<<tokens, threads, 0, stream>>>(
      dot_mix, square_sum, residual, scale, base, post_mix, comb_mix,
      layer_input, tokens, hidden, options, compute_square_sum);
  return cudaGetLastError();
}

} // namespace

cudaError_t pre(const float *dot_mix, const float *square_sum,
                const __nv_bfloat16 *residual, const float *scale,
                const float *base, float *post_mix, float *comb_mix,
                __nv_bfloat16 *layer_input, int tokens, int hidden,
                PreOptions options, bool compute_square_sum,
                cudaStream_t stream) {
  return launchPre<false>(dot_mix, square_sum, residual, scale, base, post_mix,
                          comb_mix, layer_input, tokens, hidden, options,
                          compute_square_sum, stream);
}

cudaError_t prePtx(const float *dot_mix, const float *square_sum,
                   const __nv_bfloat16 *residual, const float *scale,
                   const float *base, float *post_mix, float *comb_mix,
                   __nv_bfloat16 *layer_input, int tokens, int hidden,
                   PreOptions options, bool compute_square_sum,
                   cudaStream_t stream) {
  return launchPre<true>(dot_mix, square_sum, residual, scale, base, post_mix,
                         comb_mix, layer_input, tokens, hidden, options,
                         compute_square_sum, stream);
}

} // namespace cuinfer::mhc