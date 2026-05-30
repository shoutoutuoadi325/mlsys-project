#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT32(x) TORCH_CHECK((x).scalar_type() == at::kFloat, #x " must be float32")
#define CHECK_DIM(x, d) TORCH_CHECK((x).dim() == (d), #x " must have dimension " #d)
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);        \
  CHECK_CONTIGUOUS(x);  \
  CHECK_FLOAT32(x)

namespace {

constexpr int kTile = 16;

__global__ void bt_x_kernel(const float* __restrict__ B,
                            const float* __restrict__ X,
                            float* __restrict__ T,
                            int d,
                            int n,
                            int r) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row >= r || col >= n) {
    return;
  }

  float sum = 0.0f;
  for (int i = 0; i < d; ++i) {
    sum += B[i * r + row] * X[i * n + col];
  }
  T[row * n + col] = sum;
}

__global__ void fused_wx_plus_at_kernel(const float* __restrict__ W,
                                        const float* __restrict__ X,
                                        const float* __restrict__ A,
                                        const float* __restrict__ T,
                                        float* __restrict__ Y,
                                        int d,
                                        int n,
                                        int r) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row >= d || col >= n) {
    return;
  }

  float sum = 0.0f;
  for (int k = 0; k < d; ++k) {
    sum += W[row * d + k] * X[k * n + col];
  }
  for (int rr = 0; rr < r; ++rr) {
    sum += A[row * r + rr] * T[rr * n + col];
  }
  Y[row * n + col] = sum;
}

}  // namespace

torch::Tensor forward(torch::Tensor W,
                      torch::Tensor X,
                      torch::Tensor A,
                      torch::Tensor B) {
  CHECK_INPUT(W);
  CHECK_INPUT(X);
  CHECK_INPUT(A);
  CHECK_INPUT(B);
  CHECK_DIM(W, 2);
  CHECK_DIM(X, 2);
  CHECK_DIM(A, 2);
  CHECK_DIM(B, 2);

  const auto d = static_cast<int>(W.size(0));
  TORCH_CHECK(W.size(1) == d, "W must be square [d, d]");
  TORCH_CHECK(X.size(0) == d && X.size(1) == d, "X must be [d, d]");
  TORCH_CHECK(A.size(0) == d, "A must be [d, r]");
  TORCH_CHECK(B.size(0) == d, "B must be [d, r]");
  TORCH_CHECK(A.size(1) == B.size(1), "A and B must share the same rank r");
  const auto r = static_cast<int>(A.size(1));

  auto opts = W.options();
  auto T = torch::zeros({r, d}, opts);
  auto Y = torch::zeros({d, d}, opts);

  dim3 block(kTile, kTile);
  dim3 grid_t((d + kTile - 1) / kTile, (r + kTile - 1) / kTile);
  dim3 grid_y((d + kTile - 1) / kTile, (d + kTile - 1) / kTile);

  auto stream = at::cuda::getDefaultCUDAStream();
  bt_x_kernel<<<grid_t, block, 0, stream>>>(
      B.data_ptr<float>(),
      X.data_ptr<float>(),
      T.data_ptr<float>(),
      d,
      d,
      r);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  fused_wx_plus_at_kernel<<<grid_y, block, 0, stream>>>(
      W.data_ptr<float>(),
      X.data_ptr<float>(),
      A.data_ptr<float>(),
      T.data_ptr<float>(),
      Y.data_ptr<float>(),
      d,
      d,
      r);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return Y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "Optimized LoRA forward (CUDA)");
}
