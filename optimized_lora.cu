#include <torch/extension.h>
#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>

#include <cstdint>

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT32(x) TORCH_CHECK((x).scalar_type() == at::kFloat, #x " must be float32")
#define CHECK_DIM(x, d) TORCH_CHECK((x).dim() == (d), #x " must have dimension " #d)
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);        \
  CHECK_CONTIGUOUS(x);  \
  CHECK_FLOAT32(x)

namespace {

constexpr int kRank = 16;

inline int64_t checked_d(torch::Tensor W,
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

  TORCH_CHECK(W.device() == X.device(), "W and X must be on the same device");
  TORCH_CHECK(W.device() == A.device(), "W and A must be on the same device");
  TORCH_CHECK(W.device() == B.device(), "W and B must be on the same device");

  const auto d = W.size(0);
  TORCH_CHECK(W.size(1) == d, "W must be square [d, d]");
  TORCH_CHECK(X.size(0) == d && X.size(1) == d, "X must be [d, d]");
  TORCH_CHECK(A.size(0) == d && A.size(1) == kRank, "A must be [d, 16]");
  TORCH_CHECK(B.size(0) == d && B.size(1) == kRank, "B must be [d, 16]");
  return d;
}

}  // namespace

torch::Tensor forward(torch::Tensor W,
                      torch::Tensor X,
                      torch::Tensor A,
                      torch::Tensor B) {
  const auto d = checked_d(W, X, A, B);
  (void)d;
  c10::cuda::CUDAGuard device_guard(W.device());
  torch::NoGradGuard no_grad;

  auto Y = at::mm(W, X);
  auto T = at::mm(B.transpose(0, 1), X);
  Y.addmm_(A, T, 1.0, 1.0);
  return Y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "Phase 2 optimized LoRA forward");
}
