from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LoraCandidate:
    name: str
    description: str
    source: str


def fallback_candidate() -> LoraCandidate:
    return LoraCandidate(
        name="aten_addmm_fallback",
        description=(
            "ATen/cuBLAS implementation using mm for W@X and B^T@X, then addmm "
            "to accumulate the rank-16 LoRA term."
        ),
        source=_aten_addmm_source(),
    )


def candidate_suite() -> list[LoraCandidate]:
    candidates = [
        fallback_candidate(),
        _aten_fused_lowrank_source("aten_fused_lowrank_32x8", block_x=32, block_y=8),
        _aten_fused_lowrank_source("aten_fused_lowrank_64x4", block_x=64, block_y=4),
        _aten_fused_lowrank_source("aten_fused_lowrank_16x16", block_x=16, block_y=16),
        _cublas_fused_lowrank_source(
            "cublas_fused_lowrank_default",
            block_x=32,
            block_y=8,
            math_mode="default",
        ),
        _cublas_fused_lowrank_source(
            "cublas_fused_lowrank_pedantic",
            block_x=32,
            block_y=8,
            math_mode="pedantic",
        ),
    ]
    return candidates


def _checks_prelude(extra_includes: str = "") -> str:
    return f"""#include <torch/extension.h>
#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
{extra_includes}

#include <cstdint>

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT32(x) TORCH_CHECK((x).scalar_type() == at::kFloat, #x " must be float32")
#define CHECK_DIM(x, d) TORCH_CHECK((x).dim() == (d), #x " must have dimension " #d)
#define CHECK_INPUT(x) \\
  CHECK_CUDA(x);        \\
  CHECK_CONTIGUOUS(x);  \\
  CHECK_FLOAT32(x)

namespace {{

constexpr int kRank = 16;

inline int64_t checked_d(torch::Tensor W,
                         torch::Tensor X,
                         torch::Tensor A,
                         torch::Tensor B) {{
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
}}

}}  // namespace
"""


def _module_footer() -> str:
    return """
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "Phase 2 optimized LoRA forward");
}
"""


def _aten_addmm_source() -> str:
    return (
        _checks_prelude()
        + r"""
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
  return at::addmm(Y, A, T, 1.0, 1.0);
}
"""
        + _module_footer()
    )


def _aten_fused_lowrank_source(name: str, *, block_x: int, block_y: int) -> LoraCandidate:
    source = (
        _checks_prelude(
            extra_includes="#include <ATen/cuda/CUDAContext.h>\n#include <c10/cuda/CUDAException.h>"
        )
        + f"""
namespace {{

constexpr int kBlockX = {block_x};
constexpr int kBlockY = {block_y};

__global__ void add_rank16_kernel(const float* __restrict__ A,
                                  const float* __restrict__ T,
                                  float* __restrict__ Y,
                                  int d) {{
  const int col = blockIdx.x * kBlockX + threadIdx.x;
  const int row = blockIdx.y * kBlockY + threadIdx.y;
  if (row >= d || col >= d) {{
    return;
  }}

  const float* a_row = A + row * kRank;
  float acc = 0.0f;
#pragma unroll
  for (int rr = 0; rr < kRank; ++rr) {{
    acc = fmaf(a_row[rr], T[rr * d + col], acc);
  }}
  Y[row * d + col] += acc;
}}

}}  // namespace

torch::Tensor forward(torch::Tensor W,
                      torch::Tensor X,
                      torch::Tensor A,
                      torch::Tensor B) {{
  const int d = static_cast<int>(checked_d(W, X, A, B));
  c10::cuda::CUDAGuard device_guard(W.device());
  torch::NoGradGuard no_grad;

  auto Y = at::mm(W, X);
  auto T = at::mm(B.transpose(0, 1), X);

  dim3 block(kBlockX, kBlockY);
  dim3 grid((d + kBlockX - 1) / kBlockX, (d + kBlockY - 1) / kBlockY);
  auto stream = at::cuda::getCurrentCUDAStream();
  add_rank16_kernel<<<grid, block, 0, stream>>>(
      A.data_ptr<float>(),
      T.data_ptr<float>(),
      Y.data_ptr<float>(),
      d);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return Y;
}}
"""
        + _module_footer()
    )
    return LoraCandidate(
        name=name,
        description=(
            f"ATen mm for the two large products plus a custom rank-16 add kernel "
            f"with {block_x}x{block_y} threads per block."
        ),
        source=source,
    )


def _cublas_fused_lowrank_source(
    name: str,
    *,
    block_x: int,
    block_y: int,
    math_mode: str,
) -> LoraCandidate:
    if math_mode not in {"default", "pedantic"}:
        raise ValueError(f"unsupported math mode: {math_mode}")

    mode_body = ""
    if math_mode == "pedantic":
        mode_body = """
#if defined(CUBLAS_PEDANTIC_MATH)
  CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH));
#endif
"""

    source = (
        _checks_prelude(
            extra_includes=(
                "#include <ATen/cuda/CUDAContext.h>\n"
                "#include <c10/cuda/CUDAException.h>\n"
                "#include <cublas_v2.h>"
            )
        )
        + f"""
namespace {{

constexpr int kBlockX = {block_x};
constexpr int kBlockY = {block_y};

#define CUBLAS_CHECK(expr)                                             \\
  do {{                                                                \\
    cublasStatus_t status = (expr);                                    \\
    TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS,                       \\
                "cuBLAS call failed with status ", static_cast<int>(status)); \\
  }} while (0)

__global__ void add_rank16_kernel(const float* __restrict__ A,
                                  const float* __restrict__ T,
                                  float* __restrict__ Y,
                                  int d) {{
  const int col = blockIdx.x * kBlockX + threadIdx.x;
  const int row = blockIdx.y * kBlockY + threadIdx.y;
  if (row >= d || col >= d) {{
    return;
  }}

  const float* a_row = A + row * kRank;
  float acc = 0.0f;
#pragma unroll
  for (int rr = 0; rr < kRank; ++rr) {{
    acc = fmaf(a_row[rr], T[rr * d + col], acc);
  }}
  Y[row * d + col] += acc;
}}

}}  // namespace

torch::Tensor forward(torch::Tensor W,
                      torch::Tensor X,
                      torch::Tensor A,
                      torch::Tensor B) {{
  const int d = static_cast<int>(checked_d(W, X, A, B));
  c10::cuda::CUDAGuard device_guard(W.device());
  torch::NoGradGuard no_grad;

  auto Y = torch::empty({{d, d}}, W.options());
  auto T = torch::empty({{kRank, d}}, W.options());

  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  auto stream = at::cuda::getCurrentCUDAStream();
  CUBLAS_CHECK(cublasSetStream(handle, stream.stream()));

  cublasMath_t old_mode;
  CUBLAS_CHECK(cublasGetMathMode(handle, &old_mode));
{mode_body}

  const float one = 1.0f;
  const float zero = 0.0f;

  // Row-major Y = W @ X. The byte layout is interpreted as column-major Y^T.
  CUBLAS_CHECK(cublasSgemm(
      handle,
      CUBLAS_OP_N,
      CUBLAS_OP_N,
      d,
      d,
      d,
      &one,
      X.data_ptr<float>(),
      d,
      W.data_ptr<float>(),
      d,
      &zero,
      Y.data_ptr<float>(),
      d));

  // Row-major T = B^T @ X, with B stored as a column-major 16 x d matrix.
  CUBLAS_CHECK(cublasSgemm(
      handle,
      CUBLAS_OP_N,
      CUBLAS_OP_T,
      d,
      kRank,
      d,
      &one,
      X.data_ptr<float>(),
      d,
      B.data_ptr<float>(),
      kRank,
      &zero,
      T.data_ptr<float>(),
      d));

  CUBLAS_CHECK(cublasSetMathMode(handle, old_mode));

  dim3 block(kBlockX, kBlockY);
  dim3 grid((d + kBlockX - 1) / kBlockX, (d + kBlockY - 1) / kBlockY);
  add_rank16_kernel<<<grid, block, 0, stream>>>(
      A.data_ptr<float>(),
      T.data_ptr<float>(),
      Y.data_ptr<float>(),
      d);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return Y;
}}
"""
        + _module_footer()
    )
    return LoraCandidate(
        name=name,
        description=(
            f"Direct cuBLAS SGEMM for W@X and B^T@X plus a custom rank-16 add "
            f"kernel; cuBLAS math mode={math_mode}."
        ),
        source=source,
    )
