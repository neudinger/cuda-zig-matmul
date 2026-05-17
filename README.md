# CUDA Zig Matmul

Standalone Zig/CUDA BF16 and FP16 matrix multiply in `/home/neudinger/cuda-zig-matmul`.

The project uses:

- Zig NVPTX kernel sources at `src/matmul_kernel.zig` and `src/matmul_kernel_f16.zig`;
- a Zig host runner at `src/main.zig`;
- CUDA Driver API via `libcuda.so.1`;
- optional cuBLAS comparison via `libcublas.so`;
- Bazel with pinned Zig and LLVM toolchains.

The CUDA kernels are emitted by a small Bazel rule that runs the pinned prebuilt
Zig compiler to produce LLVM IR for `nvptx64-cuda-none`, normalizes Zig's NVPTX
export alias in that IR, and then runs the pinned Bazel LLVM clang toolchain to
produce PTX.

## Hermeticity

Bazel is configured to use the pinned Bazel LLVM C/C++ toolchain for Linux
x86_64 builds and to avoid Bazel's auto-detected host C/C++ toolchain. Zig comes
from the pinned `rules_zig` index entry in `bazel/zig_index.json`; the kernel
PTX path does not require a patched source-built Zig compiler. The remaining
host requirements are runtime CUDA libraries loaded by the runner
(`libcuda.so.1` and, for `--impl=cublas`/`--impl=both`, `libcublas.so`).

One practical non-hermetic residual remains: Zig still writes local/global cache
state under `/tmp/cuda-zig-matmul-zig-cache`, so the sandbox mounts only that
cache path.

## Commands

```bash
bazel build //...
bazel build //:matmul_ptx_cuda
bazel build //:matmul_f16_ptx_cuda
bazel run //:cuda_matmul -- --list-devices
bazel run //:cuda_matmul -- --impl=zig --m=1024 --n=1024 --k=1024
bazel run //:cuda_matmul -- --impl=both --m=1024 --n=1024 --k=1024
bazel run //:cuda_matmul_f16 -- --impl=zig --m=1024 --n=1024 --k=1024
bazel run //:cuda_matmul -- --math=f16 --impl=both --m=1024 --n=1024 --k=1024
```

## CLI

- `--impl=zig`: run the Zig tensor-core kernel.
- `--impl=cublas`: run cuBLAS GEMM only.
- `--impl=both`: run and validate both paths.
- `--math=bf16`: BF16 inputs, F32 accumulation, BF16 output. This is the default.
- `--math=f16`: FP16 inputs, F32 accumulation, F32 output, matching the Vulkan FP16 output path.
- `--m=<n> --n=<n> --k=<n>`: matrix dimensions.
- `--iters=<n>`: timed dispatch iterations, default `50`.
- `--warmup=<n>`: warmup dispatch iterations, default `5`.
- `--device=<n>`: CUDA device index, default `0`.
- `--list-devices`: print CUDA devices and exit.

The Zig kernels require `m % 64 == 0`, `n % 64 == 0`, and `k % 16 == 0`.
Use `--impl=cublas` for arbitrary dimensions.
