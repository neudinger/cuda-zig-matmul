# CUDA Zig Matmul

Standalone Zig/CUDA BF16 and FP16 matrix multiply in `/home/neudinger/cuda-zig-matmul`.

The project uses:

- Zig NVPTX kernel sources at `src/matmul_kernel.zig` and `src/matmul_kernel_f16.zig`;
- a Zig host runner at `src/main.zig`;
- CUDA Driver API via `libcuda.so.1`;
- optional cuBLAS comparison via `libcublas.so`;
- Bazel to build a pinned Zig toolchain and the runner.

The CUDA kernels are emitted directly as Zig NVPTX assembly/PTX through Bazel's
`asm` output group. The repo builds a patched source Zig toolchain so NVPTX
kernel exports are emitted without LLVM aliases.

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
