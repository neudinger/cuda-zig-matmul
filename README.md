# CUDA Zig Matmul

Standalone Zig/CUDA BF16 matrix multiply in `/home/neudinger/cuda-zig-matmul`.

The project uses:

- a Zig NVPTX kernel source at `src/matmul_kernel.zig`;
- a Zig host runner at `src/main.zig`;
- CUDA Driver API via `libcuda.so.1`;
- optional cuBLAS comparison via `libcublas.so`;
- Bazel to build a pinned Zig toolchain and the runner.

The CUDA kernel is emitted as LLVM IR first, then converted to PTX with
`/usr/lib/llvm-21/bin/llc`. This avoids the current upstream Zig NVPTX export
alias crash while keeping the kernel source Zig-only.

## Commands

```bash
bazel build //...
bazel build //:matmul_ptx_cuda
bazel run //:cuda_matmul -- --list-devices
bazel run //:cuda_matmul -- --impl=zig --m=1024 --n=1024 --k=1024
bazel run //:cuda_matmul -- --impl=both --m=1024 --n=1024 --k=1024
```

## CLI

- `--impl=zig`: run the Zig BF16 tensor-core kernel.
- `--impl=cublas`: run cuBLAS BF16 GEMM only.
- `--impl=both`: run and validate both paths.
- `--m=<n> --n=<n> --k=<n>`: matrix dimensions.
- `--iters=<n>`: timed dispatch iterations, default `50`.
- `--warmup=<n>`: warmup dispatch iterations, default `5`.
- `--device=<n>`: CUDA device index, default `0`.
- `--list-devices`: print CUDA devices and exit.

The Zig kernel requires `m % 64 == 0`, `n % 64 == 0`, and `k % 16 == 0`.
Use `--impl=cublas` for arbitrary BF16 dimensions.
