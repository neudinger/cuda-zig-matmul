const std = @import("std");

const log = std.log.scoped(.cuda_matmul);

const CuResult = c_uint;
const CuDevice = c_int;
const CuDevicePtr = u64;
const CuContext = ?*anyopaque;
const CuModule = ?*anyopaque;
const CuFunction = ?*anyopaque;
const CublasHandle = ?*anyopaque;

const CLOCK_MONOTONIC: c_int = 1;
const Timespec = extern struct {
    tv_sec: isize,
    tv_nsec: isize,
};
extern fn clock_gettime(clk_id: c_int, tp: *Timespec) callconv(.c) c_int;

const CuInit = *const fn (c_uint) callconv(.c) CuResult;
const CuDeviceGet = *const fn (*CuDevice, c_int) callconv(.c) CuResult;
const CuDeviceGetCount = *const fn (*c_int) callconv(.c) CuResult;
const CuDeviceGetName = *const fn ([*]u8, c_int, CuDevice) callconv(.c) CuResult;
const CuDeviceComputeCapability = *const fn (*c_int, *c_int, CuDevice) callconv(.c) CuResult;
const CuCtxCreate = *const fn (*CuContext, c_uint, CuDevice) callconv(.c) CuResult;
const CuCtxDestroy = *const fn (CuContext) callconv(.c) CuResult;
const CuCtxSynchronize = *const fn () callconv(.c) CuResult;
const CuModuleLoadData = *const fn (*CuModule, ?*const anyopaque) callconv(.c) CuResult;
const CuModuleUnload = *const fn (CuModule) callconv(.c) CuResult;
const CuModuleGetFunction = *const fn (*CuFunction, CuModule, [*:0]const u8) callconv(.c) CuResult;
const CuMemAlloc = *const fn (*CuDevicePtr, usize) callconv(.c) CuResult;
const CuMemFree = *const fn (CuDevicePtr) callconv(.c) CuResult;
const CuMemcpyHtoD = *const fn (CuDevicePtr, ?*const anyopaque, usize) callconv(.c) CuResult;
const CuMemcpyDtoH = *const fn (?*anyopaque, CuDevicePtr, usize) callconv(.c) CuResult;
const CuLaunchKernel = *const fn (
    CuFunction,
    c_uint,
    c_uint,
    c_uint,
    c_uint,
    c_uint,
    c_uint,
    c_uint,
    ?*anyopaque,
    ?[*]?*anyopaque,
    ?[*]?*anyopaque,
) callconv(.c) CuResult;
const CuGetErrorString = *const fn (CuResult, *[*:0]const u8) callconv(.c) CuResult;

const CublasStatus = c_int;
const CublasCreate = *const fn (*CublasHandle) callconv(.c) CublasStatus;
const CublasDestroy = *const fn (CublasHandle) callconv(.c) CublasStatus;
const CublasGemmEx = *const fn (
    CublasHandle,
    c_int,
    c_int,
    c_int,
    c_int,
    c_int,
    *const anyopaque,
    *const anyopaque,
    c_int,
    c_int,
    *const anyopaque,
    c_int,
    c_int,
    *const anyopaque,
    *anyopaque,
    c_int,
    c_int,
    c_int,
    c_int,
) callconv(.c) CublasStatus;

const CUBLAS_OP_N = 0;
const CUDA_R_32F = 0;
const CUDA_R_16F = 2;
const CUDA_R_16BF = 14;
const CUBLAS_COMPUTE_32F_FAST_16F = 74;
const CUBLAS_COMPUTE_32F_FAST_16BF = 75;
const CUBLAS_GEMM_DEFAULT_TENSOR_OP = 99;

const Math = enum {
    bf16,
    f16,

    fn name(self: Math) []const u8 {
        return switch (self) {
            .bf16 => "bf16",
            .f16 => "f16",
        };
    }

    fn inputCudaType(self: Math) c_int {
        return switch (self) {
            .bf16 => CUDA_R_16BF,
            .f16 => CUDA_R_16F,
        };
    }

    fn outputCudaType(self: Math) c_int {
        return switch (self) {
            .bf16 => CUDA_R_16BF,
            .f16 => CUDA_R_32F,
        };
    }

    fn computeType(self: Math) c_int {
        return switch (self) {
            .bf16 => CUBLAS_COMPUTE_32F_FAST_16BF,
            .f16 => CUBLAS_COMPUTE_32F_FAST_16F,
        };
    }

    fn outputElementSize(self: Math) usize {
        return switch (self) {
            .bf16 => @sizeOf(u16),
            .f16 => @sizeOf(f32),
        };
    }
};

const Impl = enum {
    zig,
    cublas,
    both,

    fn runsZig(self: Impl) bool {
        return self == .zig or self == .both;
    }

    fn runsCublas(self: Impl) bool {
        return self == .cublas or self == .both;
    }
};

const Options = struct {
    ptx_path: ?[]const u8 = null,
    ptx_bf16_path: ?[]const u8 = null,
    ptx_f16_path: ?[]const u8 = null,
    impl: Impl = .zig,
    math: Math = .bf16,
    m: usize = 1024,
    n: usize = 1024,
    k: usize = 1024,
    iters: usize = 50,
    warmup: usize = 5,
    device: c_int = 0,
    list_devices: bool = false,
};

const CudaDriver = struct {
    lib: std.DynLib,
    cuInit: CuInit,
    cuDeviceGet: CuDeviceGet,
    cuDeviceGetCount: CuDeviceGetCount,
    cuDeviceGetName: CuDeviceGetName,
    cuDeviceComputeCapability: CuDeviceComputeCapability,
    cuCtxCreate: CuCtxCreate,
    cuCtxDestroy: CuCtxDestroy,
    cuCtxSynchronize: CuCtxSynchronize,
    cuModuleLoadData: CuModuleLoadData,
    cuModuleUnload: CuModuleUnload,
    cuModuleGetFunction: CuModuleGetFunction,
    cuMemAlloc: CuMemAlloc,
    cuMemFree: CuMemFree,
    cuMemcpyHtoD: CuMemcpyHtoD,
    cuMemcpyDtoH: CuMemcpyDtoH,
    cuLaunchKernel: CuLaunchKernel,
    cuGetErrorString: CuGetErrorString,

    fn open() !CudaDriver {
        var lib = try std.DynLib.open("libcuda.so.1");
        errdefer lib.close();

        return .{
            .lib = lib,
            .cuInit = lib.lookup(CuInit, "cuInit") orelse return error.SymbolNotFound,
            .cuDeviceGet = lib.lookup(CuDeviceGet, "cuDeviceGet") orelse return error.SymbolNotFound,
            .cuDeviceGetCount = lib.lookup(CuDeviceGetCount, "cuDeviceGetCount") orelse return error.SymbolNotFound,
            .cuDeviceGetName = lib.lookup(CuDeviceGetName, "cuDeviceGetName") orelse return error.SymbolNotFound,
            .cuDeviceComputeCapability = lib.lookup(CuDeviceComputeCapability, "cuDeviceComputeCapability") orelse return error.SymbolNotFound,
            .cuCtxCreate = lib.lookup(CuCtxCreate, "cuCtxCreate_v2") orelse return error.SymbolNotFound,
            .cuCtxDestroy = lib.lookup(CuCtxDestroy, "cuCtxDestroy_v2") orelse return error.SymbolNotFound,
            .cuCtxSynchronize = lib.lookup(CuCtxSynchronize, "cuCtxSynchronize") orelse return error.SymbolNotFound,
            .cuModuleLoadData = lib.lookup(CuModuleLoadData, "cuModuleLoadData") orelse return error.SymbolNotFound,
            .cuModuleUnload = lib.lookup(CuModuleUnload, "cuModuleUnload") orelse return error.SymbolNotFound,
            .cuModuleGetFunction = lib.lookup(CuModuleGetFunction, "cuModuleGetFunction") orelse return error.SymbolNotFound,
            .cuMemAlloc = lib.lookup(CuMemAlloc, "cuMemAlloc_v2") orelse return error.SymbolNotFound,
            .cuMemFree = lib.lookup(CuMemFree, "cuMemFree_v2") orelse return error.SymbolNotFound,
            .cuMemcpyHtoD = lib.lookup(CuMemcpyHtoD, "cuMemcpyHtoD_v2") orelse return error.SymbolNotFound,
            .cuMemcpyDtoH = lib.lookup(CuMemcpyDtoH, "cuMemcpyDtoH_v2") orelse return error.SymbolNotFound,
            .cuLaunchKernel = lib.lookup(CuLaunchKernel, "cuLaunchKernel") orelse return error.SymbolNotFound,
            .cuGetErrorString = lib.lookup(CuGetErrorString, "cuGetErrorString") orelse return error.SymbolNotFound,
        };
    }

    fn close(self: *CudaDriver) void {
        self.lib.close();
    }

    fn check(self: *CudaDriver, result: CuResult) !void {
        if (result == 0) return;

        var message_ptr: [*:0]const u8 = "unknown CUDA error";
        _ = self.cuGetErrorString(result, &message_ptr);
        log.err("CUDA driver error {d}: {s}", .{ result, std.mem.span(message_ptr) });
        return error.CudaFailure;
    }
};

const Cublas = struct {
    lib: std.DynLib,
    cublasCreate: CublasCreate,
    cublasDestroy: CublasDestroy,
    cublasGemmEx: CublasGemmEx,

    fn open() !Cublas {
        const candidates = [_][]const u8{
            "libcublas.so",
            "libcublas.so.13",
            "libcublas.so.12",
        };
        var last_err: anyerror = error.FileNotFound;
        for (candidates) |candidate| {
            var lib = std.DynLib.open(candidate) catch |err| {
                last_err = err;
                continue;
            };
            errdefer lib.close();
            return .{
                .lib = lib,
                .cublasCreate = lib.lookup(CublasCreate, "cublasCreate_v2") orelse return error.SymbolNotFound,
                .cublasDestroy = lib.lookup(CublasDestroy, "cublasDestroy_v2") orelse return error.SymbolNotFound,
                .cublasGemmEx = lib.lookup(CublasGemmEx, "cublasGemmEx") orelse return error.SymbolNotFound,
            };
        }
        return last_err;
    }

    fn close(self: *Cublas) void {
        self.lib.close();
    }

    fn check(_: *Cublas, result: CublasStatus) !void {
        if (result == 0) return;
        log.err("cuBLAS error {d}", .{result});
        return error.CublasFailure;
    }
};

const DeviceBuffers = struct {
    a: CuDevicePtr = 0,
    b_t: CuDevicePtr = 0,
    b_row: CuDevicePtr = 0,
    c_zig: CuDevicePtr = 0,
    c_cublas: CuDevicePtr = 0,

    fn deinit(self: *DeviceBuffers, cuda: *CudaDriver) void {
        if (self.a != 0) _ = cuda.cuMemFree(self.a);
        if (self.b_t != 0) _ = cuda.cuMemFree(self.b_t);
        if (self.b_row != 0) _ = cuda.cuMemFree(self.b_row);
        if (self.c_zig != 0) _ = cuda.cuMemFree(self.c_zig);
        if (self.c_cublas != 0) _ = cuda.cuMemFree(self.c_cublas);
    }
};

pub fn main(init: std.process.Init) !void {
    const allocator = std.heap.c_allocator;
    const args = try init.minimal.args.toSlice(allocator);
    defer allocator.free(args);

    const opts = try parseArgs(args);
    if (opts.m == 0 or opts.n == 0 or opts.k == 0 or opts.iters == 0) return error.InvalidArguments;
    if (opts.impl.runsZig()) validateZigKernelDimensions(opts);

    var cuda = try CudaDriver.open();
    defer cuda.close();
    try cuda.check(cuda.cuInit(0));

    try listDevicesIfRequested(&cuda, opts);
    if (opts.list_devices) return;

    var device: CuDevice = undefined;
    try cuda.check(cuda.cuDeviceGet(&device, opts.device));

    var context: CuContext = null;
    try cuda.check(cuda.cuCtxCreate(&context, 0, device));
    defer _ = cuda.cuCtxDestroy(context);

    const ptx_text = if (opts.impl.runsZig()) blk: {
        const ptx_path = selectedPtxPath(opts) orelse return error.MissingPtxPath;
        break :blk try readPtxText(init.io, allocator, args[0], ptx_path);
    } else try allocator.dupe(u8, "");
    defer allocator.free(ptx_text);

    const ptx_z = if (opts.impl.runsZig()) try sanitizePtxForCudaJit(allocator, ptx_text) else try allocator.dupeZ(u8, "");
    defer allocator.free(ptx_z);

    var module: CuModule = null;
    var function: CuFunction = null;
    if (opts.impl.runsZig()) {
        try cuda.check(cuda.cuModuleLoadData(&module, ptx_z.ptr));
        try cuda.check(cuda.cuModuleGetFunction(&function, module, "matmul"));
    }
    defer {
        if (module != null) _ = cuda.cuModuleUnload(module);
    }

    const a = try makeRowMajorMatrix(allocator, opts.m, opts.k, 3, opts.math);
    defer allocator.free(a);
    const b_row = try makeRowMajorMatrix(allocator, opts.k, opts.n, 11, opts.math);
    defer allocator.free(b_row);
    const b_t = try transposeForKernel(allocator, b_row, opts.k, opts.n);
    defer allocator.free(b_t);

    const expected = try buildExpectedTable(opts, allocator);
    defer allocator.free(expected);

    var buffers: DeviceBuffers = .{};
    defer buffers.deinit(&cuda);
    try allocAndUpload(&cuda, &buffers, a, b_t, b_row, opts);

    if (opts.impl.runsZig()) {
        const avg_ns = try runZigKernel(&cuda, function, &buffers, opts);
        try downloadAndValidate(&cuda, allocator, "zig", buffers.c_zig, expected, opts);
        printResult("zig", opts, avg_ns);
    }

    if (opts.impl.runsCublas()) {
        var cublas = try Cublas.open();
        defer cublas.close();
        const avg_ns = try runCublas(&cuda, &cublas, &buffers, opts);
        try downloadAndValidate(&cuda, allocator, "cublas", buffers.c_cublas, expected, opts);
        printResult("cublas", opts, avg_ns);
    }
}

fn parseArgs(args: []const []const u8) !Options {
    var opts: Options = .{};
    for (args[1..]) |arg| {
        if (std.mem.eql(u8, arg, "--list-devices")) {
            opts.list_devices = true;
        } else if (std.mem.startsWith(u8, arg, "--ptx=")) {
            opts.ptx_path = arg["--ptx=".len..];
        } else if (std.mem.startsWith(u8, arg, "--ptx-bf16=")) {
            opts.ptx_bf16_path = arg["--ptx-bf16=".len..];
        } else if (std.mem.startsWith(u8, arg, "--ptx-f16=")) {
            opts.ptx_f16_path = arg["--ptx-f16=".len..];
        } else if (std.mem.startsWith(u8, arg, "--impl=")) {
            const value = arg["--impl=".len..];
            if (std.mem.eql(u8, value, "zig")) opts.impl = .zig else if (std.mem.eql(u8, value, "cublas")) opts.impl = .cublas else if (std.mem.eql(u8, value, "both")) opts.impl = .both else return error.InvalidImpl;
        } else if (std.mem.startsWith(u8, arg, "--math=")) {
            const value = arg["--math=".len..];
            if (std.mem.eql(u8, value, "bf16")) opts.math = .bf16 else if (std.mem.eql(u8, value, "f16")) opts.math = .f16 else return error.InvalidMath;
        } else if (std.mem.startsWith(u8, arg, "--m=")) {
            opts.m = try std.fmt.parseInt(usize, arg["--m=".len..], 10);
        } else if (std.mem.startsWith(u8, arg, "--n=")) {
            opts.n = try std.fmt.parseInt(usize, arg["--n=".len..], 10);
        } else if (std.mem.startsWith(u8, arg, "--k=")) {
            opts.k = try std.fmt.parseInt(usize, arg["--k=".len..], 10);
        } else if (std.mem.startsWith(u8, arg, "--iters=")) {
            opts.iters = try std.fmt.parseInt(usize, arg["--iters=".len..], 10);
        } else if (std.mem.startsWith(u8, arg, "--warmup=")) {
            opts.warmup = try std.fmt.parseInt(usize, arg["--warmup=".len..], 10);
        } else if (std.mem.startsWith(u8, arg, "--device=")) {
            opts.device = try std.fmt.parseInt(c_int, arg["--device=".len..], 10);
        } else {
            log.err("unknown argument: {s}", .{arg});
            return error.InvalidArgument;
        }
    }
    return opts;
}

fn selectedPtxPath(opts: Options) ?[]const u8 {
    if (opts.ptx_path) |path| return path;
    return switch (opts.math) {
        .bf16 => opts.ptx_bf16_path,
        .f16 => opts.ptx_f16_path,
    };
}

fn validateZigKernelDimensions(opts: Options) void {
    if (opts.m % 64 == 0 and opts.n % 64 == 0 and opts.k % 16 == 0) return;
    log.err("zig kernel requires m % 64 == 0, n % 64 == 0, and k % 16 == 0; use --impl=cublas for arbitrary {s} GEMM", .{opts.math.name()});
    std.process.exit(2);
}

fn listDevicesIfRequested(cuda: *CudaDriver, opts: Options) !void {
    var count: c_int = 0;
    try cuda.check(cuda.cuDeviceGetCount(&count));
    if (opts.list_devices) {
        var i: c_int = 0;
        while (i < count) : (i += 1) {
            var device: CuDevice = undefined;
            try cuda.check(cuda.cuDeviceGet(&device, i));
            var name_buf: [256]u8 = undefined;
            try cuda.check(cuda.cuDeviceGetName(&name_buf, name_buf.len, device));
            var cc_major: c_int = 0;
            var cc_minor: c_int = 0;
            try cuda.check(cuda.cuDeviceComputeCapability(&cc_major, &cc_minor, device));
            std.debug.print("device[{d}]: {s} compute={d}.{d}\n", .{
                i,
                std.mem.sliceTo(name_buf[0..], 0),
                cc_major,
                cc_minor,
            });
        }
    }
    if (!opts.list_devices and (opts.device < 0 or opts.device >= count)) return error.InvalidDevice;
}

fn readPtxText(io: std.Io, allocator: std.mem.Allocator, exe_path: []const u8, ptx_path: []const u8) ![]u8 {
    return std.Io.Dir.cwd().readFileAlloc(io, ptx_path, allocator, .limited(64 * 1024 * 1024)) catch |err| switch (err) {
        error.FileNotFound => {
            const runfiles_dir = try std.mem.concat(allocator, u8, &.{ exe_path, ".runfiles" });
            defer allocator.free(runfiles_dir);
            const runfiles_path = try std.fs.path.join(allocator, &.{ runfiles_dir, "_main", ptx_path });
            defer allocator.free(runfiles_path);
            return try std.Io.Dir.cwd().readFileAlloc(io, runfiles_path, allocator, .limited(64 * 1024 * 1024));
        },
        else => |e| return e,
    };
}

fn sanitizePtxForCudaJit(allocator: std.mem.Allocator, ptx: []const u8) ![:0]u8 {
    var owned = try allocator.dupe(u8, ptx);
    errdefer allocator.free(owned);

    const replacements = [_][2][]const u8{
        .{ ".u2 builtin_$_output_mode", ".b8 builtin_$_output_mode" },
        .{ ".u5 builtin_$_abi", ".b8 builtin_$_abi" },
        .{ ".u4 builtin_$_object_format", ".b8 builtin_$_object_format" },
    };

    for (replacements) |pair| {
        const next = try std.mem.replaceOwned(u8, allocator, owned, pair[0], pair[1]);
        if (next.ptr != owned.ptr) allocator.free(owned);
        owned = next;
    }

    defer allocator.free(owned);
    return try allocator.dupeZ(u8, owned);
}

fn allocAndUpload(cuda: *CudaDriver, buffers: *DeviceBuffers, a: []const u16, b_t: []const u16, b_row: []const u16, opts: Options) !void {
    const c_bytes = opts.m * opts.n * opts.math.outputElementSize();
    try cuda.check(cuda.cuMemAlloc(&buffers.a, a.len * @sizeOf(u16)));
    try cuda.check(cuda.cuMemAlloc(&buffers.b_t, b_t.len * @sizeOf(u16)));
    try cuda.check(cuda.cuMemAlloc(&buffers.b_row, b_row.len * @sizeOf(u16)));
    try cuda.check(cuda.cuMemAlloc(&buffers.c_zig, c_bytes));
    try cuda.check(cuda.cuMemAlloc(&buffers.c_cublas, c_bytes));
    try cuda.check(cuda.cuMemcpyHtoD(buffers.a, a.ptr, a.len * @sizeOf(u16)));
    try cuda.check(cuda.cuMemcpyHtoD(buffers.b_t, b_t.ptr, b_t.len * @sizeOf(u16)));
    try cuda.check(cuda.cuMemcpyHtoD(buffers.b_row, b_row.ptr, b_row.len * @sizeOf(u16)));
}

fn runZigKernel(cuda: *CudaDriver, function: CuFunction, buffers: *DeviceBuffers, opts: Options) !f64 {
    var param_a = buffers.a;
    var param_b = buffers.b_t;
    var param_c = buffers.c_zig;
    var param_m = opts.m;
    var param_n = opts.n;
    var param_k = opts.k;
    var params = [_]?*anyopaque{
        @ptrCast(&param_a),
        @ptrCast(&param_b),
        @ptrCast(&param_c),
        @ptrCast(&param_m),
        @ptrCast(&param_n),
        @ptrCast(&param_k),
    };

    const grid_x: c_uint = @intCast(opts.m / 64);
    const grid_y: c_uint = @intCast(opts.n / 64);
    const shared_bytes: c_uint = @intCast(2 * (64 * 32) * @sizeOf(u16));

    for (0..opts.warmup) |_| try launch(cuda, function, grid_x, grid_y, shared_bytes, &params);
    try cuda.check(cuda.cuCtxSynchronize());

    const start = try nanoTimestamp();
    for (0..opts.iters) |_| try launch(cuda, function, grid_x, grid_y, shared_bytes, &params);
    try cuda.check(cuda.cuCtxSynchronize());
    const elapsed = try nanoTimestamp() - start;
    return @as(f64, @floatFromInt(elapsed)) / @as(f64, @floatFromInt(opts.iters));
}

fn launch(cuda: *CudaDriver, function: CuFunction, grid_x: c_uint, grid_y: c_uint, shared_bytes: c_uint, params: *[6]?*anyopaque) !void {
    try cuda.check(cuda.cuLaunchKernel(
        function,
        grid_x,
        grid_y,
        1,
        256,
        1,
        1,
        shared_bytes,
        null,
        params,
        null,
    ));
}

fn runCublas(cuda: *CudaDriver, cublas: *Cublas, buffers: *DeviceBuffers, opts: Options) !f64 {
    var handle: CublasHandle = null;
    try cublas.check(cublas.cublasCreate(&handle));
    defer _ = cublas.cublasDestroy(handle);

    const alpha: f32 = 1;
    const beta: f32 = 0;
    const b_ptr: *const anyopaque = @ptrFromInt(buffers.b_row);
    const a_ptr: *const anyopaque = @ptrFromInt(buffers.a);
    const c_ptr: *anyopaque = @ptrFromInt(buffers.c_cublas);

    for (0..opts.warmup) |_| try cublasGemm(cublas, handle, b_ptr, a_ptr, c_ptr, &alpha, &beta, opts);
    try cuda.check(cuda.cuCtxSynchronize());

    const start = try nanoTimestamp();
    for (0..opts.iters) |_| try cublasGemm(cublas, handle, b_ptr, a_ptr, c_ptr, &alpha, &beta, opts);
    try cuda.check(cuda.cuCtxSynchronize());
    const elapsed = try nanoTimestamp() - start;
    return @as(f64, @floatFromInt(elapsed)) / @as(f64, @floatFromInt(opts.iters));
}

fn cublasGemm(cublas: *Cublas, handle: CublasHandle, b_ptr: *const anyopaque, a_ptr: *const anyopaque, c_ptr: *anyopaque, alpha: *const f32, beta: *const f32, opts: Options) !void {
    try cublas.check(cublas.cublasGemmEx(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        @intCast(opts.n),
        @intCast(opts.m),
        @intCast(opts.k),
        @ptrCast(alpha),
        b_ptr,
        opts.math.inputCudaType(),
        @intCast(opts.n),
        a_ptr,
        opts.math.inputCudaType(),
        @intCast(opts.k),
        @ptrCast(beta),
        c_ptr,
        opts.math.outputCudaType(),
        @intCast(opts.n),
        opts.math.computeType(),
        CUBLAS_GEMM_DEFAULT_TENSOR_OP,
    ));
}

fn makeRowMajorMatrix(allocator: std.mem.Allocator, rows: usize, cols: usize, seed: u32, math: Math) ![]u16 {
    const data = try allocator.alloc(u16, rows * cols);
    for (0..rows) |row| {
        for (0..cols) |col| {
            data[row * cols + col] = encodeInputBits(inputValue(row, col, seed), math);
        }
    }
    return data;
}

fn transposeForKernel(allocator: std.mem.Allocator, b_row: []const u16, rows: usize, cols: usize) ![]u16 {
    const out = try allocator.alloc(u16, rows * cols);
    for (0..rows) |row| {
        for (0..cols) |col| {
            out[col * rows + row] = b_row[row * cols + col];
        }
    }
    return out;
}

fn buildExpectedTable(opts: Options, allocator: std.mem.Allocator) ![]f32 {
    const out = try allocator.alloc(f32, 23 * 23);
    for (0..23) |row_mod| {
        for (0..23) |col_mod| {
            var sum: f32 = 0;
            for (0..opts.k) |kk| {
                const a = inputBitsToF32(encodeInputBits(inputValue(row_mod, kk, 3), opts.math), opts.math);
                const b = inputBitsToF32(encodeInputBits(inputValue(kk, col_mod, 11), opts.math), opts.math);
                sum += a * b;
            }
            out[row_mod * 23 + col_mod] = switch (opts.math) {
                .bf16 => bf16BitsToF32(f32ToBf16Bits(sum)),
                .f16 => sum,
            };
        }
    }
    return out;
}

fn inputValue(row: usize, col: usize, seed: u32) f32 {
    const raw: f32 = @floatFromInt((row * 17 + col * 13 + seed) % 23);
    return (raw - 11.0) / 8.0;
}

fn downloadAndValidate(
    cuda: *CudaDriver,
    allocator: std.mem.Allocator,
    label: []const u8,
    device_ptr: CuDevicePtr,
    expected: []const f32,
    opts: Options,
) !void {
    switch (opts.math) {
        .bf16 => {
            const c_out = try allocator.alloc(u16, opts.m * opts.n);
            defer allocator.free(c_out);
            try cuda.check(cuda.cuMemcpyDtoH(c_out.ptr, device_ptr, c_out.len * @sizeOf(u16)));
            try validateBf16Output(label, c_out, expected, opts);
        },
        .f16 => {
            const c_out = try allocator.alloc(f32, opts.m * opts.n);
            defer allocator.free(c_out);
            try cuda.check(cuda.cuMemcpyDtoH(c_out.ptr, device_ptr, c_out.len * @sizeOf(f32)));
            try validateF32Output(label, c_out, expected, opts);
        },
    }
}

fn validateBf16Output(label: []const u8, got: []const u16, expected_table: []const f32, opts: Options) !void {
    for (0..opts.m) |row| {
        for (0..opts.n) |col| {
            const actual = bf16BitsToF32(got[row * opts.n + col]);
            const expected = expected_table[(row % 23) * 23 + (col % 23)];
            const tolerance = @max(@as(f32, 0.25), @abs(expected) * 0.025);
            if (@abs(actual - expected) > tolerance) {
                log.err("{s} mismatch at ({d}, {d}): got {d:.6}, expected {d:.6}, tolerance {d:.6}", .{ label, row, col, actual, expected, tolerance });
                return error.ValidationFailed;
            }
        }
    }
}

fn validateF32Output(label: []const u8, got: []const f32, expected_table: []const f32, opts: Options) !void {
    for (0..opts.m) |row| {
        for (0..opts.n) |col| {
            const actual = got[row * opts.n + col];
            const expected = expected_table[(row % 23) * 23 + (col % 23)];
            const tolerance = @max(@as(f32, 1.0e-2), @abs(expected) * 5.0e-3);
            if (@abs(actual - expected) > tolerance) {
                log.err("{s} mismatch at ({d}, {d}): got {d:.6}, expected {d:.6}, tolerance {d:.6}", .{ label, row, col, actual, expected, tolerance });
                return error.ValidationFailed;
            }
        }
    }
}

fn printResult(label: []const u8, opts: Options, avg_ns: f64) void {
    const ops = 2.0 * @as(f64, @floatFromInt(opts.m)) * @as(f64, @floatFromInt(opts.n)) * @as(f64, @floatFromInt(opts.k));
    const tflops = ops / avg_ns / 1.0e3;
    std.debug.print("{s} validation passed: math={s} m={d} n={d} k={d} avg_ns={d:.2} TFLOP/s={d:.4}\n", .{ label, opts.math.name(), opts.m, opts.n, opts.k, avg_ns, tflops });
}

fn encodeInputBits(value: f32, math: Math) u16 {
    return switch (math) {
        .bf16 => f32ToBf16Bits(value),
        .f16 => f32ToF16Bits(value),
    };
}

fn inputBitsToF32(bits: u16, math: Math) f32 {
    return switch (math) {
        .bf16 => bf16BitsToF32(bits),
        .f16 => f16BitsToF32(bits),
    };
}

fn f32ToBf16Bits(value: f32) u16 {
    const bits: u32 = @bitCast(value);
    const rounding_bias: u32 = 0x7fff + ((bits >> 16) & 1);
    return @truncate((bits + rounding_bias) >> 16);
}

fn f32ToF16Bits(value: f32) u16 {
    const half: f16 = @floatCast(value);
    return @bitCast(half);
}

fn bf16BitsToF32(bits: u16) f32 {
    return @bitCast(@as(u32, bits) << 16);
}

fn f16BitsToF32(bits: u16) f32 {
    const half: f16 = @bitCast(bits);
    return @floatCast(half);
}

fn nanoTimestamp() !i128 {
    var ts: Timespec = undefined;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) return error.ClockFailure;
    return @as(i128, ts.tv_sec) * 1_000_000_000 + ts.tv_nsec;
}
