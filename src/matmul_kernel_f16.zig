const std = @import("std");
pub const panic = std.debug.no_panic;

const F16x2 = @Vector(2, f16);
const U32x4 = @Vector(4, u32);

pub fn ExternVector(comptime len: usize, comptime child_type: type) type {
    var field_names: [len][]const u8 = undefined;
    for (&field_names, 0..) |*field_name, i| {
        field_name.* = std.fmt.comptimePrint("_{d}", .{i});
    }
    return @Struct(
        .@"extern",
        null,
        &field_names,
        &(@splat(child_type)),
        &(@splat(.{})),
    );
}

pub fn LLVMIntrinsic(comptime T: type) type {
    const info = @typeInfo(T);
    var param_types: [info.@"fn".params.len]type = undefined;
    for (info.@"fn".params, &param_types) |param, *param_type| {
        param_type.* = param.type.?;
    }
    return *const @Fn(
        &param_types,
        &(@splat(.{})),
        info.@"fn".return_type.?,
        .{
            .@"callconv" = .c,
            .varargs = false,
        },
    );
}

pub fn llvm(comptime name: []const u8, comptime T: type) LLVMIntrinsic(T) {
    return @extern(LLVMIntrinsic(T), .{ .name = "llvm." ++ name });
}

pub fn nvvm(comptime name: []const u8, comptime T: type) LLVMIntrinsic(T) {
    return llvm("nvvm." ++ name, T);
}

const Axis = enum { x, y, z };

inline fn sreg(comptime reg: []const u8) u32 {
    return (comptime nvvm("read.ptx.sreg." ++ reg, fn () u32))();
}

fn ctaId(comptime axis: Axis) u32 {
    return sreg("ctaid." ++ @tagName(axis));
}

fn threadId(comptime axis: Axis) u32 {
    return sreg("tid." ++ @tagName(axis));
}

fn warpSize() u32 {
    return sreg("warpsize");
}

const syncThreads = nvvm("barrier0", fn () void);

const SharedMemory = struct {
    extern var base: [0]u8 align(16) addrspace(.shared);

    inline fn total() u32 {
        return asm ("mov.u32 %[r], %total_smem_size;"
            : [r] "=r" (-> u32),
        );
    }

    inline fn as(comptime T: type) []align(16) addrspace(.shared) T {
        return @ptrCast((&base)[0..total()]);
    }
};

fn ldMatrixB16(comptime regs: usize, comptime trans: bool, src: [*]addrspace(.shared) const u16) @Vector(regs, u32) {
    const f = comptime nvvm(
        std.fmt.comptimePrint(
            "ldmatrix.sync.aligned.m8n8.x{d}{s}.b16",
            .{ regs, if (trans) ".trans" else "" },
        ),
        fn ([*]addrspace(.shared) const u16) ExternVector(regs, u32),
    );
    return @bitCast(f(src));
}

fn toF16x2(bits: u32) F16x2 {
    return @bitCast(bits);
}

fn mmaM16N8K16(a: U32x4, b: @Vector(2, u32), c: @Vector(4, f32)) @Vector(4, f32) {
    const f = comptime nvvm(
        "mma.m16n8k16.row.col.f32.f32",
        fn (
            a0: F16x2,
            a1: F16x2,
            a2: F16x2,
            a3: F16x2,
            b0: F16x2,
            b1: F16x2,
            c0: f32,
            c1: f32,
            c2: f32,
            c3: f32,
        ) ExternVector(4, f32),
    );
    return @bitCast(f(
        toF16x2(a[0]),
        toF16x2(a[1]),
        toF16x2(a[2]),
        toF16x2(a[3]),
        toF16x2(b[0]),
        toF16x2(b[1]),
        c[0],
        c[1],
        c[2],
        c[3],
    ));
}

fn doMatmul(
    a: []const u16,
    b_t: []const u16,
    c: []f32,
    m: usize,
    n: usize,
    k: usize,
    comptime block_m: usize,
    comptime block_n: usize,
    comptime block_k: usize,
) void {
    _ = m;

    const m_block = ctaId(.x) * block_m;
    const n_block = ctaId(.y) * block_n;
    const warp_id = threadId(.x) / warpSize();
    const lane_id = threadId(.x) % warpSize();
    var shmem = SharedMemory.as(@Vector(8, u16));

    const a_shared = shmem[0 .. block_m * 2];
    const b_shared = shmem[block_m * 2 ..][0 .. block_n * 2];

    const group_id = lane_id >> 2;
    const thread_in_group = lane_id & 3;
    const warp_tile_m = 32 * (warp_id / 4);
    const warp_tile_n = 16 * (warp_id % 4);

    var c_frag = std.mem.zeroes([2][2]@Vector(4, f32));

    var k_block: usize = 0;
    while (k_block < k) : (k_block += block_k) {
        const tile_tid = (warp_id & 3) * 32 + lane_id;
        const row = tile_tid / 2;
        const col_vec = tile_tid % 2;
        const a_src: *align(16) const @Vector(8, u16) = @ptrCast(@alignCast(&a[(m_block + row) * k + k_block + col_vec * 8]));
        const b_src: *align(16) const @Vector(8, u16) = @ptrCast(@alignCast(&b_t[(n_block + row) * k + k_block + col_vec * 8]));
        switch (warp_id >> 2) {
            0 => a_shared[tile_tid] = a_src.*,
            1 => b_shared[tile_tid] = b_src.*,
            else => unreachable,
        }
        syncThreads();

        inline for (0..2) |m_i| {
            const frag_m = warp_tile_m + m_i * 16;
            const a_matrix = lane_id / 8;
            const a_row = frag_m + (lane_id % 8) + (a_matrix & 1) * 8;
            const a_col = a_matrix / 2;
            const a_frag = ldMatrixB16(4, false, @ptrCast(@alignCast(&a_shared[a_row * 2 + a_col])));

            inline for (0..2) |n_i| {
                const frag_n = warp_tile_n + n_i * 8;
                const b_lane = lane_id & 15;
                const b_matrix = b_lane / 8;
                const b_row = frag_n + (b_lane % 8);
                const b_frag = ldMatrixB16(2, false, @ptrCast(@alignCast(&b_shared[b_row * 2 + b_matrix])));
                c_frag[m_i][n_i] = mmaM16N8K16(a_frag, b_frag, c_frag[m_i][n_i]);
            }
        }
        syncThreads();
    }

    inline for (0..2) |m_i| {
        const row0 = m_block + warp_tile_m + m_i * 16 + group_id;
        const row1 = row0 + 8;
        inline for (0..2) |n_i| {
            const col = n_block + warp_tile_n + n_i * 8 + 2 * thread_in_group;
            c[row0 * n + col + 0] = c_frag[m_i][n_i][0];
            c[row0 * n + col + 1] = c_frag[m_i][n_i][1];
            c[row1 * n + col + 0] = c_frag[m_i][n_i][2];
            c[row1 * n + col + 1] = c_frag[m_i][n_i][3];
        }
    }
}

export fn matmul(
    a: [*]const u16,
    b_t: [*]const u16,
    c: [*]f32,
    m: usize,
    n: usize,
    k: usize,
) callconv(.kernel) void {
    doMatmul(a[0 .. m * k], b_t[0 .. n * k], c[0 .. m * n], m, n, k, 64, 64, 16);
}
