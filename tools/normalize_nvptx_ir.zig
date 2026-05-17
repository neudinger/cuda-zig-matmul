const std = @import("std");

const alias_prefix = "@matmul = alias ";
const kernel_prefix = "define private ptx_kernel void @";
const kernel_suffix = ".matmul";

pub fn main(init: std.process.Init) !void {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    const args = try init.minimal.args.toSlice(allocator);
    defer allocator.free(args);

    if (args.len != 3) return error.InvalidArguments;

    const input = try std.Io.Dir.cwd().readFileAlloc(init.io, args[1], allocator, .limited(64 * 1024 * 1024));
    defer allocator.free(input);

    var output = try std.Io.Dir.cwd().createFile(init.io, args[2], .{});
    defer output.close(init.io);

    var start: usize = 0;
    while (start < input.len) {
        const newline = std.mem.indexOfScalarPos(u8, input, start, '\n');
        const end = newline orelse input.len;
        try writeNormalizedLine(init.io, output, input[start..end]);
        if (newline != null) try output.writeStreamingAll(init.io, "\n");
        start = end + 1;
    }
}

fn writeNormalizedLine(io: std.Io, output: std.Io.File, line: []const u8) !void {
    if (std.mem.startsWith(u8, line, alias_prefix)) return;

    if (std.mem.startsWith(u8, line, kernel_prefix)) {
        const rest = line[kernel_prefix.len..];
        if (std.mem.indexOf(u8, rest, kernel_suffix)) |suffix_index| {
            const after_name = rest[suffix_index + kernel_suffix.len ..];
            try output.writeStreamingAll(io, "define ptx_kernel void @matmul");
            try output.writeStreamingAll(io, after_name);
            return;
        }
    }

    try output.writeStreamingAll(io, line);
}
