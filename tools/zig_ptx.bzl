load("@rules_cc//cc:action_names.bzl", "ACTION_NAMES")
load("@rules_cc//cc:find_cc_toolchain.bzl", "find_cc_toolchain", "use_cc_toolchain")
load("@rules_cc//cc/common:cc_common.bzl", "cc_common")

def _zig_ptx_impl(ctx):
    zig_toolchain = ctx.toolchains["@rules_zig//zig:toolchain_type"].zigtoolchaininfo
    if zig_toolchain.zig_exe.file == None or zig_toolchain.zig_lib.file == None:
        fail("zig_ptx requires the pinned file-based rules_zig toolchain")

    cc_toolchain = find_cc_toolchain(ctx)
    feature_configuration = cc_common.configure_features(
        ctx = ctx,
        cc_toolchain = cc_toolchain,
    )
    clang = cc_common.get_tool_for_action(
        feature_configuration = feature_configuration,
        action_name = ACTION_NAMES.c_compile,
    )

    src = ctx.file.src
    out = ctx.actions.declare_file(ctx.attr.out)
    ir = ctx.actions.declare_file(ctx.label.name + ".ll")
    fixed_ir = ctx.actions.declare_file(ctx.label.name + ".fixed.ll")

    zig_args = ctx.actions.args()
    zig_args.add_all([
        "build-lib",
        "--zig-lib-dir",
        zig_toolchain.zig_lib.file,
        "--cache-dir",
        zig_toolchain.zig_cache,
        "--global-cache-dir",
        zig_toolchain.zig_cache,
        "-O",
        "ReleaseFast",
        "-fllvm",
        "-target",
        "nvptx64-cuda-none",
        "-fno-compiler-rt",
        "-mcpu=" + ctx.attr.mcpu,
        "-fno-stack-protector",
        "-fstrip",
        "-femit-llvm-ir=" + ir.path,
        "-fno-emit-bin",
        src,
    ])

    inputs = depset(
        direct = [
            src,
            zig_toolchain.validation,
            zig_toolchain.zig_lib.file,
        ],
    )

    ctx.actions.run(
        inputs = inputs,
        outputs = [ir],
        executable = zig_toolchain.zig_exe.file,
        arguments = [zig_args],
        env = {
            "ZIG_GLOBAL_CACHE_DIR": zig_toolchain.zig_cache,
            "ZIG_LOCAL_CACHE_DIR": zig_toolchain.zig_cache,
        },
        mnemonic = "ZigNvptxIr",
        progress_message = "Emitting NVPTX LLVM IR %{output}",
        toolchain = "@rules_zig//zig:toolchain_type",
    )

    ctx.actions.run(
        inputs = [ir],
        outputs = [fixed_ir],
        executable = ctx.executable._normalizer,
        arguments = [ir.path, fixed_ir.path],
        mnemonic = "ZigNormalizeNvptxIr",
        progress_message = "Normalizing NVPTX LLVM IR %{output}",
    )

    clang_args = ctx.actions.args()
    clang_args.add_all([
        "-target",
        "nvptx64-nvidia-cuda",
        "-march=" + ctx.attr.mcpu,
        "-S",
        fixed_ir,
        "-o",
        out,
    ])
    ctx.actions.run(
        inputs = [fixed_ir],
        outputs = [out],
        tools = cc_toolchain.all_files,
        executable = clang,
        arguments = [clang_args],
        mnemonic = "ClangNvptxPtx",
        progress_message = "Emitting PTX %{output}",
        toolchain = "@rules_cc//cc:toolchain_type",
    )

    return [DefaultInfo(files = depset([out]))]

zig_ptx = rule(
    implementation = _zig_ptx_impl,
    attrs = {
        "src": attr.label(
            allow_single_file = [".zig"],
            mandatory = True,
        ),
        "out": attr.string(
            mandatory = True,
        ),
        "mcpu": attr.string(
            default = "sm_120",
        ),
        "_normalizer": attr.label(
            default = "//tools:normalize_nvptx_ir",
            executable = True,
            cfg = "exec",
        ),
    },
    fragments = ["cpp"],
    toolchains = use_cc_toolchain() + ["@rules_zig//zig:toolchain_type"],
)
