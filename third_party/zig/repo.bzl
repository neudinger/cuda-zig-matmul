load("@bazel_tools//tools/build_defs/repo:git.bzl", "new_git_repository")

def repo():
    new_git_repository(
        name = "zig",
        remote = "https://codeberg.org/ziglang/zig.git",
        commit = "24fdd5b7a4c1c8b5deb5b56756b9dbc8e08c86a8",
        build_file = "//third_party/zig:zig.bazel",
        patch_args = ["-p1"],
        patches = [
            "//third_party/zig/patches:0.16.0/0005-ZML-zig-enable-configurable-LLVM-Clang-and-LLVM-AR.patch",
            "//third_party/zig/patches:0.16.0/0007-llvm-do-not-use-LLVM-aliases-for-nvptx-exports.patch",
        ],
    )
