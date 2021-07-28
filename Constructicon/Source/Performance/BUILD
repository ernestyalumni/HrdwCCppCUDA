# Sharing Variables
# cf. https://docs.bazel.build/versions/main/skylark/tutorial-sharing-variables.html
# Share values by introducing variables.
LDFLAGS = "-Wall -lrt -flto -fuse-ld=gold"
COMPILER_WARNING_FLAGS = "-Wall -Wextra -pedantic -Werror"

cc_binary(
    name="SizesCExecutable",

    # cf. https://docs.bazel.build/versions/main/user-manual.html#flag--conlyopt
    copts=[LDFLAGS],

    srcs=["CPrimer/sizes.c"],
)

cc_binary(
    name="pointer.o",

    # cf. https://docs.bazel.build/versions/main/user-manual.html#flag--conlyopt
    copts=["-c"],

    srcs=["CPrimer/pointer.c"],
)

cc_binary(
    name="PointerCExecutable",

    # cf. https://docs.bazel.build/versions/main/user-manual.html#flag--conlyopt
    copts=["-Wall"],

    srcs=[
        "CPrimer/pointer.c",
        # Complains below not in directory with BUILD file,
        # TODO: Consider
        # https://stackoverflow.com/questions/60568146/bazel-how-to-build-a-simple-application-that-links-a-pre-built-external-library
        #"//bazel-bin/Source/Performance/pointer.o"
        ],
)

cc_binary(
    name="MatMultCustomMain.exe",

    linkopts=[LDFLAGS],

    srcs=[
            "MatrixMultiply/custom_main.c",
            "MatrixMultiply/matrix_multiply.c"],

    includes=["MatrixMultiply/"]
)

cc_library(
    name="Performance",
    srcs = [
        "MatrixMultiply/MatrixMultiply.cc"
        ],
    hdrs = [
        "MatrixMultiply/MatrixMultiply.h"],
    deps = [
        "//Source/Utilities:Utilities"
        ],
    copts = [
        "-ISource/",
        COMPILER_WARNING_FLAGS],
    visibility = [
        # This target is visible to targets and subtargets in package
        # //Source/UnitTests .
        # __subpackages__ grants access to targets defined in //Source/UnitTests
        # or any of its direct or indirect subpackages.
        "//Source/UnitTests:__subpackages__"]
)

cc_binary(
    name="MatMultiplyTestBed.exe",

    deps = ["//Source/Performance:Performance"],
    copts = [
        COMPILER_WARNING_FLAGS,
        "-O3",
        "-ISource/Performance"
        ],
    # cf. https://github.com/bazelbuild/bazel/issues/8846
    # includes = ["//Source/Performance"],
    srcs=[
            "MatrixMultiply/TestBed.cc"],

)
