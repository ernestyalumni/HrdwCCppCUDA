# UnitTests is its own "package" in Bazel, because of this BUILD file.

load("@rules_cc//cc:defs.bzl", "cc_test")

cc_test(
  name = "ConstructiconGoogleUnitTests",
  size = "small",
  srcs = [
      "Performance/MatrixMultiply/MatrixMultiply_tests.cc"
      ],
  deps = [
      "@com_google_googletest//:gtest_main",
      "//Source/Performance:Performance"],
  copts = ["-ISource/"]
)