cc_library(
    name="Utilities",
    srcs = [
        "DebuggingMacros.h",
        "Time/SuperTimeSpecification.h"
        ],
    hdrs = [
        "DebuggingMacros.h",
        "Time/SuperTimeSpecification.h"
        ],
    copts = [
        "-ISource/",
        "-Wall -Wextra -pedantic -Werror"],
    visibility = [
        # This target is visible to targets and subtargets in package
        # //Source/UnitTests .
        # __subpackages__ grants access to targets defined in //Source/UnitTests
        # or any of its direct or indirect subpackages.
        "//Source/Performance:__subpackages__",
        "//Source/UnitTests:__subpackages__"
        ]
)