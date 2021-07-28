#!/bin/bash

# cf. https://www.xmodulo.com/command-line-arguments-bash-script.html
# Read in command lines by their positions

# Checks if first argument exists, and proceeds with checking for option.
if [ ! -z $1 ]; then
    echo "First argument is set and is: " $1
fi

# C Primer.

## Sizes demonstration.

printf "\n"

echo "Building sizes.c executable."

printf "\n"

# --strategy=CppCompile=standalone is to resolve this error:
# ccache: error: Failed to create temporary file for /home/topolo/.ccache/tmp/sizes.stdout: Read-only file system
# cf. https://stackoverflow.com/questions/50858954/bazel-building-c-sample-with-ccache-fails
# https://stackoverflow.com/questions/49178997/bazel-failed-to-create-temporary-file

CC=clang

echo ${CC}

bazel build //Source/Performance:SizesCExecutable --strategy=CppCompile=standalone

./bazel-bin/Source/Performance/SizesCExecutable


## Pointer object

bazel build --conlyopt=-c //Source/Performance:pointer.o --strategy=CppCompile=standalone


## Pointer executable.

# -c for compilation mode
# cf. https://docs.bazel.build/versions/main/command-line-reference.html#flag--compilation_mode
# https://docs.bazel.build/versions/main/user-manual.html#semantics-options

# opt is -O2 -DNDEBUG
bazel build -c opt //Source/Performance:PointerCExecutable --strategy=CppCompile=standalone

# Or try

./bazel-bin/Source/Performance/PointerCExecutable

./bazel-bin/Source/Performance/PointerCExecutable 1 2 3 a b cc ddd grind "OnIt"


################################################################################

# Matrix Multiplication.

## Custom Main file for demonstrating features.

bazel build -c opt //Source/Performance:MatMultCustomMain.exe --strategy=CppCompile=standalone

./bazel-bin/Source/Performance/MatMultCustomMain.exe


################################################################################
################################################################################

# C++ version of Performance

bazel build //Source/Performance:Performance --strategy=CppCompile=standalone

bazel test --test_output=all //Source/UnitTests:ConstructiconGoogleUnitTests --strategy=CppCompile=standalone

################################################################################
# cf. https://stackoverflow.com/questions/32823625/bazel-build-verbose-compiler-commands-logging

bazel build -s //Source/Performance:MatMultiplyTestBed.exe --strategy=CppCompile=standalone

./bazel-bin/Source/Performance/MatMultiplyTestBed.exe

