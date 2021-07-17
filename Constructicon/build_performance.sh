#!/bin/bash

# cf. https://www.xmodulo.com/command-line-arguments-bash-script.html
# Read in command lines by their positions

# Checks if first argument exists, and proceeds with checking for option.
if [ ! -z $1 ]; then
    echo "First argument is set and is: " $1
fi

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