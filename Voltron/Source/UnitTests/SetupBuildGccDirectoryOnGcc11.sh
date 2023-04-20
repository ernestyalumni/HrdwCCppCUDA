#/bin/bash

mkdir BuildGcc
cd BuildGcc
cmake -DCMAKE_CXX_COMPILER=g++-11 -DCMAKE_C_COMPILER=gcc-11 ../Source/
make -j