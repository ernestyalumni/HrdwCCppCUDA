ADD_SUBDIRECTORY(Bits)
ADD_SUBDIRECTORY(PeakFinding)

ADD_LIBRARY(Algorithms
  BinarySearch.cpp
  Bits/OnesComplementSum.cpp
  Bits/PrintBytes.cpp
  PeakFinding.cpp
  Permutations.cpp
  Recursion.cpp
  QuickSort.cpp)