# cf. https://stackoverflow.com/questions/50468620/what-does-enable-testing-do-in-cmake
#ENABLE_TESTING()

# Dependencies
# Find a boost install with the libraries unit_test_framework.
# Boost filesystem needs boost system.
FIND_PACKAGE(Boost COMPONENTS filesystem system unit_test_framework REQUIRED)

# Build unit tests
ADD_DEFINITIONS(-DBOOST_TEST_DYN_LINK)

ADD_EXECUTABLE(Check
  Main.cpp
  Algorithms/Permutations_tests.cpp
  Algorithms/Search_tests.cpp
  Algorithms/Sorting_tests.cpp
  Categories/AlgebraicDataTypes/ProgramStateExample_tests.cpp
  Categories/AlgebraicDataTypes/WordCountingState_tests.cpp
  Categories/Functors_tests.cpp
  Categories/Monoidal/Monad_tests.cpp
  Categories/Monoidal/Monoid_tests.cpp
  Cpp/Functions_tests.cpp
  Cpp/Recursion_tests.cpp
  Cpp/IO/IO_tests.cpp
  Cpp/Operations_test.cpp
  Cpp/Std/Algorithm/Transform_tests.cpp
  Cpp/Std/Pointers_tests.cpp
  Cpp/Templates/Template_Metaprogramming_tests.cpp
  DataStructures/DataStructures_tests.cpp
  DataStructures/LinkedLists_tests.cpp
  DeskCalculator/Token_tests.cpp
  Hierarchy/AccessAndInheritance_tests.cpp
  Mutable/Mutable_tests.cpp
  Mutable/Mutex_tests.cpp
  Tools/TemporaryDirectoriesAndFiles_tests.cpp
  Utilities/BitSet_test.cpp
  Utilities/Curry_test.cpp
  Utilities/FunctionObject_test.cpp
  Utilities/LazyValuation_tests.cpp
  Utilities/NarrowCast_test.cpp
  Utilities/OnlyOneArgument_test.cpp
  Utilities/StdForward_tests.cpp
  Utilities/StdTuple_tests.cpp
  Utilities/ToBytes_test.cpp
  Utilities/ToHexString_tests.cpp)

TARGET_LINK_LIBRARIES(Check
  Algorithms
  Hierarchy
  Tools
  # cf. https://stackoverflow.com/questions/1620918/cmake-and-libpthread
  ${CMAKE_THREAD_LIBS_INIT}
  ${Boost_FILESYSTEM_LIBRARY}
  ${Boost_SYSTEM_LIBRARY}
  ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY})

SET_TARGET_PROPERTIES(Check
  PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/")