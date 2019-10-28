#CMAKE_MINIMUM_REQUIRED (VERSION 3.10)

# cf. https://stackoverflow.com/questions/50468620/what-does-enable-testing-do-in-cmake
#ENABLE_TESTING()

# Dependencies
# Find a boost install with the libraries unit_test_framework.
FIND_PACKAGE(Boost COMPONENTS filesystem unit_test_framework REQUIRED)

# Build unit tests
ADD_DEFINITIONS(-DBOOST_TEST_DYN_LINK)

ADD_SUBDIRECTORY(Categories)

#ADD_LIBRARY(UnitTesting
#  Utilities/FunctionObject_test.cpp)

ADD_EXECUTABLE(Check
  Main.cpp
  Categories/Functors_tests.cpp
  Categories/Monoidal/Monad_tests.cpp
  Categories/Monoidal/Monoid_tests.cpp
  Cpp/Functions_tests.cpp
  Cpp/Operations_test.cpp
  Cpp/Std/Algorithm/Transform_tests.cpp
  DeskCalculator/Token_tests.cpp
  Utilities/BitSet_test.cpp
  Utilities/Curry_test.cpp
  Utilities/FunctionObject_test.cpp
  Utilities/NarrowCast_test.cpp
  Utilities/OnlyOneArgument_test.cpp
  Utilities/StdForward_tests.cpp
  Utilities/ToBytes_test.cpp)

TARGET_LINK_LIBRARIES(Check
  Hierarchy
  ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY})

SET_TARGET_PROPERTIES(Check
  PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/")