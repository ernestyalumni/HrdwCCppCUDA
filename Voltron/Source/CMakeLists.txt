CMAKE_MINIMUM_REQUIRED (VERSION 3.10)

# Language of the project is CXX (for C++)
PROJECT (VOLTRON CXX)

# Include will load and run CMake code of this module, "CheckCXXCompilerFlag."
INCLUDE(CheckCXXCompilerFlag)

# cf. https://stackoverflow.com/questions/25451254/getting-cmake-check-cxx-compiler-flag-to-work
CHECK_CXX_COMPILER_FLAG(-std=c++17 COMPILER_SUPPORTS_CXX17)

IF(COMPILER_SUPPORTS_CXX17)
  SET(CMAKE_CXX_STANDARD 17) # C++17
  SET(CMAKE_CXX_STANDARD_REQUIRED ON) # ... is required
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
ELSE()
  SET(CMAKE_CXX_STANDARD 14)
  SET(CMAKE_CXX_STANDARD_REQUIRED ON)
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
ENDIF()

FIND_PROGRAM(CCACHE_FOUND ccache)

IF(CCACHE_FOUND)
  SET(CMAKE_CXX_COMPILER_LAUNCHER ccache)
ENDIF()

# Include directories so that header files can be found.
INCLUDE_DIRECTORIES ("${PROJECT_SOURCE_DIR}/Hierarchy")
#  "${PROJECT_SOURCE_DIR}/UnitTests")

# add_subdirectory call makes use of libraries.
ADD_SUBDIRECTORY(Hierarchy
  UnitTests)

ADD_EXECUTABLE(Main Main.cpp)

#ADD_EXECUTABLE(MainRuleOf5 Main_RuleOf5.cpp)

#TARGET_LINK_LIBRARY(MainRuleOf5 Hierarchy)

#ADD_EXECUTABLE(Check
#  Main.cpp)

#TARGET_LINK_LIBRARIES(Check
#  Hierarchy)

