CMAKE_MINIMUM_REQUIRED (VERSION 3.10)

# Language of the project is CXX (for C++)
PROJECT (VOLTRON LANGUAGES CXX)

# BEFORE prepends the directories specified onto current list of directories.
# cf. https://cmake.org/cmake/help/v3.0/command/include_directories.html
INCLUDE_DIRECTORIES(BEFORE ${CMAKE_SOURCE_DIR})

#LIST(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/../CMake)

# Include will load and run CMake code of this module, "CheckCXXCompilerFlag."
INCLUDE(CheckCXXCompilerFlag)

# cf. https://stackoverflow.com/questions/25451254/getting-cmake-check-cxx-compiler-flag-to-work

CHECK_CXX_COMPILER_FLAG(-std=c++20 COMPILER_SUPPORTS_CXX20)

# Check if Compiler supports the Working draft for ISO C++ 2020 standard.
CHECK_CXX_COMPILER_FLAG(-std=c++2a COMPILER_SUPPORTS_CXX2A)

IF(COMPILER_SUPPORTS_CXX20)
  SET(CMAKE_CXX_STANDARD "20") # C++20
  SET(CMAKE_CXX_STANDARD_REQUIRED ON) # ... is required
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++20")
ELSEIF(COMPILER_SUPPORTS_CXX2A)
  SET(CMAKE_CXX_STANDARD "20") # C++20, but with c++2a
  MESSAGE("Setting to the Working draft of C++20: " ${COMPILER_SUPPORTS_CXX2A})
  SET(CMAKE_CXX_STANDARD_REQUIRED ON) # ... is required
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++2a")
ELSE()
  SET(CMAKE_CXX_STANDARD "17")
  SET(CMAKE_CXX_STANDARD_REQUIRED ON)
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
ENDIF()

MESSAGE("CMAKE_CXX_STANDARD: " ${CMAKE_CXX_STANDARD})

FIND_PROGRAM(CCACHE_FOUND ccache)

IF(CCACHE_FOUND)
  SET(CMAKE_CXX_COMPILER_LAUNCHER ccache)
ENDIF()

SET(BOOST_MIN_VERSION "1.65.0")

IF (ALLOW_NEWER_BOOST)
#  FIND_PACKAGE(Boost ${BOOST_MIN_VERSION} REQUIRED)
  FIND_PACKAGE(Boost)
ELSE()
  # FIND_PACKAGE(Boost 1.65.0 EXACT REQUIRED)
  FIND_PACKAGE(Boost)
ENDIF()

# Include directories so that header files can be found.
INCLUDE_DIRECTORIES(BOOST_INCLUDES ${Boost_INCLUDE_DIRS})

# cf. https://cmake.org/cmake/help/latest/command/include_directories.html?highlight=s
# Add given directories to those compiler uses to search for include files.
# Relative paths are interpreted as relative to current source directory.
# If SYSTEM option given, compiler told directories meant as system include directories in some platforms.
INCLUDE_DIRECTORIES(SYSTEM ThirdParty)

# Calls a CMake module that first, searches the file system for the appropriate
# threads package for this platform, and then sets CMAKE_THREAD_LIBS_INIT
# variable (and other variables as well).
FIND_PACKAGE(Threads REQUIRED)

# cf. https://stackoverflow.com/questions/69282407/linking-to-tbb-libraries-with-cmake
FIND_PACKAGE(TBB REQUIRED)

# Do we need to include these source project directories? I don't know.
#INCLUDE_DIRECTORIES ("${PROJECT_SOURCE_DIR}/Hierarchy")
#  "${PROJECT_SOURCE_DIR}/UnitTests")

# Add_subdirectory call makes use of libraries.
# Add_subdirectory adds the source CMakeLists.txt and code files in the
# directory.
ADD_SUBDIRECTORY(Algorithms)
ADD_SUBDIRECTORY(Categories)
ADD_SUBDIRECTORY(Cpp)
ADD_SUBDIRECTORY(DataStructures)
#ADD_SUBDIRECTORY(DeskCalculator)
ADD_SUBDIRECTORY(FileIO)
ADD_SUBDIRECTORY(Hierarchy)
ADD_SUBDIRECTORY(IO)
ADD_SUBDIRECTORY(IPC)
ADD_SUBDIRECTORY(Projects)
ADD_SUBDIRECTORY(QuestionsDEntrevue)
ADD_SUBDIRECTORY(Tools)
ADD_SUBDIRECTORY(UnitTests)
ADD_SUBDIRECTORY(Utilities)

ADD_EXECUTABLE(Main Main.cpp)
ADD_EXECUTABLE(CoderByteChallenges CoderByteChallenges.cpp)
ADD_EXECUTABLE(HackerRankPractice HackerRankPractice.cpp)
ADD_EXECUTABLE(Entrevue Entrevue.cpp)

TARGET_LINK_LIBRARIES(Main
  Hierarchy)