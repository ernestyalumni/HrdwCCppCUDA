OPTION(DEBUG_MATMUL "Debug Matrix Multiply" OFF)
OPTION(ASAN_MATMUL "Asan Matrix Multiply" OFF)
OPTION(RELEASE_MATMUL "Release Matrix Multiply" OFF)

ADD_EXECUTABLE(MatrixMultiplyTestBed
  MatrixMultiply/MatrixMultiply.cc
  MatrixMultiply/TestBed.cc
  )

SET_TARGET_PROPERTIES(MatrixMultiplyTestBed
  PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/")

# cf. https://stackoverflow.com/questions/24238937/how-to-change-a-compiler-flag-for-just-one-executable-in-cmake

IF (DEBUG_MATMUL)
  MESSAGE("Building Debug Matrix Multiply ${DEBUG_MATMUL}")

  SET_TARGET_PROPERTIES(MatrixMultiplyTestBed
    PROPERTIES
    COMPILE_FLAGS
    # These flags are applied. -g generates debugging symbols, -O0 disables
    # compiler optimization, so that the binary generated more directly
    # corresponds to your source code.
    "-g -O0")
ELSEIF(ASAN_MATMUL)

  MESSAGE("Building ASAN Matrix Multiply ${ASAN_MATMUL}")

  SET_TARGET_PROPERTIES(MatrixMultiplyTestBed
    PROPERTIES
    COMPILE_FLAGS
    # These flags are used to invoke the address sanitizer
    "-O1 -g -fsanitize=address")

  # cf. https://stackoverflow.com/questions/62770060/setting-compiler-linker-flags-per-target-in-cmake
  # cf. https://stackoverflow.com/questions/37970758/how-to-use-addresssanitizer-with-gcc

  IF (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")

    SET_TARGET_PROPERTIES(MatrixMultiplyTestBed
      PROPERTIES
      LINK_FLAGS
      "-fsanitize=address")
  ENDIF()

ELSEIF(RELEASE_MATMUL)

  MESSAGE("Building Release Matrix Multiply ${RELEASE_MATMUL}")

  SET_TARGET_PROPERTIES(MatrixMultiplyTestBed
    PROPERTIES
    COMPILE_FLAGS
    # In the release version, we ask for many optimizations; -O3 sets the
    # optimization level to 3.
    "-O3")

ENDIF()