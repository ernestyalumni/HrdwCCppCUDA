ENABLE_TESTING()

ADD_EXECUTABLE(Check
  CUDASamples/Introduction/VectorAdd_tests.cu
  CUDAToolkit/MemoryManagement_tests.cu
  )

TARGET_LINK_LIBRARIES(Check
  CUDASamples
  gtest_main)

INCLUDE(GoogleTest)
gtest_discover_tests(Check)

SET_TARGET_PROPERTIES(Check
  PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

SET_TARGET_PROPERTIES(Check
  PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/")