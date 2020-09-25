ADD_EXECUTABLE(DisplayMemoryAddress
  DisplayMemoryAddress.cpp)

SET_TARGET_PROPERTIES(DisplayMemoryAddress
  PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/UnitTests/Systems/")
