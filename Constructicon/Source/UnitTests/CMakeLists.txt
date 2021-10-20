ENABLE_TESTING()

ADD_EXECUTABLE(Check
  Utilities/AsBits_tests.cpp
  Utilities/Time/SuperTimeSpecification_tests.cc
  )

TARGET_LINK_LIBRARIES(Check
  gtest_main)

INCLUDE(GoogleTest)
gtest_discover_tests(Check)

SET_TARGET_PROPERTIES(Check
  PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/")
