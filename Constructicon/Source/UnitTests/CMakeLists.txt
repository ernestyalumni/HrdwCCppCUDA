ENABLE_TESTING()

ADD_EXECUTABLE(Check
  Utilities/AsBits_tests.cpp
  Utilities/Time/GetClockTime_tests.cpp
  Utilities/Time/TimeSpecification_tests.cpp
  )

TARGET_LINK_LIBRARIES(Check
  Utilities
  gtest_main)

INCLUDE(GoogleTest)
gtest_discover_tests(Check)

SET_TARGET_PROPERTIES(Check
  PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/")
