ENABLE_TESTING()

ADD_EXECUTABLE(Check
  Concurrency/ThreadPool_tests.cpp
  Dependencies/RapidJson_tests.cpp
  Dependencies/json_tests.cpp
  Utilities/AsBits_tests.cpp
  Utilities/ErrorHandling/signal_child_handler_tests.cpp
  Utilities/FileIO/Filestream_tests.cpp
  Utilities/FileIO/ProjectPaths_tests.cpp
  Utilities/Testing/CaptureCerr_tests.cpp
  Utilities/Testing/CaptureCout_tests.cpp
  Utilities/Time/ElapsedTime_tests.cpp
  Utilities/Time/GetClockTime_tests.cpp
  Utilities/Time/TimeSpecification_tests.cpp
  )

TARGET_LINK_LIBRARIES(Check
  Concurrency
  Utilities
  gtest_main
  nlohmann_json::nlohmann_json
  )

INCLUDE(GoogleTest)
gtest_discover_tests(Check)

SET_TARGET_PROPERTIES(Check
  PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/")
