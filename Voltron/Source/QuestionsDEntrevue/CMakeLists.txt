ADD_EXECUTABLE(DaysOld
  DaysOld02.cpp)

SET_TARGET_PROPERTIES(DaysOld
  PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/")

ADD_LIBRARY(QuestionsDEntrevue
  CameraCSIDataParser.cpp
  LeetCode/LeetCodeQuestions.cpp
  PageFaults.cpp
  PassionneDordinateurRang.cpp
  Streams/FireAlertParser.cpp
  TirePressureSensor.cpp)

