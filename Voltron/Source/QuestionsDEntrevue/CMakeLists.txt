ADD_EXECUTABLE(DaysOld
  DaysOld02.cpp)

SET_TARGET_PROPERTIES(DaysOld
  PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/")

ADD_EXECUTABLE(WarmUpChallenges
  WarmUpChallenges.cpp)

SET_TARGET_PROPERTIES(WarmUpChallenges
  PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/")

ADD_LIBRARY(QuestionsDEntrevue
  LeetCode/LeetCodeQuestions.cpp
  PassionneDordinateurRang.cpp)

