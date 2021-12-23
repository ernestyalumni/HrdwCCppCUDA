# cf. https://stackoverflow.com/questions/50468620/what-does-enable-testing-do-in-cmake
#ENABLE_TESTING()

# Dependencies
# Find a boost install with the libraries unit_test_framework.
# Boost filesystem needs boost system.
FIND_PACKAGE(Boost COMPONENTS filesystem system unit_test_framework REQUIRED)

# Build unit tests
ADD_DEFINITIONS(-DBOOST_TEST_DYN_LINK)

ADD_EXECUTABLE(Check
  Main.cpp
  Algorithms/Bits/OnesComplement_tests.cpp
  Algorithms/Bits/PrintBytes_tests.cpp
  Algorithms/Bits/Radix_tests.cpp
  Algorithms/DynamicProgramming/Memoization_tests.cpp
  Algorithms/ExpertIo/Level1_tests.cpp
  Algorithms/HackerRank/BitManipulationLonelyInteger_tests.cpp
  Algorithms/HackerRank/BubbleSort_tests.cpp
  Algorithms/HackerRank/CountingSort_tests.cpp
  Algorithms/HackerRank/InsertionSort_tests.cpp
  Algorithms/HackerRank/QuickSort_tests.cpp
  Algorithms/LeetCode/Course_Schedule_II_210_tests.cpp
  Algorithms/LeetCode/PrintMatrixSpiral_tests.cpp
  Algorithms/LeetCode/Surrounded_Regions_130_tests.cpp
  Algorithms/PeakFinding/LoadPeakProblem_tests.cpp
  Algorithms/PeakFinding_tests.cpp
  Algorithms/Permutations_tests.cpp
  Algorithms/Recursion_tests.cpp
  Algorithms/Search_tests.cpp
  Algorithms/Sorting/SelectionSort_tests.cpp
  Algorithms/Sorting_tests.cpp
  Categories/AlgebraicDataTypes/ProgramStateExample_tests.cpp
  Categories/AlgebraicDataTypes/WordCountingState_tests.cpp
  Categories/Category_tests.cpp
  Categories/Functors_tests.cpp
  Categories/Monads/ContinuationMonad_tests.cpp
  Categories/Monads/IOMonad_tests.cpp
  Categories/Monads/OptionalMonad_tests.cpp
  # TODO: Use C++17 C++20 exclusively for std::filesystem
  #Categories/Monads/ReaderMonad_tests.cpp
  Categories/Monads/OldStateMonad_tests.cpp
  Categories/Monads/StateMonad_tests.cpp
  Categories/Monads/StateMonadExamples_tests.cpp
  Categories/Monads/WriterMonad_tests.cpp
  Categories/Monoidal/Monad_tests.cpp
  Categories/Monoidal/Monoid_tests.cpp
  Cpp/CallByValueCallByReference_tests.cpp
  Cpp/Classes/BitField_tests.cpp
  Cpp/Classes/ConstructorsToDestructors_tests.cpp
  Cpp/Classes/CopyMoveArrays_tests.cpp
  Cpp/Closure_tests.cpp
  Cpp/Enumeration_tests.cpp
  Cpp/FileIO/CStyleFileIO_tests.cpp
  Cpp/FileIO/GetCurrentDirectory_tests.cpp
  Cpp/Functions_tests.cpp
  Cpp/IO/IO_tests.cpp
  Cpp/IO/StdStringStream_tests.cpp
  Cpp/IO/Streams_tests.cpp
  Cpp/Keywords_tests.cpp
  Cpp/MainCommandLineArguments_tests.cpp
  Cpp/Numerics/BitCast_tests.cpp
  Cpp/Operations_test.cpp
  Cpp/PointersArraysReferences_tests.cpp
  Cpp/Recursion_tests.cpp
  Cpp/RuleOf5_tests.cpp
  Cpp/Std/Algorithm/Algorithms_tests.cpp
  Cpp/Std/Algorithm/EqualAndMismatch_tests.cpp
  Cpp/Std/Algorithm/LeviCivitaSymbol_tests.cpp
  Cpp/Std/Algorithm/Transform_tests.cpp
  Cpp/Std/Containers/SpanView_tests.cpp
  Cpp/Std/Containers/Vector_tests.cpp
  Cpp/Std/Pointers_tests.cpp
  Cpp/Std/Span_tests.cpp
  Cpp/Std/Utilities/Optional_tests.cpp
  Cpp/Strings/StringView_tests.cpp
  Cpp/Strings/Strings_tests.cpp
  Cpp/Templates/FoldExpressions_tests.cpp
  Cpp/Templates/FunctionT/FunctionT_tests.cpp
  Cpp/Templates/FunctionT/Max_tests.cpp
  Cpp/Templates/FunctionT/MoreMax_tests.cpp
  Cpp/Templates/Template_Metaprogramming_tests.cpp
  Cpp/Templates/TypeUtilities_tests.cpp
  Cpp/Threads/Future_tests.cpp
  Cpp/Utilities/FunctionObject_test.cpp
  Cpp/Utilities/FunctionObjects/Invoke_tests.cpp
  Cpp/Utilities/FunctionObjects/References_tests.cpp
  Cpp/Utilities/Memory/PointerWrappers_tests.cpp
  Cpp/Utilities/Memory/SmartPointers_tests.cpp
  Cpp/Utilities/Memory/StdAddressOf_tests.cpp
  Cpp/Utilities/Memory/StdToAddress_tests.cpp
  Cpp/Utilities/SignalHandler_tests.cpp
  Cpp/Utilities/StdInitializerList_tests.cpp
  Cpp/Utilities/SuperBitSet_tests.cpp
  DataStructures/Arrays/ArrayQuestions_tests.cpp
  DataStructures/Arrays/Array_tests.cpp
  DataStructures/Arrays/FixedSizeArrays_tests.cpp
  DataStructures/Arrays/ResizeableArray_tests.cpp
  DataStructures/Arrays_tests.cpp
  DataStructures/BagsImplementations_tests.cpp
  DataStructures/BinarySearchTrees_tests.cpp
  DataStructures/BinaryTrees_tests.cpp
  DataStructures/DataStructures_tests.cpp
  DataStructures/HashSet_tests.cpp
  DataStructures/Heaps/BinaryHeap_tests.cpp
  DataStructures/LinkedLists/LinkedList_tests.cpp
  DataStructures/LinkedLists/Node_tests.cpp
  DataStructures/LinkedLists/SingleNode_tests.cpp
  DataStructures/LinkedLists_tests.cpp
  DataStructures/Lists/ArrayList_tests.cpp
  DataStructures/Queues/CircularQueue_tests.cpp
  DataStructures/Queues/DynamicQueue_tests.cpp
  DataStructures/Queues/QueuesImplementations_tests.cpp
  DataStructures/Queues/QueuesWithArrays_tests.cpp
  DataStructures/Queues/Queues_tests.cpp
  DataStructures/Queues/ResizableDeque_tests.cpp
  DataStructures/Stacks/DijkstraExpressionEvaluation_tests.cpp
  DataStructures/Stacks/DynamicStack_tests.cpp
  DataStructures/Stacks/StacksImplementations_tests.cpp
  DataStructures/Stacks/StacksWithArrays_tests.cpp
  DataStructures/Stacks/Stacks_tests.cpp
  DeskCalculator/Token_tests.cpp
  Entrevue/Entrevue_tests.cpp
  Entrevue/LeetCode/LeetCode_tests.cpp
  Entrevue/MiscellaneousQuestions_tests.cpp
  FileIO/OpenFile_tests.cpp
  Hierarchy/AccessAndInheritance_tests.cpp
  IO/Epoll/ControlInterestList_tests.cpp
  IO/Epoll/EpollFd_tests.cpp
  IPC/MessageQueue/CreateOrOpen_tests.cpp
  IPC/Sockets/Bind_tests.cpp
  IPC/Sockets/CreateSocket_tests.cpp
  IPC/Sockets/GetSocketName_tests.cpp
  IPC/Sockets/InternetAddress_tests.cpp
  IPC/Sockets/Listen_tests.cpp
  IPC/Sockets/ParameterFamilies_tests.cpp
  IPC/Sockets/SetSocketOptions_tests.cpp
  IPC/Sockets/Socket_tests.cpp
  Mutable/Mutable_tests.cpp
  Mutable/Mutex_tests.cpp
  Tools/CaptureCout_tests.cpp
  Tools/Filepaths_tests.cpp
  Tools/TemporaryDirectoriesAndFiles_tests.cpp
  Utilities/BitsStream_tests.cpp
  Utilities/BooleanAlgebra_tests.cpp
  Utilities/Curry_test.cpp
  Utilities/ErrorHandling/ErrorHandling_tests.cpp
  Utilities/ErrorHandling/ErrorNumber_tests.cpp
  Utilities/ErrorHandling/HandleError_tests.cpp
  Utilities/ErrorHandling/HandleReturnValue_tests.cpp
  Utilities/FloatingPoint_tests.cpp
  Utilities/FromAddress_tests.cpp
  Utilities/HexDump_tests.cpp
  Utilities/LazyValuation_tests.cpp
  Utilities/Memory/HashTable_tests.cpp
  Utilities/NarrowCast_test.cpp
  Utilities/OnlyOneArgument_test.cpp
  Utilities/SplitString_tests.cpp
  Utilities/StdForward_tests.cpp
  Utilities/StdTuple_tests.cpp
  Utilities/Time/GetElapsedTime_tests.cpp
  Utilities/Time/TimeSpec_tests.cpp
  Utilities/ToBytes_test.cpp
  Utilities/ToHexString_tests.cpp)

ADD_LIBRARY(UnitTestTools
  Tools/Contains.cpp)

TARGET_LINK_LIBRARIES(Check
  Algorithms
  Categories
  Cpp
  DataStructures
  FileIO
  Hierarchy
  IO
  IPC
  QuestionsDEntrevue
  Tools
  UnitTestTools
  Utilities
  # cf. https://stackoverflow.com/questions/1620918/cmake-and-libpthread
  ${CMAKE_THREAD_LIBS_INIT}
  ${Boost_FILESYSTEM_LIBRARY}
  ${Boost_SYSTEM_LIBRARY}
  ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY})

# Recognizes Windows Subsystem Linux through CMAKE_SYSTEM; CMAKE_SYSTEM_NAME
# results only in "Linux".
IF(CMAKE_SYSTEM MATCHES "^Linux\..*microsoft-standard-WSL2$")
  MESSAGE("Recognizes WSL2 : " ${CMAKE_SYSTEM})
  MESSAGE("But CMake System Name: " ${CMAKE_SYSTEM_NAME})

  FIND_PACKAGE(TBB REQUIRED)
  TARGET_LINK_LIBRARIES(Check TBB::tbb)
ELSE()
  MESSAGE("Not WSL2 : " ${CMAKE_SYSTEM})
  MESSAGE("But CMake System Name: " ${CMAKE_SYSTEM_NAME})
ENDIF()

SET_TARGET_PROPERTIES(Check
  PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/")

# Without specifying target property runtime_output_directory, this goes into
# build folder's subdirectory "UnitTests"

ADD_EXECUTABLE(CommandLineArguments
  MainCommandLineArguments.cpp)

ADD_EXECUTABLE(TClapping
  MainTClap.cpp)

ADD_EXECUTABLE(UDP-Receiver
  MainUDP-Receiver.cpp)

TARGET_LINK_LIBRARIES(UDP-Receiver
  IPC
  Utilities)

ADD_EXECUTABLE(UDP-Sender
  MainUDP-Sender.cpp)

TARGET_LINK_LIBRARIES(UDP-Sender
  IPC
  Utilities)

ADD_EXECUTABLE(MQTestReceive
  IPC/MessageQueue/MainTestReceive.cpp)

TARGET_LINK_LIBRARIES(MQTestReceive
  IPC
  Utilities)

ADD_EXECUTABLE(MQTestSend
  IPC/MessageQueue/MainTestSend.cpp)

TARGET_LINK_LIBRARIES(MQTestSend
  IPC
  Utilities)

ADD_EXECUTABLE(StringViewAvoidsCopying
  Cpp/Strings/StringViewAvoidsCopying.cpp)

# LeetCode

ADD_EXECUTABLE(LeetCodePlayground
  Entrevue/LeetCode/LeetCodePlayground.cpp)

SET_TARGET_PROPERTIES(LeetCodePlayground
  PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/LeetCode/")
