ADD_LIBRARY(FileIO
  FileDescriptor.cpp
  FileFlagsModes.cpp
  OpenFile.cpp)

TARGET_LINK_LIBRARIES(FileIO
  Utilities)
