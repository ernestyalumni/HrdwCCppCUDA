ADD_LIBRARY(FileIO
  FileDescriptor.cpp
  FileFlagsModes.cpp
  GetFileSize.cpp
  OpenFile.cpp)

TARGET_LINK_LIBRARIES(FileIO
  Utilities)
