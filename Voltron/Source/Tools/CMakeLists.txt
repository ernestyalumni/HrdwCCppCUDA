ADD_LIBRARY(Tools
  CaptureCout.cpp
  Filepaths.cpp
  TemporaryDirectory.cpp)

TARGET_LINK_LIBRARIES(Tools
  ${Boost_FILESYSTEM_LIBRARY}
  ${Boost_SYSTEM_LIBRARY})

