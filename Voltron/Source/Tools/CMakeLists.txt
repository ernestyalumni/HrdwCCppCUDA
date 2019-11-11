ADD_LIBRARY(Tools
  TemporaryDirectory.cpp)

TARGET_LINK_LIBRARIES(Tools
  ${Boost_FILESYSTEM_LIBRARY}
  ${Boost_SYSTEM_LIBRARY})

