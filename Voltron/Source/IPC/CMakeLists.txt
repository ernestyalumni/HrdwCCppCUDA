ADD_LIBRARY(IPC
  Sockets/Bind.cpp
  Sockets/GetSocketName.cpp
  Sockets/Listen.cpp
  Sockets/Receive.cpp
  Sockets/Socket.cpp
  Sockets/InternetAddress.cpp)

TARGET_LINK_LIBRARIES(IPC
  Utilities)