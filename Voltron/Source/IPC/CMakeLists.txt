ADD_LIBRARY(IPC
  Sockets/Bind.cpp
  Sockets/GetSocketName.cpp
  Sockets/InternetAddress.cpp
  Sockets/Listen.cpp
  Sockets/Receive.cpp
  Sockets/Socket.cpp
  Sockets/UDP/CreateSocket.cpp
  Sockets/UDP/Receiver.cpp
  Sockets/UDP/Sender.cpp)

TARGET_LINK_LIBRARIES(IPC
  Utilities)