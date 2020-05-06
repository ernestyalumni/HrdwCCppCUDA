ADD_LIBRARY(IPC
  MessageQueue/CreateOrOpen.cpp
  Sockets/Bind.cpp
  Sockets/GetSocketName.cpp
  Sockets/InternetAddress.cpp
  Sockets/Listen.cpp
  Sockets/Receive.cpp
  Sockets/Socket.cpp
  Sockets/UDP/CreateSocket.cpp
  Sockets/UDP/Receiver.cpp
  Sockets/UDP/Sender.cpp)

#FIND_LIBRARY(LIBRT rt)
#IF (LIBRT)
#  TARGET_LINK_LIBRARIES(IPC ${LIBRT})
#ENDIF()

TARGET_LINK_LIBRARIES(IPC
  Utilities
  -lrt)