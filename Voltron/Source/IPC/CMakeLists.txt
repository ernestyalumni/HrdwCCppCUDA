ADD_LIBRARY(IPC
  GetReadyAddressInfo.cpp
  MessageQueue/CreateMessageQueue.cpp
  MessageQueue/CreateOrOpen.cpp
  MessageQueue/FlagsAndModes.cpp
  MessageQueue/MessageQueueDescription.cpp
  MessageQueue/Receive.cpp
  MessageQueue/Send.cpp
  Sockets/Bind.cpp
  Sockets/CreateSocket.cpp
  Sockets/GetSocketName.cpp
  Sockets/InternetAddress.cpp
  Sockets/Listen.cpp
  Sockets/Receive.cpp
  Sockets/SetSocketOptions.cpp
  Sockets/Socket.cpp
  Sockets/TCP/CreateSocket.cpp
  Sockets/UDP/CreateSocket.cpp
  Sockets/UDP/Receiver.cpp
  Sockets/UDP/Sender.cpp)

#FIND_LIBRARY(LIBRT rt)
#IF (LIBRT)
#  TARGET_LINK_LIBRARIES(IPC ${LIBRT})
#ENDIF()

TARGET_LINK_LIBRARIES(IPC
  FileIO
  Utilities
  -lrt)