//------------------------------------------------------------------------------
/// \file ParameterFamilies.h
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \ref https://linux.die.net/man/2/socket
//------------------------------------------------------------------------------
#ifndef IPC_SOCKETS_PARAMETER_FAMILIES_H
#define IPC_SOCKETS_PARAMETER_FAMILIES_H

#include <sys/socket.h> // AF_INET,AF_UNIX,AF_LOCAL,etc.; communication domains

namespace IPC
{

namespace Sockets
{

//------------------------------------------------------------------------------
/// \brief enum class for all socket domains (communication domains)
/// \details Selects protocol family which will be used for communication.
///   These families are defined in <sys/socket.h>
/// \ref https://linux.die.net/man/2/socket
/// https://manpages.ubuntu.com/manpages/precise/man7/unix.7.html
///
/// AF_UNIX, also known as AF_LOCAL socket family us used to communicate between
/// processes on same machine efficiently. Traditionally, UNIX domain sockets
/// can be either unnamed, or bound to file system pathname (marked as being of
/// type socket).
///
/// Valid types for AF_UNIX are SOCK_STREAM, for strema-oriented socket, and
/// SOCK_DGRAM, for datagram-oriented socket that preserves message boundaries.
//------------------------------------------------------------------------------
enum class Domain : int
{
  unix_ = AF_UNIX,
  local = AF_LOCAL, // Local communication
  ipv4 = AF_INET,
  ipv6 = AF_INET6,
  ipx_Novell = AF_IPX,
  kernel_user_interface = AF_NETLINK, // Kernel user interface device
  itu_t_x25 = AF_X25, // ITU-T X.25 / ISO-8208 protocol
  amateur_radio = AF_AX25, // Amateur radio AX.25 protocol
  raw_ATM_PVCs = AF_ATMPVC, // Access to raw ATM PVCs
  appletalk = AF_APPLETALK, // AppleTalk
  packet = AF_PACKET, // low level packet interface 
  crypto = AF_ALG, // Interface to kernel crypto API
  // https://students.mimuw.edu.pl/SO/Linux/Kod/include/linux/socket.h.html
  maximum = AF_MAX // For now ...
};

//------------------------------------------------------------------------------
/// \brief enum class for specifying communication semantics
//------------------------------------------------------------------------------
enum class Type : int
{
  stream = SOCK_STREAM, // Provides sequenced, reliable, 2-way, connection
  datagram = SOCK_DGRAM, // supports datagrams (connectionless, unreliable)
  sequenced_packets = SOCK_SEQPACKET, 
  raw = SOCK_RAW, // provides raw network protocol access.
  reliable_datagram_layer = SOCK_RDM // provides reliable datagram layer
};

enum class BehaviorModifyingValue : int
{
  nonblocking = SOCK_NONBLOCK,
  close_on_exec = SOCK_CLOEXEC  
};

//------------------------------------------------------------------------------
// \ref https://www.gnu.org/software/libc/manual/html_node/Socket_002dLevel-Options.html
//------------------------------------------------------------------------------
enum class Level : int
{
  socket = SOL_SOCKET
};

//------------------------------------------------------------------------------
/// \brief enum class for socket-level options
/// \ref https://linux.die.net/man/3/setsockopt
/// http://man7.org/linux/man-pages/man7/socket.7.html
/// https://linux.die.net/man/7/socket
//------------------------------------------------------------------------------
enum class Option : int
{
  // Turns on recording of debugging information. Option enables or disabled
  // debugging in underlying protocol modules. Option takes int value. This is
  // a Boolean option.
  debug = SO_DEBUG,
  // Permits sending broadcast messages, if supported by protocol. Option takes
  // int value. This is a boolean option.
  broadcast = SO_BROADCAST,
  // Specify receiving timeouts until reporting error.
  // argument is struct timeval.
  // If input or output function blocks for this time period, and data has been
  // received, return value of that function will be amount of data transferred;
  // if no data been transferred and timeout reached, then -1 returned with
  // errno set to EAGAIN or EWOULDBLOCK, or EINPROGRESS (for connect), just as
  // if socket was specified to be nonblocking.
  // If timeout set to 0 (default), operation will never timeout.
  // timeouts only have effect for system calls that perform socket I/O;
  // timeouts have no effect for select, poll, epoll_wait.
  receiving_timeout = SO_RCVTIMEO,
  // Specifies rules used in validating addresses supplied to ::bind() should
  // allow reuse of local addresses, if this is supported by protocol. Option
  // takes int value. This is a Boolean option.
  reuse_address = SO_REUSEADDR,
  // Permits multiple AF_INET or AF_INET6 sockets to be bound to identical
  // socket address. Option must be set on each socket (including first socket)
  // prior to calling ::bind on socket.
  reuse_port = SO_REUSEPORT,
  // Keeps connections active by enabling periodic transmission of messages, if
  // this is supported by protocol. Option takes int value.
  keep_alive = SO_KEEPALIVE,
  // enable or disable receiving of SO_TIMESTAMP control message. Timestamp
  // control message sent with level SOL_SOCKET and cmsg_type of SCM_TIMESTAMP.
  timestamp = SO_TIMESTAMP
};

} // namespace Sockets
} // namespace IPC

#endif // IPC_SOCKETS_PARAMETER_FAMILIES_H
