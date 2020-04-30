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
enum class Domains : int
{
  unix_ = AF_UNIX, // Local communication
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
  crypto = AF_ALG // Interface to kernel crypto API
};

//------------------------------------------------------------------------------
/// \brief enum class for specifying communication semantics
//------------------------------------------------------------------------------
enum class Types : int
{
  stream = SOCK_STREAM, // Provides sequenced, reliable, 2-way, connection
  datagram = SOCK_DGRAM, // supports datagrams (connectionless, unreliable)
  sequenced_packets = SOCK_SEQPACKET, 
  raw = SOCK_RAW, // provides raw network protocol access.
  reliable_datagram_layer = SOCK_RDM, // provides reliable datagram layer
  nonblocking = SOCK_NONBLOCK, 
  close_on_exec = SOCK_CLOEXEC
};

} // namespace Sockets
} // namespace IPC

#endif // IPC_SOCKETS_PARAMETER_FAMILIES_H
