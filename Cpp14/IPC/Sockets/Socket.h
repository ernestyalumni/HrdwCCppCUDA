//------------------------------------------------------------------------------
/// \file Socket.h
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief  A socket as RAII 
/// \ref      
/// \details Using RAII for socket. 
/// \copyright If you find this code useful, feel free to donate directly
/// (username ernestyalumni or email address above), going directly to: 
///
/// paypal.me/ernestyalumni
/// 
/// which won't go through a 3rd. party like indiegogo, kickstarter, patreon.
/// Otherwise, I receive emails and messages on how all my (free) material on
/// physics, math, and engineering have helped students with their studies, and
/// I know what it's like to not have money as a student, but love physics (or 
/// math, sciences, etc.), so I am committed to keeping all my material 
/// open-source and free, whether or not sufficiently crowdfunded, under the 
/// open-source MIT license: feel free to copy, edit, paste, make your own
/// versions, share, use as you wish.    
/// Peace out, never give up! -EY
//------------------------------------------------------------------------------
/// COMPILATION TIPS:
///   g++ -I ../../ -std=c++14 Socket.cpp Socket_main.cpp ../../Utilities/ErrorHandling.cpp -o Socket_main
//------------------------------------------------------------------------------
#ifndef _IPC_SOCKET_H_
#define _IPC_SOCKET_H_

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
//------------------------------------------------------------------------------
enum class Domains : int
{
  unix = AF_UNIX, // Local communication
  local = AF_LOCAL, // Local communication
  ipv4_internet = AF_INET,
  ipv6_internet = AF_INET6,
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

//------------------------------------------------------------------------------
/// \brief Creates an endpoint for communication
//------------------------------------------------------------------------------
class Socket
{
  public:

    Socket(const Domains domain, const int type, const int protocol = 0);

    virtual ~Socket();

    // Accessors

    Domains domain() const
    {
      return domain_;
    }

    int domain_as_int() const
    {
      return static_cast<int>(domain_);
    }

    int type() const
    {
      return type_;
    }

    int protocol() const
    {
      return protocol_;
    }

  protected:

    // protected, so one wouldn't be able to do something like ::close(fd)
    const int fd() const
    {
      return fd_;
    }

  private:

    Domains domain_;
    int type_;
    int protocol_;

    int fd_;
};

} // namespace Sockets
} // namespace IPC

#endif // _IPC_SOCKET_H_

