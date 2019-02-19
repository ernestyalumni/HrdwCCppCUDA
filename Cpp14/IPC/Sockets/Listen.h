//------------------------------------------------------------------------------
/// \file Listen.h
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief  Wrapper for ::listen as a C++ functor.
/// \ref http://man7.org/linux/man-pages/man2/bind.2.html
/// \details Listen for connections on a socket.
/// To accept connections,
/// 1. Socket created with ::socket
/// 2. Socket bound to local address using ::bind, so other sockets may be
/// ::connect 'ed to it
/// 3. Willingness to accept incoming connections and a queue limit for incoming
/// connections are specified with ::listen()
/// 4. Connections accepted with ::accept().
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
///   g++ -I ../../ -std=c++14 Socket.cpp Listen.cpp Listen_main.cpp \
///     ../../Utilities/ErrorHandling.cpp ../../Utilities/Errno.cpp -o \
///     Listen_main
//------------------------------------------------------------------------------
#ifndef _IPC_SOCKETS_LISTEN_H_
#define _IPC_SOCKETS_LISTEN_H_

#include "Socket.h"
#include "Utilities/ErrorHandling.h" // HandleReturnValue

namespace IPC
{

namespace Sockets
{

/// \url https://stackoverflow.com/questions/33636157/\
/// c-declaring-a-class-in-another-class-header-file
class Socket;

//------------------------------------------------------------------------------
/// \class Listen
/// \url http://man7.org/linux/man-pages/man2/listen.2.html
/// \brief listen for connections on a socket
/// \details 
/// #include <sys/socket.h>
/// int listen(int sockfd, int backlog);
/// ::listen() marks socket referred to by sockfd as a passive socket, that is,
/// as a socket that'll be used to accept incoming connection requests using
/// ::accept.
/// sockfd argument is a fd that refers to socket of type SOCK_STREAM or
/// SOCK_SEQPACKET.
/// backlog argument defines max. length to which queue of pending connections
/// for sockfd may grow.
///   - If connection request arrives when queue is full, client may receive an
///     error with indication of ECONNREFUSED, or, if underlying protocol
///     supports retransmission, request may be ignored so that a later
///     reattempt at connection succeeds.
//------------------------------------------------------------------------------
class Listen
{
  public:

    Listen();

    Listen(const int backlog);

    void operator()(Socket& socket);

    void operator()(Socket& socket, const int backlog);

  protected:

    //--------------------------------------------------------------------------
    /// \class HandleListen
    /// \ref http://man7.org/linux/man-pages/man2/listen.2.html
    /// \brief On success, 0 returned. On error, -1 returned, and errno is set
    /// appropriately.
    /// \details ERRORS:
    /// EADDRINUSE Another socket is already listening on the same port.
    /// EADDRINUSE (Internet domain sockets) The socket referred to by sockfd
    /// had not previously been bound to an address and, upon attempting to bind
    /// it to an ephemeral port, it was determined that all port numbers in the
    /// ephemeral port range are currently in use.
    /// EBADF The argument sockfd isn't a valid fd.
    /// ENOTSOCK fd sockfd doesn't refer to a socket.
    /// EOPNOTSUPP Socket isn't of a type that supports ::listen() operation.
    //--------------------------------------------------------------------------
    class HandleListen : public Utilities::ErrorHandling::HandleReturnValue
    {
      public:

        HandleListen() = default;

        void operator()(const int result)
        {
          this->operator()(
            result,
            "Listen for connections on a socket (::listen)");
        }

      private:

        using HandleReturnValue::operator();
    };

  private:

    int backlog_;
}; // class Listen

} // namespace Sockets
} // namespace IPC

#endif // _IPC_SOCKETS_LISTEN_H_