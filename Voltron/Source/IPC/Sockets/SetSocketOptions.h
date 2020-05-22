//------------------------------------------------------------------------------
/// \file SetSocketOptions.h
/// \author Ernest Yeung
/// \brief ::setsockopt wrapper.
/// \ref https://linux.die.net/man/2/setsockopt
//------------------------------------------------------------------------------
#ifndef IPC_SOCKETS_SET_SOCKET_OPTIONS_H
#define IPC_SOCKETS_SET_SOCKET_OPTIONS_H

#include "IPC/Sockets/ParameterFamilies.h"
#include "IPC/Sockets/Socket.h"
#include "Utilities/ErrorHandling/ErrorNumber.h"

#include <optional>

namespace IPC
{
namespace Sockets
{

//------------------------------------------------------------------------------
/// \brief Manipulates options for the socket referred to by fd sockfd.
/// \ref https://linux.die.net/man/2/setsockopt
/// \details int setsockopt(int sockfd, int level, int optname,
/// const void *optval, socklen_t optlen);
/// When manipulating socket options, level which option resides and name of
/// option must be specified.
/// - level SOL_SOCKET - to manipulate options at sockets API level.
/// To manipulate options at any other level, protocol number of appropriate
/// protocol controlling the option is supplied.
/// e.g. to indicate that option is to be interpreted by TCP protocol, level
/// should be set to protocol number of TCP.
/// arguments optval, optlen used to access option values.
/// int argument for optval, nonzero to enable boolean option, 0 if option is
/// disabled.
//------------------------------------------------------------------------------

class SetSocketOptions
{
	public:

		SetSocketOptions(int level, int option_name, int optional_value);

		std::optional<Utilities::ErrorHandling::ErrorNumber> operator()(
			Socket& socket);

	private:

    //--------------------------------------------------------------------------
    /// \brief Return 0 on on success. On error, -1 is returned and errno is set
    /// appropriately.
    /// \details 
    /// EBADF Argument sockfd isn't a valid fd.
    /// EFAULT optval argument not valid part of process address space
    /// EINVAL optlen invalid in setsockopt(). 
    /// ENOBUFS Insufficient resources were available in system to perform
    /// operation.
    /// ENOTSOCK fd sockfd is a file, not a socket.
    //--------------------------------------------------------------------------
    class HandleSetSocketOptions : public
    	Utilities::ErrorHandling::HandleReturnValuePassively
    {
      public:

      	using HandleReturnValuePassively::HandleReturnValuePassively;
    };		

    // Example values include SOL_SOCKET, for the level which option resides in.
    int level_;
    // Example values include SO_REUSEADDR or SO_REUSEADDR | SO_REUSEPORT
    int option_name_;
    // Example value include 1 to indicate boolean option enabled.
    int optional_value_;
};

class SetReusableSocketAddress : public SetSocketOptions
{
	public:

		SetReusableSocketAddress();

	protected:

		using SetSocketOptions::SetSocketOptions;
};

class SetReusableAddressAndPort : public SetSocketOptions
{
	public:

		SetReusableAddressAndPort();

	protected:

		using SetSocketOptions::SetSocketOptions;
};

} // namespace Sockets
} // namespace IPC

#endif // IPC_SOCKETS_SET_SOCKET_OPTIONS_H