//------------------------------------------------------------------------------
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \ref https://linux.die.net/man/2/socket
//------------------------------------------------------------------------------
#ifndef IPC_SOCKETS_CREATE_SOCKET_H
#define IPC_SOCKETS_CREATE_SOCKET_H

#include "IPC/Sockets/ParameterFamilies.h"
#include "Utilities/ErrorHandling/ErrorNumber.h"
#include "Utilities/ErrorHandling/HandleReturnValue.h"

#include <optional>
#include <set>

namespace IPC
{
namespace Sockets
{

class SocketFd
{
  public:

    SocketFd() = delete;

    SocketFd(
      const int domain,
      const int type,
      const int protocol,
      const int fd);

    // Copy Constructor.
    //SocketFd(const SocketFd&) = delete;

    // Copy Assignment.
    //SocketFd& operator=(const SocketFd&) = delete;

    // Move constructor.
    //SocketFd

  private:

    int domain_;
    int type_;
    int protocol_;
    int fd_;
};

//------------------------------------------------------------------------------
/// \brief Creates an endpoint for communication
//------------------------------------------------------------------------------
class CreateSocket
{
  public:

    using OptionalSocketFd = std::optional<SocketFd>;

    //--------------------------------------------------------------------------
    /// \fn CreateSocket
    /// \brief Constructor matching ::socket function signature.
    /// \url http://man7.org/linux/man-pages/man2/socket.2.html
    /// \details The protocol specifies a particular protocol to be used with
    /// the socket. Normally, only a single protocol exists to support a
    /// particular socket type within a given protocol family, in which case
    /// protocol can be specified as 0.
    //--------------------------------------------------------------------------
    CreateSocket(
      const int domain,
      const int type_value,
      const int protocol = 0);

    CreateSocket(
      const Domain domain,
      const Type type_enumeration,
      const int protocol = 0);

    void add_behavior_modification(const BehaviorModifyingValue value);

    // TODO: change return type to SocketFd
    //--------------------------------------------------------------------------
    /// \details If it's the case that we want the user to be able to try again
    /// in creating a socket if it fails beforehand, then we do not want to
    /// throw here and stop the overall program; instead we'll handle exceptions
    /// to running ::socket() by an optional value.
    //--------------------------------------------------------------------------    
    OptionalSocketFd operator()();

    static const int to_domain_value(const Domain domain);

    static const Domain from_domain_value(const int domain_value)
    {
      return static_cast<Domain>(domain_value);
    }

    static const int to_type_value(const Type type_enumeration);

    static const Type from_type_value(const int type_value)
    {
      return static_cast<Type>(type_value);
    }

    static const int to_behavior_modifying_underlying_value(
      const BehaviorModifyingValue enumeration);

    static const BehaviorModifyingValue
      from_behavior_modifying_underlying_value(const int value)
    {
      return static_cast<BehaviorModifyingValue>(value);
    }

    int type_value() const
    {
      return type_value_;
    }

    Utilities::ErrorHandling::ErrorNumber get_error_number() const
    {
      return socket_handle_.error_number();
    }

  private:

    int domain_;
    int type_value_;
    Type type_;
    std::set<BehaviorModifyingValue> modifying_types_;
    int protocol_;    
    Utilities::ErrorHandling::HandleReturnValueWithOptional socket_handle_;
};

} // namespace Sockets
} // namespace IPC

#endif // IPC_SOCKETS_CREATE_SOCKET_H
