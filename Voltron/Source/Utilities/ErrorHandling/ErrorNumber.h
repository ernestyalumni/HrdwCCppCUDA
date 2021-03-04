//------------------------------------------------------------------------------
/// \file ErrorNumber.h
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief  Wrappers and examples classes for errno and error handling.
/// \ref https://en.cppreference.com/w/cpp/error/errc
/// https://en.cppreference.com/w/cpp/error/errno_macros
/// \details Scoped enumeration (enum class) std::errc defines values of
/// portable error conditions corresponding to most of the POSIX error codes.
//------------------------------------------------------------------------------
#ifndef UTILITIES_ERROR_HANDLING_ERROR_NUMBER_H
#define UTILITIES_ERROR_HANDLING_ERROR_NUMBER_H

#include <cerrno> // E2BIG, EACCESS, ...
#include <cstring> // std::strerror
#include <string>
#include <system_error> // std::errc, std::make_error_code, std::error_code

namespace Utilities
{
namespace ErrorHandling
{

//------------------------------------------------------------------------------
/// \class ErrorCodeNumber
/// \name ErrorCodeNumber
/// \brief Wraps POSIX error codes macros.
/// \details Each of the macros expands to integer constant expressions of type
/// int, each with positive value, matching most POSIX error codes.
///
/// This enum class is defined here in the header since it needs to exposed to
/// other code when it's including this header .h file.
///
/// \ref https://stackoverflow.com/questions/1284529/enums-can-they-do-in-h-or-must-stay-in-cpp
//------------------------------------------------------------------------------
enum class ErrorCodeNumber: int
{
  // TODO: change variable names to follow this:
  // https://en.cppreference.com/w/cpp/header/system_error
  argument_list_too_long = E2BIG, // Argument list too long
  permission_denied = EACCES, // Permission defined
  address_in_use = EADDRINUSE, // Address in use
  address_not_available = EADDRNOTAVAIL, // Address not available
  address_family_not_supported = EAFNOSUPPORT, // Address family not supported
  eagain = EAGAIN, // Resource unavailable, try again
  ealready = EALREADY, // connection already in progress
  ebadf = EBADF, // Bad file descriptor
  ebadmsg = EBADMSG, // Bad message
  ebusy = EBUSY, // Device or resource busy
  ecanceled = ECANCELED, // Operation canceled.
  echild = ECHILD, // no child processes
  econnaborted = ECONNABORTED, // Connection aborted
  econnrefused = ECONNREFUSED, // Connection refused
  econnreset = ECONNRESET, // Connection reset
  edeadlk = EDEADLK, // Resource deadlock would occur
  edestaddrreq = EDESTADDRREQ, // Destination address required
  edom = EDOM, // Mathematics argument out of domain of function
  eexist = EEXIST, // File exists
  efault = EFAULT, // Bad address
  efbig = EFBIG, // File too large
  ehostunreach = EHOSTUNREACH, // Host is unreachable
  eidrm = EIDRM, // Identifier removed
  eilseq = EILSEQ, // Illegal byte sequence
  einprogress = EINPROGRESS, // Operation in progress
  eintr = EINTR, // Interrupted function
  invalid_argument = EINVAL, // Invalid argument
  eio = EIO, // I/O error
  eisconn = EISCONN, // Socket is connected
  eisdir = EISDIR, // Is a directory
  eloop = ELOOP, // Too many levels of symbolic links
  emfile = EMFILE, // File descriptor value too large
  emlink = EMLINK, // Too many links
  message_size_too_large = EMSGSIZE, // Message too large
  enametoolong = ENAMETOOLONG, // Filename too long
  network_down = ENETDOWN, // Network is down
  network_reset = ENETRESET, // Connection aborted by network
  network_unreachable = ENETUNREACH, // Network unreachable
  enfile = ENFILE, // Too many files open in system
  enobufs = ENOBUFS, // No buffer space available
  no_message_available = ENODATA, // No message is available on the STREAM head
  // read queue
  no_such_device = ENODEV, // No such device
  no_such_file_or_directory = ENOENT, // No such file or directory
  enoexec = ENOEXEC, // Executable file format error
  no_lock_available = ENOLCK, // No locks available
  no_link = ENOLINK, // Link has been severed
  enomem = ENOMEM, // Not enough space
  enomsg = ENOMSG, // No message of the desired type
  enoprotoopt = ENOPROTOOPT, // Protocol not available
  enospc = ENOSPC, // No space left on device
  enosr = ENOSR, // No STREAM resources
  enostr = ENOSTR, // Not a STREAM
  enosys = ENOSYS, // Function not supported
  enotconn = ENOTCONN, // The socket is not connected
  enotdir = ENOTDIR, // Not a directory
  enotempty = ENOTEMPTY, // Directory not empty
  enotrecoverable = ENOTRECOVERABLE, // State not recoverable
  enotsock = ENOTSOCK, // Not a socket
  enotsup = ENOTSUP, // Not supported
  enotty = ENOTTY, // Inappropriate I/O control operation
  enxio = ENXIO, // No such device or address
  eopnotsupp = EOPNOTSUPP, // Operation not supported on socket
  eoverflow = EOVERFLOW, // Value too large to be stored in data type
  eownerdead = EOWNERDEAD, // Previous owner died
  eperm = EPERM, // Operation not permitted
  epipe = EPIPE, // Broken pipe
  eproto = EPROTO, // Protocol error
  eprotonosupport = EPROTONOSUPPORT, // Protocol not supported
  eprototype = EPROTOTYPE, // Protocol wrong type for socket
  erange = ERANGE, //Result too large
  erofs = EROFS, // Read-only file system
  espipe = ESPIPE, // Invalid seek
  esrch = ESRCH, // No such process
  etime = ETIME, // Stream ioctl() timeout
  etimedout = ETIMEDOUT, // Connection timed out
  etxbsy = ETXTBSY, // Text file busy
  ewouldblock = EWOULDBLOCK, // Operation would block
  exdev = EXDEV // Cross-device link
};

//------------------------------------------------------------------------------
/// \brief Wrapper class for errno, or error numbers.
//------------------------------------------------------------------------------
class ErrorNumber
{
  public:

    //--------------------------------------------------------------------------
    /// \brief Default constructor.
    /// \details Provides important feature of using the preprocessor macro
    /// errno used for error indication. Library functions and implementation-
    /// dependent functions are allowed to write positive integers to errno
    /// whether or not an error occurred.
    //--------------------------------------------------------------------------
    ErrorNumber();

    explicit ErrorNumber(const int error_number);

    explicit ErrorNumber(const std::error_code& error_code);

    explicit ErrorNumber(const std::system_error& err);

    /// \ref https://en.cppreference.com/w/cpp/error/error_code/error_code
    ErrorNumber(const int error_number, const std::error_category& ecat);

    // Accessors
    /// \ref https://stackoverflow.com/questions/44935159/why-must-accessor-functions-be-const-where-is-the-vulnerability

    /// \ref https://en.cppreference.com/w/cpp/string/byte/strerror
    std::string as_string() const
    {
      return std::string{std::strerror(error_number_)};
    }

    int error_number() const
    {
      return error_number_;
    }

    std::error_code error_code() const
    {
      return error_code_;
    }

    std::error_condition error_condition() const
    {
      return error_condition_;
    }

    static const int to_error_code_value(ErrorCodeNumber error);

    static const ErrorCodeNumber from_error_number(const int error_number)
    {
      return static_cast<ErrorCodeNumber>(error_number);
    }

  private:

    int error_number_;

    std::error_code error_code_;

    /// \ref https://en.cppreference.com/w/cpp/error/error_condition/error_condition
    std::error_condition error_condition_;
};

} // namespace ErrorHandling
} // namespace Utilities

#endif // UTILITIES_ERROR_HANDLING_ERROR_NUMBER_H
