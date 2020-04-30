//------------------------------------------------------------------------------
/// \file ErrorNumber.h
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief  Wrappers and examples classes for errno and error handling.
/// \ref https://en.cppreference.com/w/cpp/error/errc
/// https://en.cppreference.com/w/cpp/error/errno_macros
/// \details Scoped enumeration (enum class) std::errc defines values of
/// portable error conditions corresponding to POSIX error codes.
//------------------------------------------------------------------------------
#ifndef UTILITIES_ERROR_HANDLING_ERROR_NUMBER_H
#define UTILITIES_ERROR_HANDLING_ERRNO_NUMBER_H

#include <cerrno> // E2BIG, EACCESS, ...
#include <cstring> // std::strerror
#include <string>
#include <system_error> // std::errc, std::make_error_code, std::error_code

namespace Utilities
{
namespace ErrorHandling
{

//------------------------------------------------------------------------------
/// \class ErrorNumbers
/// \name ErrorNumbers
/// \brief Wraps POSIX error codes macros.
/// \details Each of the macros expands to integer constant expressions of type
/// int, each with positive value, matching most POSIX error codes.
//------------------------------------------------------------------------------
enum class ErrorNumbers: int
{
  e2big = E2BIG, // Argument list too long
  eacces = EACCES, // Permission defined
  eaddrinuse = EADDRINUSE, // Address in use
  eaddrnotavail = EADDRNOTAVAIL, // Address not available
  eafnosupport = EAFNOSUPPORT, // Address family not supported
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
  einval = EINVAL, // Invalid argument
  eio = EIO, // I/O error
  eisconn = EISCONN, // Socket is connected
  eisdir = EISDIR, // Is a directory
  eloop = ELOOP, // Too many levels of symbolic links
  emfile = EMFILE, // File descriptor value too large
  emlink = EMLINK, // Too many links
  emsgsize = EMSGSIZE, // Message too large
  enametoolong = ENAMETOOLONG, // Filename too long
  enetdown = ENETDOWN, // Network is down
  enetreset = ENETRESET, // Connection aborted by network
  enetunreach = ENETUNREACH, // Network unreachable
  enfile = ENFILE, // Too many files open in system
  enobufs = ENOBUFS, // No buffer space available
  enodata = ENODATA, // No message is available on the STREAM head read queue
  enodev = ENODEV, // No such device
  enoent = ENOENT, // No such file or directory
  enoexec = ENOEXEC, // Executable file format error
  enolck = ENOLCK, // No locks available
  enolink = ENOLINK, // Link has been severed
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

    ErrorNumber();

    explicit ErrorNumber(const int error_number);

    explicit ErrorNumber(const std::error_code& error_code);

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

  private:

    int error_number_;

    std::error_code error_code_;

    /// \ref https://en.cppreference.com/w/cpp/error/error_condition/error_condition
    std::error_condition error_condition_;
};

} // namespace ErrorHandling
} // namespace Utilities

#endif // UTILITIES_ERROR_HANDLING_ERRNO_H
