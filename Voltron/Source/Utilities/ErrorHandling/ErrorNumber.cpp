//------------------------------------------------------------------------------
/// \file ErrorNumber.cpp
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief  Wrappers and examples classes for errno and error handling.
/// \ref https://en.cppreference.com/w/cpp/error/errc
/// https://en.cppreference.com/w/cpp/error/errno_macros
/// \details Scoped enumeration (enum class) std::errc defines values of
/// portable error conditions corresponding to POSIX error codes.
//------------------------------------------------------------------------------
#include "ErrorNumber.h"

#include <system_error> // std::errc, std::error_category, std::system_category

namespace Utilities
{
namespace ErrorHandling
{

/*
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
*/

ErrorNumber::ErrorNumber():
  error_number_{errno},
  error_code_{errno, std::system_category()},
  error_condition_{static_cast<std::errc>(errno)}
{}

ErrorNumber::ErrorNumber(const int error_number):
  error_number_{error_number},
  error_code_{error_number, std::system_category()},
  error_condition_{static_cast<std::errc>(error_number)}
{}

ErrorNumber::ErrorNumber(const std::error_code& error_code):
  error_number_{error_code.value()},
  error_code_{error_code},
  error_condition_{error_code.value(), error_code.category()}
{}

ErrorNumber::ErrorNumber(
  const int error_number,
  const std::error_category& ecat
  ):
  error_number_{error_number},
  error_code_{error_number, ecat},
  error_condition_{error_number, ecat}
{}

} // namespace ErrorHandling
} // namespace Utilities
