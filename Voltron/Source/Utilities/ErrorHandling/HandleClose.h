//------------------------------------------------------------------------------
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
//------------------------------------------------------------------------------
#ifndef UTILITIES_ERROR_HANDLING_HANDLE_CLOSE_H
#define UTILITIES_ERROR_HANDLING_HANDLE_CLOSE_H

#include "HandleReturnValue.h"

namespace Utilities
{
namespace ErrorHandling
{

//------------------------------------------------------------------------------
/// \ref http://man7.org/linux/man-pages/man2/close.2.html
/// \details On error, -1 is returned, and errno is set appropriately.
/// 0 indicates success.
/// EBADF fd isn't valid open fd
/// EINTR close() call interrupted by signal
/// EIO I/O error occurred.
/// ENOSPC, EDQUOT On NFS, errors aren't normally reported against first write
/// which exceeds available storage space, but instead against subsequent write,
/// fsync, or close().
///
/// ::close() should not be retried after an error, since this may cause a
/// reused fd from another thread to be closed. This can occur because Linux
/// kernel always releases the fd early in close operation, freeing it for
/// reuse; steps that may return an error, such as flushing data to the
/// filesystem or device, occur only later in close operation.
///
/// Thus, throw a system error.
//------------------------------------------------------------------------------
class HandleClose : public ThrowSystemErrorOnNegativeReturnValue
{
  public:

    HandleClose():
      ThrowSystemErrorOnNegativeReturnValue{"Error for ::close()"}
    {}
};


} // namespace ErrorHandling
} // namespace Utilities

#endif // UTILITIES_ERROR_HANDLING_HANDLE_CLOSE_H
