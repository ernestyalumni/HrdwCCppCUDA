//------------------------------------------------------------------------------
/// \file ErrorHandling.h
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief Error handling C++ functors to check POSIX Linux system call
/// results.
/// \ref
/// \details
//------------------------------------------------------------------------------
#ifndef UTILITIES_ERROR_HANDLING_ERROR_HANDLING_H
#define UTILITIES_ERROR_HANDLING_ERROR_HANDLING_H

#include "ErrorNumber.h" // ErrorNumber

#include <optional>
#include <string> // std::string

namespace Utilities
{
namespace ErrorHandling
{

//------------------------------------------------------------------------------
/// \brief Virtual base class for C++ functor for checking the result of a
/// POSIX Linux system calls, usually when creating a new file descriptor.
//------------------------------------------------------------------------------
class HandleReturnValue
{
  public:

    HandleReturnValue();

    HandleReturnValue(const int error_number);

    virtual void operator()(
      const int result,
      const std::string& custom_error_string);

    //virtual void operator()(const int result);

    ErrorNumber error_number() const
    {
      return error_number_;
    }

  protected:

    void get_error_number();

  private:

    ErrorNumber error_number_;
};

class HandleReturnValuePassively
{
  public:

    using OptionalErrorNumber = std::optional<ErrorNumber>;

    HandleReturnValuePassively();

    //--------------------------------------------------------------------------
    /// \details If the return_value is less than 0, the error number is
    /// obtained (since the errno would have been set), and a non-empty optional
    /// is returned. Otherwise, an empty optional is returned.
    //--------------------------------------------------------------------------
    virtual std::optional<ErrorNumber> operator()(const int return_value);

    ErrorNumber error_number() const
    {
      return error_number_;
    }

  protected:

    void get_error_number();

  private:

    ErrorNumber error_number_;
};


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
//------------------------------------------------------------------------------
class HandleClose : public HandleReturnValue
{
  public:

    HandleClose() = default;

    std::optional<ErrorNumber> operator()(const int return_value);

  protected:

    using HandleReturnValue::get_error_number;

  private:

    using HandleReturnValue::HandleReturnValue;
    using HandleReturnValue::operator();
};

//------------------------------------------------------------------------------
/// \ref http://man7.org/linux/man-pages/man2/read.2.html
/// \details On error, -1 is returned, and errno is set appropriately.
/// In this case, it's left unspecified whether the file position (if any)
/// changes.
//------------------------------------------------------------------------------
class HandleRead : public HandleReturnValue
{
  public:

    HandleRead() = default;

    //--------------------------------------------------------------------------
    /// \ref http://man7.org/linux/man-pages/man2/read.2.html
    /// \details On success, number of bytes read is returned (0 indicates end
    /// of file), and file position is advanced by this number. It's not an
    /// error if this number is smaller than number of bytes requested.
    /// On error, -1 returned and errno set appropriately. In this case, it's
    /// left unspecified whether file position (if any) changes.
    //--------------------------------------------------------------------------
    virtual void operator()(const ssize_t number_of_bytes);
};

} // namespace ErrorHandling
} // namespace Utilities

#endif // UTILITIES_ERROR_HANDLING_ERROR_HANDLING_H
