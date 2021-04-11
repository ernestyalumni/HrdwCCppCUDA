//------------------------------------------------------------------------------
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
//------------------------------------------------------------------------------
#include "FileDescriptor.h"

#include "Utilities/ErrorHandling/HandleClose.h"

#include <unistd.h> // ::close

using Utilities::ErrorHandling::HandleClose;

namespace FileIO
{

FileDescriptor::FileDescriptor(const int fd):
  fd_{fd}
{}

// Move ctor.
FileDescriptor::FileDescriptor(FileDescriptor&& other):
  fd_{other.fd()}
{
  // So not to invoke the ::close in dtor of other.
  other.fd_ = -1;
}

// Move assignment.
FileDescriptor& FileDescriptor::operator=(FileDescriptor&& other)
{
  fd_ = other.fd_;

  other.fd_ = -1;

  return *this;
}

FileDescriptor::~FileDescriptor()
{
  if (fd_ > -1)
  {
    HandleClose close_handler;

    const int close_return {::close(fd_)};
    close_handler(close_return);
  }
}

} // namespace FileIO
