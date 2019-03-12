//------------------------------------------------------------------------------
/// \file Write.h
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief ::write as a C++ functor 
/// \ref http://man7.org/linux/man-pages/man2/write.2.html
/// \details Write to a fd.
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
///  g++ -std=c++17 -I ../ Close.cpp ../Utilities/Errno.cpp
///   ../Utilities/ErrorHandling.cpp Close_main.cpp -o Close_main
//------------------------------------------------------------------------------
#ifndef _SYSTEM_WRITE_H_
#define _SYSTEM_WRITE_H_

#include "Utilities/ErrorHandling.h" // HandleReturnValue

namespace System
{

//------------------------------------------------------------------------------
/// \class Write
/// \brief ::write() system call wrapped in a C++ functor.
/// \details
///
/// #include <unistd.h>
/// ssize_t ::write(int fd, const void* buf, size_t count);
///
/// ::write() writes up to count bytes from the buffer starting at buf to the
/// file referred to by the fd.
///
/// Number of bytes written may be less than count if, for example, there's
/// insufficient space on underlying physical medium, or RLIMIT_FSIZE resource
/// limit is encountered (see ::setrlimit), or
/// - call interrupted by signal handler after having written less than count
///   bytes
///
/// For seekable file (i.e., one to which ::lseek may be applied, e.g., a
/// regular file), writing takes place at file offset, and file offset is
/// incremented by number of bytes actually written.
//------------------------------------------------------------------------------
class Write
{
  public:

    Write();

    //--------------------------------------------------------------------------
    /// \url http://man7.org/linux/man-pages/man2/write.2.html
    //--------------------------------------------------------------------------
    ssize_t operator()(const int fd, const void *buf, const size_t count);

    ssize_t last_number_of_bytes_to_write() const
    {
      return last_number_of_bytes_to_write_;
    }

  protected:

    //--------------------------------------------------------------------------
    /// \class HandleWrite
    /// \brief On success, the number of bytes written is returned. On error, -1
    /// is returned, and errno set to indicate cause of error.
    ///
    /// Note that successful write() may transfer fewer than count bytes.
    //--------------------------------------------------------------------------
    class HandleWrite : public Utilities::ErrorHandling::HandleReturnValue
    {
      public:

        HandleWrite();

        void operator()(const ssize_t result);

      private:

        using HandleReturnValue::operator();
    };

  private:

    ssize_t last_number_of_bytes_to_write_;
}; // class Write

}; // namespace System

#endif // _SYSTEM_WRITE_H_