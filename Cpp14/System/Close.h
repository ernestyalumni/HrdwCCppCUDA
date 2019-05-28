//------------------------------------------------------------------------------
/// \file Close.h
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief ::close as a C++ functor 
/// \ref http://man7.org/linux/man-pages/man2/close.2.html
/// \details Close a file descriptor.
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
#ifndef _SYSTEM_CLOSE_H_
#define _SYSTEM_CLOSE_H_

#include "Utilities/ErrorHandling.h" // HandleReturnValue

namespace System
{

//------------------------------------------------------------------------------
/// \class Close
/// \brief ::close() system call wrapped in a C++ functor.
/// \details 
///
/// #include <unistd.h>
/// int ::close(int fd)
/// 
/// ::close() closes a fd, so "it no longer refers to any file and may be
/// reused."
/// - Any record locks (see ::fcntl) held on file it was associated with, and
///   owned by process are removed (regardless of fd that was used to obtain the
///   lock)
/// - If fd is last fd referring to underlying open description (see ::open),
///   resources assocated with open file description are freed.
/// - If fd was last reference to a file which has been removed using ::unlink,
///   the file is deleted.
//------------------------------------------------------------------------------
class Close
{
  public:

    Close();

    //--------------------------------------------------------------------------
    /// \brief Calls ::close
    /// \details On success, ::close returns 0. On error, -1 is returned and
    /// errno set appropriately.
    /// \url http://man7.org/linux/man-pages/man2/close.2.html
    //--------------------------------------------------------------------------
    void operator()(const int fd);    

  protected:

    //--------------------------------------------------------------------------
    /// \class HandleClose
    /// \details A careful programmer will check return value of ::close(),
    /// since it's quite possible that errors on a previous ::write operation
    /// are reported only on the final ::close() that releases the open fd.
    /// Failing to check the return value when closing a file may lead to silent
    /// loss of dta. This can especially be observed with NFS and with disk
    /// quota.
    /// Note, however, that a failure return should be used only for diagnostic
    /// purposes (i.e., warning to application that there may still be I/O
    /// pending or there may have been failed I/O) or remedial purposes (e.g.,
    /// writing file once more or creating a backup).
    /// Retrying ::close() after failure return is wrong thing to do, since this
    /// may cause a reused fd from another thread to be closed.
    //--------------------------------------------------------------------------
    class HandleClose : public Utilities::ErrorHandling::HandleReturnValue
    {
      public:

        HandleClose();

        void operator()(const int result);

      private:

        using HandleReturnValue::error_number;
        using HandleReturnValue::get_error_number;
    };

  private:

    int fd_;
};


} // namespace System

#endif // _SYSTEM_FORK_H_