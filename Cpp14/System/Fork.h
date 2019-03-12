//------------------------------------------------------------------------------
/// \file Fork.h
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief ::fork as a C++ functor 
/// \ref http://man7.org/linux/man-pages/man2/fork.2.html
/// \details Open and possibly create a file.
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
///  g++ -std=c++17 -I ../ Fork.cpp ../Utilities/Errno.cpp
///   ../Utilities/ErrorHandling.cpp Fork_main.cpp -o Fork_main
//------------------------------------------------------------------------------
#ifndef _SYSTEM_FORK_H_
#define _SYSTEM_FORK_H_

#include "Utilities/ErrorHandling.h" // HandleReturnValue

namespace System
{

//------------------------------------------------------------------------------
/// \class Fork
/// \brief ::fork() system call wrapped in a C++ functor.
/// \details 
/// 
/// pid_t ::fork(void);
//------------------------------------------------------------------------------
class Fork
{
  public:

    Fork();

    //--------------------------------------------------------------------------
    /// \brief Calls ::fork
    /// \details On success, PID of child process is returned in the parent, and
    /// 0 is returned in the child. On failure, -1 is returned in the parent, no
    /// child process created, and errno set appropriately.
    /// \url http://man7.org/linux/man-pages/man2/fork.2.html
    //--------------------------------------------------------------------------
    pid_t operator()();    

    pid_t process_id() const
    {
      return process_id_;
    }

  protected:

    //--------------------------------------------------------------------------
    /// \class HandleFork
    /// 
    //--------------------------------------------------------------------------
    class HandleFork : public Utilities::ErrorHandling::HandleReturnValue
    {
      public:

        HandleFork();

        void operator()(const pid_t result);

      private:

        using HandleReturnValue::operator();
    };

  private:

    pid_t process_id_;
};


}; // namespace System

#endif // _SYSTEM_FORK_H_