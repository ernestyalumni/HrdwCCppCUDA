//------------------------------------------------------------------------------
/// \file Write.cpp
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief Example based on Love's Linux System Programming for ::select.
/// \ref https://github.com/raoulmillais/linux-system-programming/blob/master/\
/// src/select-example.c
/// \details select for synchronous I/O multiplexing.
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
/// g++ -std=c++17 -I ../ Write.cpp ../Utilities/Errno.cpp
///   ../Utilities/ErrorHandling.cpp Open_main.cpp -o Fork_main
//------------------------------------------------------------------------------
#include "Utilities/ErrorHandling.h" // HandleReturnValue

#include <iostream>
#include <unistd.h> // STDIN_FILENO

// According to POSIX.1-2001, POSIX.1-2008
#include <sys/select.h>

#include <sys/types.h> // time_t, suseconds_t

// According to earlier standards
//#include <sys/time.h>
//#include <sys/types.h>
//#include <unistd.h>

// wait up to 5 seconds
constexpr long timeout {5};

using Utilities::ErrorHandling::HandleReturnValue;

class HandleSelect : public HandleReturnValue
{
  public:

    HandleSelect();

    void operator()(const int result)

  private:

    using HandleReturnValue::error_number;
};

int main()
{

  // Concerning types involved, classical situation is that the 2 fields of a
  // ::timeval structure are typed as long.
  // POSIX.1 situation
  // struct timeval
  // {
  //  time_t tv_sec; // seconds
  //  suseconds_t tv_usec; // microseconds
  //  };
  // select() may update timeout argument to indicate how much time was left.
  // The timeout argument specifies interval that select() should block waiting
  // for waiting for a fd to become ready. The call will block until either:
  // * a fd becomes ready
  // * call is interrupted by signal handler; or  
  // * timeout expires.
  {
    std::cout << " sizeof(time_t) : " << sizeof(time_t) << '\n';
    std::cout << " sizeof(suseconds_t) : " << sizeof(suseconds_t) << '\n';
    std::cout << " sizeof(timeval) : " << sizeof(timeval) << '\n';
  }

  ::timeval tv;

  // fd_set is a fixed size behavior
  // The fds listed in readfds will be watched to see if characters become
  // available for reading (more precisely, to see if read won't block;
  // in particular, a fd is also ready on end-of-file).
  ::fd_set read_fds;

  // Wait on stdin for input.

  // void FD_ZERO(fd_set* set)
  // FD_ZERO, FD_SET are 2 of 4 macros to manipulate sets.
  // FD_ZERO() clears a set.
  FD_ZERO(&read_fds);

  // void FD_SET(int fd, fd_set* set)
  // FD_SET() add and remove given fd from a set.
  //
  // cf. http://man7.org/linux/man-pages/man3/stdin.3.html
  // On program startup, the integer fds associated with streams stdin, stdout,
  // and stderr are 0, 1, 2, respectively.
  FD_SET(STDIN_FILENO, &read_fds);  

  // Wait up to 5 seconds
  tv.tv_sec = timeout;
  tv.tv_usec = 0;

  std::cout << " tv.tv_sec : " << tv.tv_sec << '\n';
  std::cout << " tv.tv_usec : " << tv.tv_usec << '\n';

  //----------------------------------------------------------------------------
  /// \url http://man7.org/linux/man-pages/man2/select.2.html
  /// \details
  /// int select(int nfds, fd_set* readfds, fd_set* writefds,
  ///   fd_set* exceptfds, struct timeval* timeout)
  /// 3 independent sets of fds are watched. 
  /// On exit, each of the fd sets is modified in place to indicate which fds
  /// actually changed status. (thus, if using select() within a loop, the sets
  /// must be reinitialized before each call)
  ///
  /// Each of the 3 fd sets may be specified as nullptr if no fds are to be
  /// watched for corresponding class of events.
  /// nfds should be set to the highest-numbered fd in any of the 3 sets, plus
  /// 1. The indicated fds in each set are checked, up to this limit.
  ///
  /// timeout argument specifies interval that select() should block waiting for
  /// a fd to become ready. CAll will block until either:
  /// - fd becomes ready
  /// - call is interrupted by signal handler; or
  /// - timeout expires 
  //----------------------------------------------------------------------------

  int return_value {
    ::select(
      STDIN_FILENO + 1,
      &read_fds,
      nullptr,
      nullptr,
      &tv)};
}
