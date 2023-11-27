#ifndef UTILITIES_ERROR_HANDLING_SIGNAL_HANDLER_H
#define UTILITIES_ERROR_HANDLING_SIGNAL_HANDLER_H

#include <signal.h>

namespace Utilities
{
namespace ErrorHandling
{

//------------------------------------------------------------------------------
/// pp. 938 Ch. 12 Concurrent Programming, 3rd. Ed. Computer Systems: A
/// Programmer's Perspective. Bryant and O'Hallaron.
/// http://csapp.cs.cmu.edu/3e/code.html
/// http://csapp.cs.cmu.edu/3e/ics3/code/include/csapp.h
/// This is a replacement for the Signal function in csapp.c, which is a wrapper
/// for a Unix signal function.
//------------------------------------------------------------------------------
class SignalHandler
{
  public:

    SignalHandler():
      old_action_{}
    {}

    //--------------------------------------------------------------------------
    /// https://www.gnu.org/software/libc/manual/html_node/Basic-Signal-Handling.html
    /// 24.3.1. Basic Signal Handling
    /// sighandler_t - data type of signal handler functions. Signal handlers
    /// take 1 integer argument specifying the signal number, and have return
    /// type void. 
    //--------------------------------------------------------------------------
    sighandler_t handle_signal(
      const int signal_number,
      sighandler_t handler);

    struct sigaction old_action_;
};

} // namespace ErrorHandling
} // namespace Utilities

#endif // UTILITIES_ERROR_HANDLING_SIGNAL_HANDLER_H
