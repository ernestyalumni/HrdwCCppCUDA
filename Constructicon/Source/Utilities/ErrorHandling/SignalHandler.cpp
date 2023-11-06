#include "SignalHandler.h"
#include "Utilities/ErrorHandling/GetErrorNumber.h"

#include <iostream>
#include <signal.h>

namespace Utilities
{
namespace ErrorHandling
{

sighandler_t* SignalHandler::handle_signal(
  const int signal_number,
  sighandler_t* handler)
{
  struct sigaction action {};
  struct sigaction old_action {};

  action.sa_handler = *handler;

  // https://www.man7.org/linux/man-pages/man3/sigemptyset.3p.html
  // int sigemptyset(sigset_t* set);
  // Initializes signal set pointed to by set, such that all signals defined in
  // POSIX.1-2008 are excluded.
  // Block signals of type being handled.
  sigemptyset(&action.sa_mask);

  // Provide behavior compatible with BSD signal semantics by making certain
  // certain system calls restartable across signals. This flag is meaningful
  // only when establishing a signal handler.
  // Restart syscalls if possible.
  action.sa_flags = SA_RESTART;

  // https://www.man7.org/linux/man-pages/man2/sigaction.2.html
  // int sigaction(int signum, const struct sigaction * _Nullable restrict act,
  // struct sigaction *_Nullable restrict oldact);
  // sigaction() system call used to change action taken by process on receipt
  // of a specific signal.
  // signum specifies the signal and can be any valid signal except SIGKILL and
  // SIGSTOP.
  // If act is non-NULL, the new action for signal signum is installed from act.
  // If oldact is non-NULL, the previous action is saved in oldact.
  if (sigaction(signal_number, &action, &old_action) < 0)
  {
    GetErrorNumber get_error_number {};
    std::cerr << get_error_number.as_string() << " Signal error\n";
  }
  return (old_action.sa_handler);
}

} // namespace ErrorHandling
} // namespace Utilities