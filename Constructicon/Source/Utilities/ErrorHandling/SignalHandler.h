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
//------------------------------------------------------------------------------
class SignalHandler
{
  public:

    static sighandler_t* handle_signal(
      const int signal_number,
      sighandler_t* handler);
};

} // namespace ErrorHandling
} // namespace Utilities

#endif // UTILITIES_ERROR_HANDLING_SIGNAL_HANDLER_H
