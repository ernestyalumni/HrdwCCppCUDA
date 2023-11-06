#include "signal_child_handler.h"

#include <sys/wait.h>

namespace Utilities
{
namespace ErrorHandling
{

void signal_child_handler(const int)
{
  // busy wait.
  // https://linux.die.net/man/2/waitpid
  // pid_t waitpid(pid_t pid, int* status, int options);
  // Wait for state changes in a child of calling process, and obtain
  // information about child whose state has changed. A state change is
  // considered to be: the child terminated; child was stopped by a signal; or
  // the child was resumed by a signal.
  // pid value can be
  // < -1 meaning wait for any child process whose process group ID is equal to
  // the absolute value of pid.
  // -1 meaning wait for any child process.
  // WNOHANG - return immediately if no child has exited.
  while (::waitpid(-1, 0, WNOHANG) > 0)
  {
    ;
  }
  return;
}

} // namespace ErrorHandling
} // namespace Utilities