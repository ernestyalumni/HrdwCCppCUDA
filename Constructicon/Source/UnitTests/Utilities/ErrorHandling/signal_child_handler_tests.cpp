#include "Utilities/ErrorHandling/signal_child_handler.h"
#include <chrono>
#include <gtest/gtest.h>
#include <sys/wait.h>
#include <thread> // std::this_thread::sleep_for
#include <unistd.h> // pause

using Utilities::ErrorHandling::signal_child_handler;

namespace GoogleUnitTests
{
namespace Utilities
{
namespace ErrorHandling
{

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(SignalChildHandlerTests, HandleChildProcessExit)
{
  // https://linux.die.net/man/2/fork
  // fork() creates a new process, child, by duplicating the calling process,
  // parent.
  // - child has its own unique process ID and this PID doesn't match ID of any
  // existing process group.
  // - child doesn't inherit its parent's memory locks.
  // child doesn't inherit record locks from its parent.
  const pid_t pid {::fork()};

  // On success, PID of child process is returned and 0 is returned in child.
  // On failure -1 is returned in parent, no child process is created, errno
  // set.
  ASSERT_NE(pid, -1);

  if (pid == 0)
  {
    // Child process.
    // Simulate long-running operation or suspended state.
    // Wait for a signal indefinitely.
    // pause() function shall suspend calling thread until delivery of a signal
    // whose action is either to execute a signal-catching function or to
    // terminate the process.
    ::pause();
    // Exit when a signal is received.
    // https://linux.die.net/man/3/exit
    // exit() causes normal process termination.
    ::exit(0);
  }
  // if (pid > 0)
  else
  {
    // Parent process.
    // Give child process a chance to start and get into 
    // Linux version: sleep(1);
    // https://en.cppreference.com/w/cpp/thread/sleep_for
    std::this_thread::sleep_for(std::chrono::seconds{1});

    // Send a termination signal to the child process
    ::kill(pid, SIGTERM);

    // https://www.ibm.com/docs/en/ztpf/2019?topic=signals-sigchld-signal
    // When child process ends, system sends a SIGCHILD signal to parent process
    // to indicate child process has ended.
    signal_child_handler(SIGCHLD);

    int status {-2};
    const pid_t waited_pid {::waitpid(pid, &status, 0)};
    // Ensure we're reaping the correct child.
    ASSERT_EQ(waited_pid, pid);

    // Check if child was terminated by SIGTERM signal.
    // https://www.ibm.com/docs/en/ztpf/1.1.0.15?topic=zca-wifsignaledquery-status-see-if-child-process-ended-abnormally
    // Query status to see if child process ended abnormally.
    // https://people.cs.rutgers.edu/~pxk/416/notes/c-tutorials/wait.html
    // True if process exited because of receipt of some signal.
    ASSERT_TRUE(WIFSIGNALED(status));
    // https://www.gnu.org/software/libc/manual/html_node/Process-Completion-Status.html
    // Returns signal number of signal that terminated the child process.
    ASSERT_EQ(WTERMSIG(status), SIGTERM);
  }
}

} // namespace Errorhandling
} // namespace Utilities
} // namespace GoogleUnitTests