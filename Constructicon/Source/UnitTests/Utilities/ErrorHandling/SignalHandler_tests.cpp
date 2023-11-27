#include "Utilities/ErrorHandling/SignalHandler.h"
#include "Utilities/Testing/CaptureCerr.h"

#include <csignal>
#include <gtest/gtest.h>
#include <iostream>

using Utilities::ErrorHandling::SignalHandler;
using Utilities::Testing::CaptureCerr;

// Unnamed or i.e. anonymous namespace, to make functions, objects, etc.
// accessible only within that file.
namespace
{

// https://en.cppreference.com/w/cpp/utility/program/sig_atomic_t
// std::sig_atomic_t an integer type which can be accessed as an atomic entity
// even in the presence of asynchronous interrupts made by signals.
// https://en.cppreference.com/w/cpp/language/cv
// Every access (read or write operation, member function call, etc.) made
// through a glvalue expression of volatile-qualified type is treated as a
// visible side-effect for purposes of optimization (that is, within a single
// thread of execution, volatile accesses cannot be optimized out or reordered
// with another visible side effect that's sequenced-before or sequenced-after
// the volatile access.
// This makes volatile objects suitable for communciation with a signal-handler,
// but not with another thread of execution.
volatile std::sig_atomic_t signal_count {0};

void dummy_signal_handler(int)
{
  signal_count = signal_count + 1;
  std::cerr << "Dummy signal handler was called with signal count: " <<
    signal_count << "\n";
}

class SignalHandlerTest : public ::testing::Test
{
  protected:

    void SetUp() override
    {
      signal_count = 0;
    }
};

} // namespace

namespace GoogleUnitTests
{
namespace Utilities
{
namespace ErrorHandling
{

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(SignalHandlerTests, HandleChildProcessExit)
{
  CaptureCerr capture_cerr {};  
  signal_count = 0;
  SignalHandler handler {};

  // 24.2.7 Miscellaneous Signals. SIGUSR1, SIGUSR2 signals are set aside for
  // you to use any way you want. The default action is to terminate the
  // process.

  auto old_handler = handler.handle_signal(SIGUSR1, dummy_signal_handler);

  // Send the signal.
  std::raise(SIGUSR1);

  // Ensure the dummy handler was called.
  EXPECT_EQ(signal_count, 1);
  EXPECT_EQ(
    capture_cerr.local_oss_.str(),
    "Dummy signal handler was called with signal count: 1\n");

  // Restore original handler if necessary.
  if (old_handler)
  {
    handler.handle_signal(SIGUSR1, old_handler);
  }
}

} // namespace Errorhandling
} // namespace Utilities
} // namespace GoogleUnitTests