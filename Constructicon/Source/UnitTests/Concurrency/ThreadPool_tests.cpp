#include "Concurrency/ThreadPool.h"

#include "gtest/gtest.h"

#include <thread>

using Concurrency::ThreadPool;

namespace GoogleUnitTests
{
namespace Concurrency
{

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(ThreadPoolTests, DefaultConstructs)
{
  ThreadPool tp {};

  EXPECT_EQ(tp.get_tasks_queued(), 0);
  EXPECT_EQ(tp.get_tasks_running(), 0);
  EXPECT_EQ(tp.get_tasks_total(), 0);
  EXPECT_EQ(tp.get_thread_count(), std::thread::hardware_concurrency());
}

} // namespace Concurrency
} // namespace GoogleUnitTests