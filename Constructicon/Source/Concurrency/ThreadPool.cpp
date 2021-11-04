#include "ThreadPool.h"

#include <chrono> // std::chrono
#include <functional> // std::function
#include <mutex>
#include <thread>

using ui64 = Concurrency::ThreadPool::ui64;

namespace Concurrency
{

ThreadPool::ThreadPool(const ui32 sleep_duration, const ui32 thread_count):
  thread_count_{
    thread_count ? thread_count : std::thread::hardware_concurrency()},
  threads_{new std::thread[
    thread_count ? thread_count : std::thread::hardware_concurrency()]},
  tasks_{},
  paused_{false},
  sleep_duration_{sleep_duration},
  queue_mutex_{},
  running_{false},
  tasks_total_{0}
{
  create_threads();
}

ThreadPool::~ThreadPool()
{
  wait_for_tasks();
  running_ = false;
  destroy_threads();
}

ui64 ThreadPool::get_tasks_queued() const
{
  const std::scoped_lock lock {queue_mutex_};
  return tasks_.size();
}

void ThreadPool::sleep_or_yield()
{
  if (sleep_duration_ > 0)
  {
    std::this_thread::sleep_for(std::chrono::microseconds(sleep_duration_));
  }
  else
  {
    std::this_thread::yield();
  }
}

void ThreadPool::worker()
{
  while (running_)
  {
    std::function<void()> task;

    if (!paused_ && pop_task(task))
    {
      task();
      tasks_total_--;
    }
    else
    {
      sleep_or_yield();
    }
  }
}

void ThreadPool::wait_for_tasks()
{
  while (true)
  {
    if (!paused_)
    {
      if (tasks_total_ == 0)
      {
        break;
      }
    }
    else
    {
      if (get_tasks_running() == 0)
      {
        break;
      }
    }
    sleep_or_yield();
  }
}

void ThreadPool::create_threads()
{
  for (ui32 i {0}; i < thread_count_; ++i)
  {
    threads_[i] = std::thread(&ThreadPool::worker, this);
  }
}

void ThreadPool::destroy_threads()
{
  for (ui32 i {0}; i < thread_count_; ++i)
  {
    threads_[i].join();
  }
}

bool ThreadPool::pop_task(std::function<void()>& task)
{
  // cf. https://en.cppreference.com/w/cpp/thread/scoped_lock
  // The class scoped_lock is a mutex wrapper that owns 1 or more mutexes for
  // the 
  const std::scoped_lock lock {queue_mutex_};
  if (tasks_.empty())
  {
    return false;
  }
  else
  {
    task = std::move(tasks_.front());
    tasks_.pop();
    return true;
  }
}

} // namespace Concurrency