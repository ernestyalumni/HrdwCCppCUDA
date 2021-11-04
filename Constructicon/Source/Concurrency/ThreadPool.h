#ifndef CONCURRENCY_THREAD_POOL_H
#define CONCURRENCY_THREAD_POOL_H

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>

namespace Concurrency
{

//------------------------------------------------------------------------------
/// \brief A C++17 thread pool class. The user submits tasks to be executed into
/// a queue. Whenever a thread becomes available, it pops a task from the queue
/// and executes it.
///
/// Each task is automatically assigned a future, which can be used to wait for
/// the task to finish executing and/or obtain its eventual return value.
/// \ref https://github.com/bshoshany/thread-pool/blob/master/thread_pool.hpp
//------------------------------------------------------------------------------
class ThreadPool
{
  public:

    // Fastest unsigned integer type with width of at least 32, and 64 bits,
    // respectively.
    using ui32 = uint_fast32_t;
    using ui64 = uint_fast64_t;

    //--------------------------------------------------------------------------
    /// \details 
    /// cf. https://en.cppreference.com/w/cpp/thread/thread/hardware_concurrency
    /// static unsigned int hardware_concurrency() returns number of concurrent
    /// threads supported by the implementation. Value should be considered only
    /// a hint.
    //--------------------------------------------------------------------------
    ThreadPool(
      const ui32 sleep_duration = 1000,
      const ui32 thread_count = std::thread::hardware_concurrency());

    //--------------------------------------------------------------------------
    /// \brief Destruct the thread pool. Waits for all tasks to complete, then
    /// destroys all threads. Note that if the variable paused is set to true,
    /// then any tasks still in the queue will never be executed.
    //--------------------------------------------------------------------------
    ~ThreadPool();

    //--------------------------------------------------------------------------
    /// \brief Get the number of tasks currently waiting in the queue to be
    /// executed by the threads.
    ///
    /// \return The number of queued tasks.
    //--------------------------------------------------------------------------
    ui64 get_tasks_queued() const;

    //--------------------------------------------------------------------------
    /// \brief Get the number of tasks currently being executed by the threads.
    ///
    /// \return The number of running tasks.
    //--------------------------------------------------------------------------
    ui32 get_tasks_running() const
    {
      return tasks_total_ - static_cast<ui32>(get_tasks_queued());
    }

    //--------------------------------------------------------------------------
    /// \brief Get the total number of unfinished tasks - either still in the
    /// queue, or running in a thread.
    ///
    /// \return The total number of tasks.
    //--------------------------------------------------------------------------
    ui32 get_tasks_total() const
    {
      return tasks_total_;
    }

    //--------------------------------------------------------------------------
    /// \brief Get the total number of threads in the pool.
    /// \return The number of threads.
    //--------------------------------------------------------------------------
    ui32 get_thread_count() const
    {
      return thread_count_;
    }

    //--------------------------------------------------------------------------
    /// \brief Sleep for sleep_duration microseconds. If that variable is set to
    /// zero, yield instead.
    //--------------------------------------------------------------------------
    void sleep_or_yield();

    //--------------------------------------------------------------------------
    /// \brief A work function to be assigned to each thread in the pool.
    /// Continuously pops tasks out of the queue and executes them, as long as
    /// the atomic variable running is set to true.
    //--------------------------------------------------------------------------
    void worker();

    //--------------------------------------------------------------------------
    /// \brief Wait for tasks to be completed. Normally, this function waits for
    /// all tasks, both those that are currently running in the threads and
    /// those that are still waiting in the queue. However, if the variable
    /// paused is set to true, this function only waits for the currently
    /// running tasks (otherwise it would wait forever). To wait for a specific
    /// task, use submit() instead, and call the wait() member function of the
    /// generated future.
    //--------------------------------------------------------------------------
    void wait_for_tasks();

  private:

    void create_threads();

    void destroy_threads();

    //--------------------------------------------------------------------------
    /// \brief Try to pop a new task out of the queue.
    ///
    /// \param task A reference to the task. Will be populated with a function
    /// if the queue is not empty.
    /// \return True if a task was found, false if the queue is empty.
    //--------------------------------------------------------------------------
    bool pop_task(std::function<void()>& task);

    //--------------------------------------------------------------------------
    /// \brief The number of threads in the pool.
    //--------------------------------------------------------------------------
    ui32 thread_count_;

    //--------------------------------------------------------------------------
    /// \brief A smart pointer to manage the memory allocated for the threads.
    //--------------------------------------------------------------------------
    std::unique_ptr<std::thread[]> threads_;

    //--------------------------------------------------------------------------
    /// \brief A queue of tasks to be executed by the threads.
    //--------------------------------------------------------------------------
    std::queue<std::function<void()>> tasks_;

    //--------------------------------------------------------------------------
    /// \brief An atomic variable indicating to the workers to pause. When set
    /// to true, the workers temporarily stop popping new tasks out of the
    /// queue, although any tasks already executed will keep running until they
    /// are done. Set to false again to resume popping tasks.
    //--------------------------------------------------------------------------
    std::atomic<bool> paused_;

    ui32 sleep_duration_;

    //--------------------------------------------------------------------------
    /// \brief A mutex to synchronize access to the task queue by different
    /// threads.
    //--------------------------------------------------------------------------
    // cf. https://en.cppreference.com/w/cpp/thread/mutex
    // Mutex class is a synchronization primitive used to protect shared data
    // from being simultaneously accessed by multiple threads.
    mutable std::mutex queue_mutex_;

    //--------------------------------------------------------------------------
    /// \brief An atomic variable indicating to the workers to keep running.
    /// When set to false, the workers permanently stop working.
    //--------------------------------------------------------------------------
    std::atomic<bool> running_;

    //--------------------------------------------------------------------------
    /// \brief An atomic variable to keep track of the total number of
    /// unfinished tasks - either still in the queue, or running in a thread.
    //--------------------------------------------------------------------------
    std::atomic<ui32> tasks_total_;
};

} // namespace Concurrency

#endif // CONCURRENCY_THREAD_POOL_H