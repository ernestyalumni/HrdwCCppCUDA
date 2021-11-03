#ifndef CONCURRENCY_THREAD_POOL_H
#define CONCURRENCY_THREAD_POOL_H

#include <cstdint>
#include <memory>
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
    ThreadPool(const ui32 thread_count = std::thread::hardware_concurrency());



  private:

    //--------------------------------------------------------------------------
    /// \brief The number of threads in the pool.
    //--------------------------------------------------------------------------
    ui32 thread_count_;

    //--------------------------------------------------------------------------
    /// \brief A smart pointer to manage the memory allocated for the threads.
    //--------------------------------------------------------------------------
    std::unique_ptr<std::thread[]> threads_;

};

} // namespace Concurrency

#endif // CONCURRENCY_THREAD_POOL_H