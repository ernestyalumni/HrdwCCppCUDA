#include "ThreadPool.h"

#include <thread>

//using Concurrency::ThreadPool::ui32;
//using Concurrency::ThreadPool::ui64;

namespace Concurrency
{

ThreadPool::ThreadPool(const ui32 thread_count):
  thread_count_{
    thread_count ? thread_count : std::thread::hardware_concurrency()},
  threads_{new std::thread[
    thread_count ? thread_count : std::thread::hardware_concurrency()]}
{}

} // namespace Concurrency