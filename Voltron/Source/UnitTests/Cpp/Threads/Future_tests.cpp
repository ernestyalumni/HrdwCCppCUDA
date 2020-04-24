//------------------------------------------------------------------------------
// \file Future_tests.cpp
//------------------------------------------------------------------------------

#include <boost/test/unit_test.hpp>
#include <future>
#include <thread>

BOOST_AUTO_TEST_SUITE(Cpp)
BOOST_AUTO_TEST_SUITE(Threads)
BOOST_AUTO_TEST_SUITE(Future_tests)

// cf. https://en.cppreference.com/w/cpp/thread/future
// cf. https://en.cppreference.com/w/cpp/thread/future/future
// future(future& other) noexcept
// Move ctor; constructs std::future with shared state of other using move
// semantics. After ctor, other.valid() == false.
// std::future is not CopyConstructible:
// future(const future& other) = delete;

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(FutureWorksWithPackagedTaskASyncPromise)
{
  // future from a packaged_task
  std::packaged_task<int()> task([]{ return 7;}); // wrap a function
  std::future<int> f1 {task.get_future()}; // get a future
  std::thread t {std::move(task)}; // launch on a thread

  // future from an async()
  std::future <int> f2 {std::async(std::launch::async, []{ return 8; })};

  // future from a promise
  std::promise <int> p;
  std::future<int> f3 {p.get_future()};
  std::thread( [&p]{ p.set_value_at_thread_exit(9); }).detach();

  // If run here, obtain this error:
  // fatal error: in "Cpp/...": signal: SIGABRT (application abort requested)
  //BOOST_TEST(f2.get() == 8);

  // Ok to place here if f3.wait() not called. Otherwise,
  // fatal error: in "Cpp/...": signal: SIGABRT (application abort requested)
  // BOOST_TEST(f3.get() == 9);

  f1.wait();
  f2.wait();
  f3.wait();

  BOOST_TEST(f1.get() == 7);
  BOOST_TEST(f2.get() == 8);
  BOOST_TEST(f3.get() == 9);

  t.join();

  // Ok to place here.
  //BOOST_TEST(f2.get() == 8);

  // Ok to place here.
  //BOOST_TEST(f3.get() == 9);
}

BOOST_AUTO_TEST_SUITE_END() // Future_tests
BOOST_AUTO_TEST_SUITE_END() // Threads
BOOST_AUTO_TEST_SUITE_END() // Cpp