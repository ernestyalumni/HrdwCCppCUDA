//------------------------------------------------------------------------------
/// \file Mutex_tests.cpp
///
/// \ref Ch. 5 of Cukic.
//------------------------------------------------------------------------------
#include <boost/test/unit_test.hpp>

#include <mutex> // std::unique_lock
#include <string>
#include <thread>
#include <vector>
#include <utility> // std::pair

BOOST_AUTO_TEST_SUITE(Mutable)
BOOST_AUTO_TEST_SUITE(Mutex_tests)

BOOST_AUTO_TEST_SUITE(Mutex_tests)

class EmploymentHistory : public std::vector<std::string>
{
  public:
    EmploymentHistory() = default;

    bool loaded() const
    {
      return true;
    }
};

struct SequenceNumber;

// cf. https://en.cppreference.com/w/cpp/thread/unique_lock
// Modified transfer function as class.
/*
class DualSequenceNumbers
{
  public:

    DualSequenceNumbers(
      const int increasing_sequence_number,
      const int decreasing_sequence_number
      ):
      increasing_sequence_number_{increasing_sequence_number},
      decreasing_sequence_number_{decreasing_sequence_number}
    {}

  private:
    mutable std::mutex increasing_sequence_number_mutex_;
    mutable std::mutex decreasing_sequence_number_mutex_;
    mutable SequenceNumber increasing_sequence_number_;
    mutable SequenceNumber decreasing_sequence_number_;
};
*/

// cf. Listing 5.8 Using mutable to implement caching, pp. 117
class Person
{
  public:

    Person(const EmploymentHistory employment_history) :
      employment_history_{employment_history}
    {}

    EmploymentHistory employment_history() const
    {
      // Locks the mutex to guarantee that a concurrent invocation of
      // employment_history_ can't be executed until you finish retrieving
      // until you finish retrieving the data from the database.
      std::unique_lock<std::mutex> lock {
        employment_history_mutex_};
    
      // Gets the data if it isn't already loaded.
      if (!employment_history_.loaded())
      {
        load_employment_history();
      }

      // the data is loaded; you're returning it to the caller.
      return employment_history_;

    // When you exit this block, the lock variable will be destroyed and the
    // mutex will be unlocked.
    } 

  private:
    void load_employment_history() const
    {
      return;
    }

    // You want to be able to lock the mutex from a constant member function, so
    // it needs to be mutable as well as the variable you're initializing.
    mutable std::mutex employment_history_mutex_;
    mutable EmploymentHistory employment_history_;
};

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(MutexExamples)
{
  BOOST_TEST(true);
}

BOOST_AUTO_TEST_SUITE(cppreference_tests) // cppreference.com 

// cf. https://en.cppreference.com/w/cpp/thread/unique_lock
// Modified struct Box from cppreference example.
struct SequenceNumber
{
  explicit SequenceNumber(int number) :
    sequence_number_{number}
  {}

  ~SequenceNumber() = default;

  int sequence_number_;
  std::mutex m_;
};

// cf. https://en.cppreference.com/w/cpp/thread/unique_lock
void transfer(SequenceNumber& from, SequenceNumber& to, int number)
{
  // Don't actually take the locks yet
  std::unique_lock<std::mutex> lock1 {from.m_, std::defer_lock};
  std::unique_lock<std::mutex> lock2 {from.m_, std::defer_lock};

  // Lock both unique_locks without deadlock
  std::lock(lock1, lock2);

  from.sequence_number_ -= number;
  to.sequence_number_ += number;

  // 'from.m_' and 'to.m_' mutexes unlocked in 'unique_lock' dtors
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(cppreferenceExamples)
{
  SequenceNumber acc1 {100};
  SequenceNumber acc2 {50};

  BOOST_TEST(acc1.sequence_number_ == 100);
  BOOST_TEST(acc2.sequence_number_ == 50);

  // Threads begin process once it gets created.
  //std::thread t1 {transfer, std::ref(acc1), std::ref(acc2), 10};

  //BOOST_TEST(acc1.sequence_number_ == 90);
  //BOOST_TEST(acc2.sequence_number_ == 60);

//  std::thread t2 {transfer, std::ref(acc2), std::ref(acc1), 5};

  //t1.join();
//  t2.join();

  //BOOST_TEST(acc1.sequence_number_ == 95);
  //BOOST_TEST(acc2.sequence_number_ == 55);
  
}

BOOST_AUTO_TEST_SUITE_END() // cppreference_tests

BOOST_AUTO_TEST_SUITE_END() // Mutex_tests

BOOST_AUTO_TEST_SUITE_END() // Mutex_tests
BOOST_AUTO_TEST_SUITE_END() // Mutable