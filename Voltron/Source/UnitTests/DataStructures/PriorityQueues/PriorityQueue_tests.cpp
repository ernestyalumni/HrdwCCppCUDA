#include "DataStructures/PriorityQueues/PriorityQueue.h"

#include <boost/test/unit_test.hpp>

using DataStructures::PriorityQueues::PriorityQueue;

BOOST_AUTO_TEST_SUITE(DataStructures)
BOOST_AUTO_TEST_SUITE(PriorityQueues)
BOOST_AUTO_TEST_SUITE(PriorityQueue_tests)

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(PushPushesMaxToTheTopByDefault)
{
  PriorityQueue<int> pq {};
  pq.push(3);
  pq.push(1);
  pq.push(4);  
  BOOST_TEST((pq.top() == 4));
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(PriorityQueueConstructsWithMinLambda)
{
  // Min-heap (user-defined lambda function)
  auto min_comparator = [](int a, int b) -> bool { return a > b; };
  PriorityQueue<int> pq {min_comparator};
  pq.push(3);
  pq.push(1);
  pq.push(4);  
  BOOST_TEST((pq.top() == 1));
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(PriorityQueueConstructsWithStdGreater)
{
  PriorityQueue<int> pq {std::greater<int>{}};
  pq.push(3);
  pq.push(1);
  pq.push(4);  
  BOOST_TEST((pq.top() == 1));
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(PriorityQueueConstructsWithCustomComparator)
{
  auto absolute_compare = [](const int& a, const int& b)
  {
    return std::abs(a) < std::abs(b);
  };

  PriorityQueue<int> pq {absolute_compare};
  pq.push(-3);
  pq.push(-1);
  pq.push(-4);  
  BOOST_TEST((pq.top() == -4));

  // Custom comparator for objects.
  struct Person
  {
    int age_;
    std::string name_;
  };

  PriorityQueue<Person> pq_people {
    [](const Person& a, const Person& b)
    {
      return a.age_ < b.age_;
    }};

  pq_people.push(Person{25, "John"});
  pq_people.push(Person{30, "Jane"});
  pq_people.push(Person{20, "Jim"});
  BOOST_TEST((pq_people.top()->age_ == 30));
  BOOST_TEST((pq_people.top()->name_ == "Jane"));
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(GetTopKGetsTopKElements)
{
  PriorityQueue<int> pq {};
  pq.push(10);
  pq.push(30);
  pq.push(20);  
  pq.push(5);
  pq.push(40);
  pq.push(50);

  BOOST_TEST((pq.get_top_k(3) == std::vector<int> {50, 40, 30}));
  BOOST_TEST((pq.get_top_k(0) == std::vector<int> {}));
  BOOST_TEST((pq.get_top_k(4) == std::vector<int> {50, 40, 30, 20}));

  BOOST_TEST((*pq.top() == 50));
  pq.pop();
  BOOST_TEST((*pq.top() == 40));
  pq.pop();
  BOOST_TEST((*pq.top() == 30));
  pq.pop();
  BOOST_TEST((*pq.top() == 20));
  pq.pop();
  BOOST_TEST((*pq.top() == 10));
  pq.pop();
  BOOST_TEST(!pq.is_empty());
}

BOOST_AUTO_TEST_SUITE_END() // PriorityQueue_tests
BOOST_AUTO_TEST_SUITE_END() // PriorityQueues
BOOST_AUTO_TEST_SUITE_END() // DataStructures