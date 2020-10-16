//------------------------------------------------------------------------------
// \file DataStructures_tests.cpp
//------------------------------------------------------------------------------
#include <array>
#include <boost/test/unit_test.hpp>
#include <deque>
#include <forward_list>
#include <list>
#include <memory>
#include <stack>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

BOOST_AUTO_TEST_SUITE(DataStructures)
BOOST_AUTO_TEST_SUITE(DataStructures_tests)

BOOST_AUTO_TEST_SUITE(StdForwardList_tests)

// cf. https://www.geeksforgeeks.org/forward-list-c-set-1-introduction-important-functions/
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(DemonstrateStdForwardListAsLinkedList)
{
  {
    // Declaring forward list
    std::forward_list<int> flist1;

    // Declaring another forward list
    std::forward_list<int> flist2;

    // assigns values to container
    flist1.assign({1, 2, 3});

    // Assigning repeating values using assing()
    // 5 elements with value 10
    flist2.assign(5, 10);

    // Displaying forward lists

    int index {1};
    for (int& a : flist1)
    {
      BOOST_TEST(a == index);
      index++;
    }

    for (int& b : flist2)
    {
      BOOST_TEST(b == 10);
    }
  }
  {
    std::forward_list flist {10, 20, 30, 40, 50};

    // push_front pushes to last in at "[0] of flist[0]", the head

    // Inserting value using push front()
    // Inserts 60 at front
    flist.push_front(60); // 60, 10, 20, 30, 40, 50

    std::size_t index {0};
    std::vector<int> check_by_vec {60, 10, 20, 30, 40, 50};
    for (int& c : flist)
    {
      BOOST_TEST(check_by_vec[index] == c);
      index++;
      //std::cout << c << ' ';
    }

    // Inserting value using emplace_front()
    // Inserts 70 at front
    flist.emplace_front(70);

    check_by_vec.assign({70, 60, 10, 20, 30, 40, 50});
    index = 0;
    for (int& c : flist)
    {
      BOOST_TEST(check_by_vec[index] = c);
      index++;
      //std::cout << c << ' ';
    }

    // Deleting first value using pop_front()
    // Pops 70
    flist.pop_front();

    check_by_vec.assign({60, 10, 20, 30, 40, 50});
    index = 0;
    for (int& c : flist)
    {
      BOOST_TEST(check_by_vec[index] = c);
      index++;
      //std::cout << c << ' ';
    }
  }

  {
    // Initializing forward list
    std::forward_list<int> flist {10, 20, 30};

    // Declaring a forward list iterator
    std::forward_list<int>::iterator ptr;

    // Inserting value using insert_after()
    // starts insertion from second position
    // Iterator to inserted element. typically, but in this case
    // iterator to last element inserted

    // insert *after* means insert after
    ptr = flist.insert_after(flist.begin(), {1, 2, 3});

    std::vector<int> check_by_vec {10, 1, 2, 3, 20, 30};

    std::size_t index {0};
    for (int& c : flist)
    {
      BOOST_TEST(check_by_vec[index] == c);
      index++;
      //std::cout << c << ' ';
    }
    //std::cout << "\n";

    // Inerting value using emplace_after()
    // inserts 2 after ptr
    ptr = flist.emplace_after(ptr, 2);

    check_by_vec.assign({10, 1, 2, 3, 2, 20, 30});

    index = 0;
    for (int& c : flist)
    {
      BOOST_TEST(check_by_vec[index] == c);
      index++;
      //std::cout << c << ' ';
    }

  }
}

BOOST_AUTO_TEST_SUITE_END() // StdForwardList_tests

BOOST_AUTO_TEST_SUITE(StdStack_tests)

// cf. https://en.cppreference.com/w/cpp/header/stack
// cf. https://www.geeksforgeeks.org/stack-in-cpp-stl/
// Stacks are LIFO (Last In First Out)

template <typename T>
std::vector<T> copy_stack(const std::stack<T>& s)
{
  // https://en.cppreference.com/w/cpp/container/stack/operator%3D
  std::stack<T> copy_of_s = s;

  std::vector<T> result;

  while (!copy_of_s.empty())
  {
    result.emplace_back(copy_of_s.top());
    copy_of_s.pop();
  }

  return result;
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(DemonstrateStdStack)
{
  {
    // LIFO
    std::stack<int> s;
    s.push(10);
    s.push(30);
    s.push(20);
    s.push(5);
    s.push(1);

    std::vector<int> result {copy_stack(s)};

    BOOST_TEST((result == std::vector<int> {1, 5, 20, 30, 10}));

    BOOST_TEST(s.size() == 5);
    BOOST_TEST(s.top() == 1);
    s.pop();
    result = copy_stack(s);
    BOOST_TEST((result == std::vector<int> {5, 20, 30, 10}));        
  }
}

BOOST_AUTO_TEST_SUITE_END() // StdStack_tests

BOOST_AUTO_TEST_SUITE(StdDeque_tests)

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(DemonstrateStdDeque)
{
  {
    // Create a deque containing integers
    std::deque<int> d {7, 5, 16, 8};

    // Add an integer to the beginning and end of the deque
    d.push_front(13);
    d.push_back(25);

    const std::vector<int> target {13, 7, 5, 16, 8, 25};

    std::size_t index {0};
    for (int n : d)
    {
      BOOST_TEST(n == target[index]);
      index++;
    }
  }
  // FIFO
  {
    std::deque<int> d;
    d.push_back(25);
    d.push_back(8);
    d.push_back(16);
    d.push_back(5);
    d.push_back(7);
    d.push_back(13);

    BOOST_TEST(d.back() == 13);

    // if push_back is the enqueue (add to queue)
    // pop_front is the dequeue (first in, first out)
    d.pop_front();

    BOOST_TEST(d.front() == 8);    
  }

  {
    std::deque<int> d;
    d.push_front(25);
    d.push_front(8);
    d.push_front(16);
    d.push_front(5);
    d.push_front(7);
    d.push_front(13);

    BOOST_TEST(d.back() == 25);

    // if push_front is the enqueue (add to queue)
    // pop_back is the dequeue (first in, first out)
    d.pop_back();

    BOOST_TEST(d.back() == 8);
  }

}

BOOST_AUTO_TEST_SUITE_END() // StdDeque_tests

BOOST_AUTO_TEST_SUITE_END() // DataStructures_tests
BOOST_AUTO_TEST_SUITE_END() // DataStructures