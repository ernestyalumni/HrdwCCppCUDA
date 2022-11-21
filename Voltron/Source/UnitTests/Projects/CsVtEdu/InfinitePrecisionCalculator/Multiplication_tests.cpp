#include "DataStructures/LinkedLists/DoublyLinkedList.h"
#include "Projects/CsVtEdu/InfinitePrecisionCalculator/Multiplication.h"

#include <boost/test/unit_test.hpp>
#include <vector>

using std::vector;

template <typename T>
constexpr auto iterative_multiplication =
  CsVtEdu::InfinitePrecisionCalculator::iterative_multiplication<T>;

BOOST_AUTO_TEST_SUITE(Projects)
BOOST_AUTO_TEST_SUITE(CsVtEdu)
BOOST_AUTO_TEST_SUITE(InfinitePrecisionCalculator)
BOOST_AUTO_TEST_SUITE(Multiplication_tests)

template <typename T>
using DoublyLinkedList = DataStructures::LinkedLists::DoublyLinkedList<T>;

template <typename T>
using Node = DataStructures::LinkedLists::DoublyLinkedList<T>::Node;

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(IterativeMultiplicationWorksForSingleDigit)
{
  {
    DoublyLinkedList<int> operand_1 {};
    operand_1.push_front(2);
    BOOST_TEST(operand_1.size() = 1);

    DoublyLinkedList<int> operand_2 {};
    operand_2.push_front(3);
    BOOST_TEST(operand_2.size() = 1);

    DoublyLinkedList<int> result {operand_1.size() + operand_2.size(), 0};

    iterative_multiplication<int>(
      operand_1.head(),
      operand_2.head(),
      result.head());

    BOOST_TEST(result.size() == 2);
    BOOST_TEST(result.head()->retrieve() == 6);
    BOOST_TEST(result.head()->next()->retrieve() == 0);
  }
  {
    DoublyLinkedList<int> operand_1 {};
    operand_1.push_front(2);

    DoublyLinkedList<int> operand_2 {};
    operand_2.push_front(4);

    DoublyLinkedList<int> result {operand_1.size() + operand_2.size(), 0};

    iterative_multiplication<int>(
      operand_1.head(),
      operand_2.head(),
      result.head());

    BOOST_TEST(result.size() == 2);
    BOOST_TEST(result.head()->retrieve() == 8);
    BOOST_TEST(result.head()->next()->retrieve() == 0);
  }
  {
    DoublyLinkedList<int> operand_1 {};
    operand_1.push_front(3);

    DoublyLinkedList<int> operand_2 {};
    operand_2.push_front(6);

    DoublyLinkedList<int> result {operand_1.size() + operand_2.size(), 0};

    iterative_multiplication<int>(
      operand_1.head(),
      operand_2.head(),
      result.head());

    BOOST_TEST(result.head()->retrieve() == 8);
    BOOST_TEST(result.head()->next()->retrieve() == 1);
  }
  {
    DoublyLinkedList<int> operand_1 {};
    operand_1.push_front(9);

    DoublyLinkedList<int> operand_2 {};
    operand_2.push_front(9);

    DoublyLinkedList<int> result {operand_1.size() + operand_2.size(), 0};

    iterative_multiplication<int>(
      operand_1.head(),
      operand_2.head(),
      result.head());

    BOOST_TEST(result.head()->retrieve() == 1);
    BOOST_TEST(result.head()->next()->retrieve() == 8);
  }
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(IterativeMultiplicationWorksForDoubleDigitMultiplicands)
{
  {
    DoublyLinkedList<int> operand_1 {};
    operand_1.push_front(9);
    operand_1.push_front(9);
    BOOST_TEST(operand_1.size() = 2);

    DoublyLinkedList<int> operand_2 {};
    operand_2.push_front(9);
    BOOST_TEST(operand_2.size() = 1);

    DoublyLinkedList<int> result {operand_1.size() + operand_2.size(), 0};

    iterative_multiplication<int>(
      operand_1.head(),
      operand_2.head(),
      result.head());

    BOOST_TEST(result.head()->retrieve() == 1);
    BOOST_TEST(result.head()->next()->retrieve() == 9);
    BOOST_TEST(result.head()->next()->next()->retrieve() == 8);
  }
  {
    DoublyLinkedList<int> operand_1 {};
    operand_1.push_front(9);
    BOOST_TEST(operand_1.size() = 1);

    DoublyLinkedList<int> operand_2 {};
    operand_2.push_front(9);
    operand_2.push_front(9);
    BOOST_TEST(operand_2.size() = 2);

    DoublyLinkedList<int> result {operand_1.size() + operand_2.size(), 0};

    iterative_multiplication<int>(
      operand_1.head(),
      operand_2.head(),
      result.head());

    BOOST_TEST(result.head()->retrieve() == 1);
    BOOST_TEST(result.head()->next()->retrieve() == 9);
    BOOST_TEST(result.head()->next()->next()->retrieve() == 8);
  }
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(IterativeMultiplicationWorksForLargeValues)
{
  {
    // 123456789
    DoublyLinkedList<int> operand_1 {};
    operand_1.push_front(1);
    operand_1.push_front(2);
    operand_1.push_front(3);
    operand_1.push_front(4);
    operand_1.push_front(5);
    operand_1.push_front(6);
    operand_1.push_front(7);
    operand_1.push_front(8);
    operand_1.push_front(9);

    BOOST_TEST(operand_1.size() = 9);

    // 1111111111
    DoublyLinkedList<int> operand_2 {};
    for (std::size_t i {0}; i < 10; ++i)
    {
      operand_2.push_front(1);
    }
    BOOST_TEST(operand_2.size() = 10);

    DoublyLinkedList<int> result {operand_1.size() + operand_2.size(), 0};
    BOOST_TEST_REQUIRE(result.size() == 19);

    iterative_multiplication<int>(
      operand_1.head(),
      operand_2.head(),
      result.head());

    const vector<int> expected {1,3,7,1,7,4,2,0,9,9,8,6,2,8,2,5,7,9};
    BOOST_TEST_REQUIRE(expected.size() == 18);

    Node<int>* result_ptr {result.head()};
    for (std::size_t i {0}; i < expected.size(); ++i)
    {
      BOOST_TEST(result_ptr->retrieve() == expected[expected.size() - 1 - i]);
      result_ptr = result_ptr->next();
    }

    BOOST_TEST(result_ptr->retrieve() == 0);
  }
  {
    // 99999999
    DoublyLinkedList<int> operand_1 {};
    operand_1.push_front(9);
    operand_1.push_front(9);
    operand_1.push_front(9);
    operand_1.push_front(9);
    operand_1.push_front(9);
    operand_1.push_front(9);
    operand_1.push_front(9);
    operand_1.push_front(9);

    BOOST_TEST(operand_1.size() = 8);

    // 990001
    DoublyLinkedList<int> operand_2 {};
    operand_2.push_front(9);
    operand_2.push_front(9);
    operand_2.push_front(0);
    operand_2.push_front(0);
    operand_2.push_front(0);
    operand_2.push_front(1);
    BOOST_TEST(operand_2.size() = 6);

    DoublyLinkedList<int> result {operand_1.size() + operand_2.size(), 0};
    BOOST_TEST_REQUIRE(result.size() == 14);

    iterative_multiplication<int>(
      operand_1.head(),
      operand_2.head(),
      result.head());

    const vector<int> expected {9,9,0,0,0,0,9,9,0,0,9,9,9,9};
    BOOST_TEST_REQUIRE(expected.size() == 14);

    Node<int>* result_ptr {result.head()};
    for (std::size_t i {0}; i < expected.size(); ++i)
    {
      BOOST_TEST(result_ptr->retrieve() == expected[expected.size() - 1 - i]);
      result_ptr = result_ptr->next();
    }
  }
}

BOOST_AUTO_TEST_SUITE_END() // Multiplication_tests
BOOST_AUTO_TEST_SUITE_END() // InfinitePrecisionCalculator
BOOST_AUTO_TEST_SUITE_END() // CsVtEdu
BOOST_AUTO_TEST_SUITE_END() // Projects