#include "DataStructures/LinkedLists/DoublyLinkedList.h"
#include "Projects/CsVtEdu/InfinitePrecisionCalculator/Multiplication.h"

#include <boost/test/unit_test.hpp>

template <typename T>
constexpr auto iterative_multiplication =
  CsVtEdu::InfinitePrecisionCalculator::iterative_multiplication<T>;

BOOST_AUTO_TEST_SUITE(Projects)
BOOST_AUTO_TEST_SUITE(CsVtEdu)
BOOST_AUTO_TEST_SUITE(InfinitePrecisionCalculator)
BOOST_AUTO_TEST_SUITE(Multiplication_tests)

template <typename T>
using DoublyLinkedList = DataStructures::LinkedLists::DoublyLinkedList<T>;

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
}

BOOST_AUTO_TEST_SUITE_END() // Multiplication_tests
BOOST_AUTO_TEST_SUITE_END() // InfinitePrecisionCalculator
BOOST_AUTO_TEST_SUITE_END() // CsVtEdu
BOOST_AUTO_TEST_SUITE_END() // Projects