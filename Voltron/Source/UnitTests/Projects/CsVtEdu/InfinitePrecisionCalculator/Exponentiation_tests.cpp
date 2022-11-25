#include "DataStructures/LinkedLists/DoublyLinkedList.h"
#include "Projects/CsVtEdu/InfinitePrecisionCalculator/Exponentiation.h"
#include "Projects/CsVtEdu/InfinitePrecisionCalculator/ParseTextFile.h"

#include <boost/test/unit_test.hpp>

template <typename T>
constexpr auto exponentiate =
  CsVtEdu::InfinitePrecisionCalculator::exponentiate<T>;

template <typename T>
constexpr auto strip_leading_zeros =
  CsVtEdu::InfinitePrecisionCalculator::strip_leading_zeros<T>;

BOOST_AUTO_TEST_SUITE(Projects)
BOOST_AUTO_TEST_SUITE(CsVtEdu)
BOOST_AUTO_TEST_SUITE(InfinitePrecisionCalculator)
BOOST_AUTO_TEST_SUITE(Exponentiation_tests)

template <typename T>
using DoublyLinkedList = DataStructures::LinkedLists::DoublyLinkedList<T>;

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(ExponentiationWorksForExponent0)
{
  DoublyLinkedList<int> base {};
  base.push_front(2);
  DoublyLinkedList<int> result {2 * base.size(), 1};
  exponentiate<int>(base, 0, result);
  BOOST_TEST(result.size() == 1);
  BOOST_TEST(result.head()->retrieve() == 0);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(ExponentiationWorksForExponent1)
{
  DoublyLinkedList<int> base {};
  base.push_front(2);
  base.push_front(3);
  DoublyLinkedList<int> result {2 * base.size(), 1};
  exponentiate<int>(base, 1, result);
  BOOST_TEST(result.size() == 2);
  BOOST_TEST(result.head()->retrieve() == 3);
  BOOST_TEST(result.head()->next()->retrieve() == 2);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(ExponentiationWorksForSingleDigit)
{
  DoublyLinkedList<int> operand_1 {};
  operand_1.push_front(2);

  {
    DoublyLinkedList<int> result {2 * operand_1.size(), 0};
    exponentiate<int>(operand_1, 2, result);
    strip_leading_zeros<int>(result);

    BOOST_TEST(result.size() == 1);
    BOOST_TEST(result.head()->retrieve() == 4);
  }
  {
    DoublyLinkedList<int> result {2 * operand_1.size(), 0};
    exponentiate<int>(operand_1, 3, result);
    strip_leading_zeros<int>(result);

    BOOST_TEST(result.size() == 1);
    BOOST_TEST(result.head()->retrieve() == 8);
    BOOST_TEST(result.head()->next() == nullptr);

    BOOST_TEST(result.tail() != nullptr);
    BOOST_TEST(result.tail()->retrieve() == 8);
    BOOST_TEST(result.tail()->next() == nullptr);
    BOOST_TEST(result.tail()->previous() == nullptr);
  }
  {
    DoublyLinkedList<int> result {2 * operand_1.size(), 0};
    exponentiate<int>(operand_1, 4, result);
    strip_leading_zeros<int>(result);

    BOOST_TEST(result.size() == 4);
    BOOST_TEST(result.head()->retrieve() == 6);
    BOOST_TEST(result.head()->next()->retrieve() == 1);
    BOOST_TEST(result.head()->next()->next()->retrieve() == 0);
    BOOST_TEST(result.head()->next()->next()->next()->retrieve() == 0);

    BOOST_TEST(result.tail() != nullptr);
    BOOST_TEST(result.tail()->retrieve() == 1);
  }
}

BOOST_AUTO_TEST_SUITE_END() // Exponentiation_tests
BOOST_AUTO_TEST_SUITE_END() // InfinitePrecisionCalculator
BOOST_AUTO_TEST_SUITE_END() // CsVtEdu
BOOST_AUTO_TEST_SUITE_END() // Projects