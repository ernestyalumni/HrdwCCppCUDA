#include "DataStructures/LinkedLists/DoublyLinkedList.h"
#include "Projects/CsVtEdu/InfinitePrecisionCalculator/Addition.h"

#include <boost/test/unit_test.hpp>
#include <string>
#include <vector>

template <typename T>
constexpr auto addition = CsVtEdu::InfinitePrecisionCalculator::addition<T>;

template <typename T>
constexpr auto recursive_addition =
  CsVtEdu::InfinitePrecisionCalculator::recursive_addition<T>;

BOOST_AUTO_TEST_SUITE(Projects)
BOOST_AUTO_TEST_SUITE(CsVtEdu)
BOOST_AUTO_TEST_SUITE(InfinitePrecisionCalculator)
BOOST_AUTO_TEST_SUITE(Addition_tests)

template <typename T>
using DoublyLinkedList = DataStructures::LinkedLists::DoublyLinkedList<T>;

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(RecursiveAdditionAddsWithNoCarry)
{
  // 876543210
  DoublyLinkedList<int> operand_1 {};
  operand_1.push_back(0);
  operand_1.push_back(1);
  operand_1.push_back(2);
  operand_1.push_back(3);
  operand_1.push_back(4);
  operand_1.push_back(5);
  operand_1.push_back(6);
  operand_1.push_back(7);
  operand_1.push_back(8);

  // 123456789
  DoublyLinkedList<int> operand_2 {};
  operand_2.push_back(9);
  operand_2.push_back(8);
  operand_2.push_back(7);
  operand_2.push_back(6);
  operand_2.push_back(5);
  operand_2.push_back(4);
  operand_2.push_back(3);
  operand_2.push_back(2);
  operand_2.push_back(1);

  recursive_addition<int>(operand_1.head(), operand_2.head(), nullptr, 0);

  auto resulting_iterator = operand_1.begin();
  for (std::size_t i {0}; i < 9; ++i)
  {
    BOOST_TEST(*resulting_iterator == 9);
    ++resulting_iterator;
  }

  BOOST_TEST((resulting_iterator == operand_1.end()));
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(RecursiveAdditionAddsWithNoCarryAndLargerFirstOperand)
{
  // 876543210
  DoublyLinkedList<int> operand_1 {};
  operand_1.push_back(0);
  operand_1.push_back(1);
  operand_1.push_back(2);
  operand_1.push_back(3);
  operand_1.push_back(4);
  operand_1.push_back(5);
  operand_1.push_back(6);
  operand_1.push_back(7);
  operand_1.push_back(8);

  // 3456789
  DoublyLinkedList<int> operand_2 {};
  operand_2.push_back(9);
  operand_2.push_back(8);
  operand_2.push_back(7);
  operand_2.push_back(6);
  operand_2.push_back(5);
  operand_2.push_back(4);
  operand_2.push_back(3);

  recursive_addition<int>(operand_1.head(), operand_2.head(), nullptr, 0);

  auto resulting_iterator = operand_1.begin();
  for (std::size_t i {0}; i < 7; ++i)
  {
    BOOST_TEST(*resulting_iterator == 9);
    ++resulting_iterator;
  }

  BOOST_TEST(*resulting_iterator == 7);
  ++resulting_iterator;
  BOOST_TEST(*resulting_iterator == 8);
  ++resulting_iterator;

  BOOST_TEST((resulting_iterator == operand_1.end()));
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(RecursiveAdditionAddsWithCarryForNextToLeadingDigits)
{
  DoublyLinkedList<int> operand_1 {};
  operand_1.push_back(1);
  operand_1.push_back(2);
  operand_1.push_back(3);
  operand_1.push_back(4);
  operand_1.push_back(5);
  operand_1.push_back(6);
  operand_1.push_back(7);
  operand_1.push_back(7);

  DoublyLinkedList<int> operand_2 {};
  operand_2.push_back(9);
  operand_2.push_back(8);
  operand_2.push_back(7);
  operand_2.push_back(6);
  operand_2.push_back(6);
  operand_2.push_back(5);
  operand_2.push_back(4);
  operand_2.push_back(1);

  recursive_addition<int>(operand_1.head(), operand_2.head(), nullptr, 0);

  auto resulting_iterator = operand_1.begin();
  BOOST_TEST(*resulting_iterator == 0);
  ++resulting_iterator;

  for (std::size_t i {0}; i < 3; ++i)
  {
    BOOST_TEST(*resulting_iterator == 1);
    ++resulting_iterator;
  }

  for (std::size_t i {0}; i < 3; ++i)
  {
    BOOST_TEST(*resulting_iterator == 2);
    ++resulting_iterator;
  }

  BOOST_TEST(*resulting_iterator == 9);
  ++resulting_iterator;

  BOOST_TEST((resulting_iterator == operand_1.end()));
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(RecursiveAdditionAddsWithCarryForLargerSizedFirstOperand)
{
  // 99954321
  DoublyLinkedList<int> operand_1 {};
  operand_1.push_back(1);
  operand_1.push_back(2);
  operand_1.push_back(3);
  operand_1.push_back(4);
  operand_1.push_back(5);
  operand_1.push_back(9);
  operand_1.push_back(9);
  operand_1.push_back(9);

  // 66789
  DoublyLinkedList<int> operand_2 {};
  operand_2.push_back(9);
  operand_2.push_back(8);
  operand_2.push_back(7);
  operand_2.push_back(6);
  operand_2.push_back(6);

  recursive_addition<int>(operand_1.head(), operand_2.head(), nullptr, 0);

  auto resulting_iterator = operand_1.begin();
  BOOST_TEST(*resulting_iterator == 0);
  ++resulting_iterator;

  for (std::size_t i {0}; i < 3; ++i)
  {
    BOOST_TEST(*resulting_iterator == 1);
    ++resulting_iterator;
  }

  BOOST_TEST(*resulting_iterator == 2);
  ++resulting_iterator;

  for (std::size_t i {0}; i < 3; ++i)
  {
    BOOST_TEST(*resulting_iterator == 0);
    ++resulting_iterator;
  }

  BOOST_TEST(*resulting_iterator == 1);
  ++resulting_iterator;

  BOOST_TEST((resulting_iterator == operand_1.end()));
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(RecursiveAdditionAddsWithCarryForLargerSizedSecondOperand)
{
  // 66789
  DoublyLinkedList<int> operand_1 {};
  operand_1.push_back(9);
  operand_1.push_back(8);
  operand_1.push_back(7);
  operand_1.push_back(6);
  operand_1.push_back(6);

  // 99954321
  DoublyLinkedList<int> operand_2 {};
  operand_2.push_back(1);
  operand_2.push_back(2);
  operand_2.push_back(3);
  operand_2.push_back(4);
  operand_2.push_back(5);
  operand_2.push_back(9);
  operand_2.push_back(9);
  operand_2.push_back(9);

  // 100021110
  recursive_addition<int>(operand_1.head(), operand_2.head(), nullptr, 0);

  auto resulting_iterator = operand_1.begin();
  BOOST_TEST(*resulting_iterator == 0);
  ++resulting_iterator;

  for (std::size_t i {0}; i < 3; ++i)
  {
    BOOST_TEST(*resulting_iterator == 1);
    ++resulting_iterator;
  }

  BOOST_TEST(*resulting_iterator == 2);
  ++resulting_iterator;

  for (std::size_t i {0}; i < 3; ++i)
  {
    BOOST_TEST(*resulting_iterator == 0);
    ++resulting_iterator;
  }

  BOOST_TEST(*resulting_iterator == 1);
  ++resulting_iterator;

  BOOST_TEST((resulting_iterator == operand_1.end()));
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(AdditionAdds)
{
  // 56669777
  DoublyLinkedList<int> operand_1 {};
  operand_1.push_front(5);
  operand_1.push_front(6);
  operand_1.push_front(6);
  operand_1.push_front(6);
  operand_1.push_front(9);
  operand_1.push_front(7);
  operand_1.push_front(7);
  operand_1.push_front(7);

  // 99999911111
  DoublyLinkedList<int> operand_2 {};
  operand_2.push_front(9);
  operand_2.push_front(9);
  operand_2.push_front(9);
  operand_2.push_front(9);
  operand_2.push_front(9);
  operand_2.push_front(9);
  operand_2.push_front(1);
  operand_2.push_front(1);
  operand_2.push_front(1);
  operand_2.push_front(1);
  operand_2.push_front(1);

  addition<int>(operand_1, operand_2);

  auto operand_1_iter = operand_1.begin();
  // 100056580888
  for (int i {0}; i < 3; ++i)
  {
    BOOST_TEST(*operand_1_iter == 8);
    ++operand_1_iter;
  }

  std::vector<int> expected {0, 8, 5, 6, 5, 0, 0, 0, 1};

  for (auto ele : expected)
  {
    BOOST_TEST(*operand_1_iter == ele);
    ++operand_1_iter;
  }
}

BOOST_AUTO_TEST_SUITE_END() // Addition_tests
BOOST_AUTO_TEST_SUITE_END() // InfinitePrecisionCalculator
BOOST_AUTO_TEST_SUITE_END() // CsVtEdu
BOOST_AUTO_TEST_SUITE_END() // Projects