#ifndef UNIT_TESTS_DATA_STRUCTURES_LINKED_LISTS_DOUBLY_LINKED_LIST_TEST_VALUES_H
#define UNIT_TESTS_DATA_STRUCTURES_LINKED_LISTS_DOUBLY_LINKED_LIST_TEST_VALUES_H

#include "DataStructures/LinkedLists/DoublyLinkedList.h"

namespace UnitTests
{
namespace DataStructures
{
namespace LinkedLists
{

struct DoublyLinkedListTestValues
{
  template <typename T>
  using DoublyLinkedList = ::DataStructures::LinkedLists::DoublyLinkedList<T>;

  DoublyLinkedListTestValues();

  ~DoublyLinkedListTestValues() = default;

  // 99954321
  DoublyLinkedList<int> large_operand_1_;
  // 66789
  DoublyLinkedList<int> large_operand_2_;
  // 876543210
  DoublyLinkedList<int> large_operand_3_;
  // 3456789
  DoublyLinkedList<int> large_operand_4_;
};

} // namespace LinkedLists
} // namespace DataStructures
} // namespace UnitTests

#endif // UNIT_TESTS_DATA_STRUCTURES_LINKED_LISTS_DOUBLY_LINKED_LIST_TEST_VALUES_H
