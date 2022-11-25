#include "DoublyLinkedListTestValues.h"

//#include "DataStructures/LinkedLists/DoublyLinkedList.h"

namespace UnitTests
{
namespace DataStructures
{
namespace LinkedLists
{

DoublyLinkedListTestValues::DoublyLinkedListTestValues():
  // 99954321
  large_operand_1_{},
  // 66789
  large_operand_2_{},
  // 876543210
  large_operand_3_{},
  // 3456789
  large_operand_4_{}
{
  large_operand_1_.push_back(1);
  large_operand_1_.push_back(2);
  large_operand_1_.push_back(3);
  large_operand_1_.push_back(4);
  large_operand_1_.push_back(5);
  large_operand_1_.push_back(9);
  large_operand_1_.push_back(9);
  large_operand_1_.push_back(9);

  large_operand_2_.push_back(9);
  large_operand_2_.push_back(8);
  large_operand_2_.push_back(7);
  large_operand_2_.push_back(6);
  large_operand_2_.push_back(6);

  large_operand_3_.push_back(0);
  large_operand_3_.push_back(1);
  large_operand_3_.push_back(2);
  large_operand_3_.push_back(3);
  large_operand_3_.push_back(4);
  large_operand_3_.push_back(5);
  large_operand_3_.push_back(6);
  large_operand_3_.push_back(7);
  large_operand_3_.push_back(8);

  large_operand_4_.push_back(9);
  large_operand_4_.push_back(8);
  large_operand_4_.push_back(7);
  large_operand_4_.push_back(6);
  large_operand_4_.push_back(5);
  large_operand_4_.push_back(4);
  large_operand_4_.push_back(3);
}

} // namespace LinkedLists
} // namespace DataStructures
} // namespace UnitTests