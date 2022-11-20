#ifndef CSVTEDU_INFINITE_PRECISION_CALCULATOR_MULTIPLICATION_H
#define CSVTEDU_INFINITE_PRECISION_CALCULATOR_MULTIPLICATION_H

#include "DataStructures/LinkedLists/DoubleNode.h"
#include "DataStructures/LinkedLists/DoublyLinkedList.h"

#include <type_traits>

namespace CsVtEdu
{
namespace InfinitePrecisionCalculator
{

template <typename T>
using Node = DataStructures::LinkedLists::DoublyLinkedList<T>::Node;

//------------------------------------------------------------------------------
/// cf. https://www.geeksforgeeks.org/multiply-two-numbers-represented-linked-lists-third-list/
//------------------------------------------------------------------------------
template <typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
void iterative_multiplication(
  const Node<T>* operand_ptr_1,
  const Node<T>* operand_ptr_2,
  Node<T>* result)
{
  Node<T>* current_result_ptr_1 {result};
  Node<T>* current_result_ptr_2 {nullptr};
  const Node<T>* current_operand_1_ptr {nullptr};
  const Node<T>* current_operand_2_ptr {operand_ptr_2};

  // Multiply each Node of second operand with first operand.
  while (current_operand_2_ptr)
  {
    T carry_value {static_cast<T>(0)};

    // Each time we start from the next of Node from which we had started the
    // last iteration.
    current_result_ptr_2 = current_result_ptr_1;

    current_operand_1_ptr = operand_ptr_1;

    while (current_operand_1_ptr)
    {
      T single_digit_result {
        current_operand_1_ptr->retrieve() *
        current_operand_2_ptr->retrieve() +
        carry_value};

      // Assign the product to the corresponding Node of the result.
      current_result_ptr_2->get_value() += single_digit_result % 10;

      // Now resultant Node itself can have more than 1 digit.
      carry_value = single_digit_result / 10 +
        current_result_ptr_2->retrieve() / 10;

      current_operand_1_ptr = current_operand_1_ptr->next();
      current_result_ptr_2 = current_result_ptr_2->next();
    }

    // If carry is remaining from last multiplication,
    if (carry_value > static_cast<T>(0))
    {
      current_result_ptr_2->get_value() += carry_value;
    }

    current_result_ptr_1 = current_result_ptr_1->next_;
    current_operand_2_ptr = current_operand_2_ptr->next_;
  }
}

} // namespace InfinitePrecisionCalculator
} // namespace CsVtEdu

#endif // CSVTEDU_INFINITE_PRECISION_CALCULATOR_MULTIPLICATION_H