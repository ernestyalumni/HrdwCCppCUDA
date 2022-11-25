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

    // cf. https://en.wikipedia.org/wiki/Multiplication_algorithm
    // Each time we start from the next of Node from which we had started the
    // last iteration.
    // This steps corresponds to how in arithmetic for multiplication, you have
    // to shift in position with the digit position of the multiplier. See also
    // the final step at end of this iteration.
    // current_result_ptr_1 tracks the position from which we had shifted to.
    // current_result_ptr_2 then, from that position, will move along the new
    // result from multiplication.
    current_result_ptr_2 = current_result_ptr_1;

    current_operand_1_ptr = operand_ptr_1;

    while (current_operand_1_ptr)
    {
      const T single_digit_result {
        current_operand_1_ptr->retrieve() *
        current_operand_2_ptr->retrieve() +
        carry_value};

      // Assign the product to the corresponding Node of the result.
      current_result_ptr_2->get_value() += single_digit_result % 10;

      // Now resultant Node itself can have more than 1 digit.
      carry_value = single_digit_result / 10 +
        current_result_ptr_2->retrieve() / 10;

      // Now that carry value has been calculated, keep only the modulus in the
      // result digit position.
      current_result_ptr_2->get_value() = current_result_ptr_2->retrieve() % 10;

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

namespace Details
{

template <typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
void tail_recursive_multiply(
  const Node<T>* operand_ptr_1,
  const Node<T>* current_operand_1_ptr,
  const Node<T>* current_operand_2_ptr,
  Node<T>* current_result_ptr_1,
  Node<T>* current_result_ptr_2)
{
  if (current_operand_2_ptr == nullptr)
  {
    return;
  }
  T carry_value {static_cast<T>(0)};

  current_result_ptr_2 = current_result_ptr_1;
  current_operand_1_ptr = operand_ptr_1;

  while (current_operand_1_ptr)
  {
    T single_digit_result {
      current_operand_1_ptr->retrieve() *
      current_operand_2_ptr->retrieve() +
      carry_value};
    current_result_ptr_2->get_value() += single_digit_result % 10;
    carry_value = single_digit_result / 10 +
      current_result_ptr_2->retrieve() / 10;

    current_result_ptr_2->get_value() = current_result_ptr_2->retrieve() % 10;

    current_operand_1_ptr = current_operand_1_ptr->next();
    current_result_ptr_2 = current_result_ptr_2->next();
  }

  if (carry_value > static_cast<T>(0))
  {
    current_result_ptr_2->get_value() += carry_value;
  }

  current_result_ptr_1 = current_result_ptr_1->next_;
  current_operand_2_ptr = current_operand_2_ptr->next_;

  tail_recursive_multiply<T>(
    operand_ptr_1,
    current_operand_1_ptr,
    current_operand_2_ptr,
    current_result_ptr_1,
    current_result_ptr_2);
}

} // namespace Details

//------------------------------------------------------------------------------
/// cf. https://stackoverflow.com/questions/22764142/recursion-vs-tail-recursion
/// A loop has:
/// 1. header, and processing and variables before the loop
/// 2. Stop condition for loop
/// 3. actual loop body;
/// 4. tail - what happens after the loop and return result;
/// for_iterative(params)
/// {
///   header
///   while (condition)
///   {
///     loop_body
///   }
/// }
///
/// for_recursive(params)
/// {
///   header
///   return foo_recursion(params, header_vars)
/// }
///
/// for_recursion(params, header_vars)
/// {
///   if (!condition)
///   {
///      return tail
///   }
///   loop_body
///   return foo_recursion(params, modified_header_vars)
/// }
//------------------------------------------------------------------------------

template <typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
void tail_recursive_multiplication(
  const Node<T>* operand_ptr_1,
  const Node<T>* operand_ptr_2,
  Node<T>* result)
{
  Node<T>* current_result_ptr_1 {result};
  Node<T>* current_result_ptr_2 {nullptr};
  const Node<T>* current_operand_1_ptr {nullptr};
  const Node<T>* current_operand_2_ptr {operand_ptr_2};

  Details::tail_recursive_multiply<T>(
    operand_ptr_1,
    current_operand_1_ptr,
    current_operand_2_ptr,
    current_result_ptr_1,
    current_result_ptr_2);
}

} // namespace InfinitePrecisionCalculator
} // namespace CsVtEdu

#endif // CSVTEDU_INFINITE_PRECISION_CALCULATOR_MULTIPLICATION_H