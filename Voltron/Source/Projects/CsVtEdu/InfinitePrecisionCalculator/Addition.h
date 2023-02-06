#ifndef CSVTEDU_INFINITE_PRECISION_CALCULATOR_ADDITION_H
#define CSVTEDU_INFINITE_PRECISION_CALCULATOR_ADDITION_H

#include "DataStructures/LinkedLists/DoubleNode.h"
#include "DataStructures/LinkedLists/DoublyLinkedList.h"

#include <type_traits>

namespace CsVtEdu
{
namespace InfinitePrecisionCalculator
{

// cf. https://copyprogramming.com/tutorial/c-11-how-to-alias-a-function-duplicate
template <typename T>
using Node = DataStructures::LinkedLists::DoublyLinkedList<T>::Node;

//------------------------------------------------------------------------------
/// \details Assume that the resulting summation is to reside on operand 1.
//------------------------------------------------------------------------------
template <typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
void recursive_addition(
  Node<T>* operand_ptr_1,
  Node<T>* operand_ptr_2,
  Node<T>* previous_1,
  const T carry_value)
{
  if (operand_ptr_2 == nullptr)
  {
    if (operand_ptr_1 == nullptr)
    {
      if (carry_value != static_cast<T>(0))
      {
        Node<T>* new_digit {new Node<T>{carry_value}};
        previous_1->next_ = new_digit;
        new_digit->previous_ = previous_1;
      }

      return;
    }

    T new_carry_value {carry_value};

    if (new_carry_value != static_cast<T>(0))
    {
      const T sum {operand_ptr_1->retrieve() + carry_value};
      new_carry_value = sum / static_cast<T>(10);
      const T modulus {sum % static_cast<T>(10)};
      operand_ptr_1->value_ = modulus;
    }

    recursive_addition(
      operand_ptr_1->next_,
      operand_ptr_2,
      operand_ptr_1,
      new_carry_value);

    return;
  }

  if (operand_ptr_1 == nullptr)
  {
    const T sum {operand_ptr_2->retrieve() + carry_value};
    const T new_carry_value {sum / static_cast<T>(10)};
    const T modulus {sum % static_cast<T>(10)};

    Node<T>* new_digit {new Node<T>{modulus}};

    previous_1->next_ = new_digit;
    new_digit->previous_ = previous_1;

    recursive_addition(
      new_digit->next_,
      operand_ptr_2->next_,
      new_digit,
      new_carry_value);

    return;
  }

  const T sum {
    operand_ptr_1->retrieve() +
    operand_ptr_2->retrieve() +
    carry_value};

  const T new_carry_value {sum / static_cast<T>(10)};
  const T modulus {sum % static_cast<T>(10)};

  operand_ptr_1->value_ = modulus;

  recursive_addition(
    operand_ptr_1->next_,
    operand_ptr_2->next_,
    operand_ptr_1,
    new_carry_value);
}

/* g++ 11 Ubuntu cannot resolve forward declaration.
template <typename T, typename>
void recursive_addition(
  Node<T>* operand_ptr_1,
  Node<T>* operand_ptr_2,
  Node<T>* previous_1,
  const T carry_value);
*/

//------------------------------------------------------------------------------
/// \details
/// cf. https://en.cppreference.com/w/cpp/types/is_integral value is bool.
/// cf. https://en.cppreference.com/w/cpp/types/enable_if
/// template <bool B, class T = void>
/// struct enable_if
/// type - either T or no such member, depending on value of B.
/// template <bool B, class T = void>
/// using enable_if_t = typename enable_if<B, T>::type;
/// \return void, but operand1 becomes mutated with the resulting summation.
//------------------------------------------------------------------------------
template <typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
void addition(
  DataStructures::LinkedLists::DoublyLinkedList<T>& operand1,
  DataStructures::LinkedLists::DoublyLinkedList<T>& operand2)
{
  recursive_addition<T>(
    operand1.head(),
    operand2.head(),
    nullptr,
    static_cast<T>(0));
}

} // namespace InfinitePrecisionCalculator
} // namespace CsVtEdu

#endif // CSVTEDU_INFINITE_PRECISION_CALCULATOR_ADDITION_H