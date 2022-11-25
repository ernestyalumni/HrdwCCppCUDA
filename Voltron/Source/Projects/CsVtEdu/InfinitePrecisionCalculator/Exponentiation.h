#ifndef CSVTEDU_INFINITE_PRECISION_CALCULATOR_EXPONENTIATION_H
#define CSVTEDU_INFINITE_PRECISION_CALCULATOR_EXPONENTIATION_H

#include "DataStructures/LinkedLists/DoublyLinkedList.h"
#include "Multiplication.h"

#include <type_traits>

namespace CsVtEdu
{
namespace InfinitePrecisionCalculator
{

template <typename T>
using DoublyLinkedList = DataStructures::LinkedLists::DoublyLinkedList<T>;

template <typename T>
using Node = DataStructures::LinkedLists::DoublyLinkedList<T>::Node;

template <typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
void exponentiate(
  const DoublyLinkedList<T>& base,
  const int exponent,
  DoublyLinkedList<T>& result)
{
  if (exponent <= 0)
  {
    result.clear();
    result.push_front(static_cast<T>(0));
    return;
  }

  if (exponent == 1)
  {
    result = base;
    return;
  }

  DoublyLinkedList<T> previous_result {base};

  recursive_exponentiation<T>(base, exponent, previous_result, result);
}

template <typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
void recursive_exponentiation(
  const DoublyLinkedList<T>& base,
  const int exponent,
  DoublyLinkedList<T>& previous_result,
  DoublyLinkedList<T>& result)
{
  iterative_multiplication<T>(
    previous_result.head(),
    base.head(),
    result.head());

  if (exponent <= 2)
  {
    return;
  }

  previous_result = result;
  result = DoublyLinkedList<T>{
    previous_result.size() + base.size(),
    static_cast<T>(0)};

  recursive_exponentiation(base, exponent - 1, previous_result, result);
}

} // namespace InfinitePrecisionCalculator
} // namespace CsVtEdu

#endif // CSVTEDU_INFINITE_PRECISION_CALCULATOR_EXPONENTIATION_H