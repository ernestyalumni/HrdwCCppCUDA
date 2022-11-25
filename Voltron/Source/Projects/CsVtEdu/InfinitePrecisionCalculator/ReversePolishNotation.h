#ifndef CSVTEDU_INFINITE_PRECISION_CALCULATOR_REVERSE_POLISH_NOTATION_H
#define CSVTEDU_INFINITE_PRECISION_CALCULATOR_REVERSE_POLISH_NOTATION_H

#include "Addition.h"
#include "DataStructures/LinkedLists/DoubleNode.h"
#include "DataStructures/LinkedLists/DoublyLinkedList.h"
#include "DataStructures/Stacks/DynamicStack.h"
#include "Exponentiation.h"
#include "Multiplication.h"
#include "ParseTextFile.h"

#include <cctype>
#include <type_traits>

namespace CsVtEdu
{
namespace InfinitePrecisionCalculator
{

template <typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
class ReversePolishNotation
{
  public:
  
    template <typename U>
    using DynamicStack = DataStructures::Stacks::AsHierarchy::DynamicStack<U>;

    template <typename U>
    using DoublyLinkedList = DataStructures::LinkedLists::DoublyLinkedList<U>;

    ReversePolishNotation():
      operands_stack_{}
    {}

    ~ReversePolishNotation() = default;

    //--------------------------------------------------------------------------
    /// \return True on successful processing; return false when an error
    /// occurs. This happens for the stack not having enough operands.
    //--------------------------------------------------------------------------
    bool process_input(const std::string& input)
    {
      if (input.empty())
      {
        return false;
      }

      if (std::isdigit(input[0]))
      {
        DoublyLinkedList<T> operand {};
        for (char digit_character : input)
        {
          operand.push_front(static_cast<T>(digit_character - '0'));
        }

        operands_stack_.push(operand);

        return true;
      }

      if (operands_stack_.size() < 2)
      {
        return false;
      }

      if (input[0] == '+')
      {
        DoublyLinkedList<T> operand_1 {operands_stack_.pop()};
        DoublyLinkedList<T> operand_2 {operands_stack_.pop()};
        addition<T>(operand_1, operand_2);
        operands_stack_.push(operand_1);
        return true;
      }

      if (input[0] == '*')
      {
        DoublyLinkedList<T> operand_1 {operands_stack_.pop()};
        DoublyLinkedList<T> operand_2 {operands_stack_.pop()};
        DoublyLinkedList<T> result {
          operand_1.size() + operand_2.size(),
          static_cast<T>(0)};

        tail_recursive_multiplication<T>(
          operand_1.head(),
          operand_2.head(),
          result.head());

        operands_stack_.push(result);
        return true;
      }

      if (input[0] == '^')
      {
        DoublyLinkedList<T> operand_1 {operands_stack_.pop()};
        DoublyLinkedList<T> operand_2 {operands_stack_.pop()};
        const int exponent {convert_to_int<T>(operand_2)};
        DoublyLinkedList<T> result {2 * operand_1.size(), static_cast<T>(0)};

        exponentiate<T>(operand_1, exponent, result);

        operands_stack_.push(result);
      }

      return false;
    }

    DoublyLinkedList<int> top() const
    {
      return operands_stack_.top();
    }

  protected:

    // TODO: consider a pop that would pass the reference to the underlying object.
    DoublyLinkedList<int> pop()
    {
      return operands_stack_.pop();
    }

  private:

    DynamicStack<DoublyLinkedList<int>> operands_stack_;
};

} // namespace InfinitePrecisionCalculator
} // namespace CsVtEdu

#endif // CSVTEDU_INFINITE_PRECISION_CALCULATOR_REVERSE_POLISH_NOTATION_H