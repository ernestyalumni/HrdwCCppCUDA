//------------------------------------------------------------------------------
/// \file RuleOf5.cpp
/// \author Ernest Yeung
/// \brief Classes demonstrating copy and move constructors, assignments,
///   destructors.
//------------------------------------------------------------------------------
#include "RuleOf5.h"

#include <string>
#include <utility> // std::exchange, std::move

//------------------------------------------------------------------------------
/// \ref https://en.cppreference.com/w/cpp/utility/exchange
/// \details
/// template <class T, class U = T>
/// T exchange(T& obj, U&& new_value);
/// 
/// Replaces value of obj with new_value and returns old value of obj.
//------------------------------------------------------------------------------
using std::exchange;
//------------------------------------------------------------------------------
/// \ref https://en.cppreference.com/w/cpp/utility/move
/// std::move used to indicate object may be "moved from"; particularly,
/// produces a xvalue expression that identifies its argument; it's exactly
/// equivalent to static_cast to rvalue reference type.
///
/// xvalue (eXpiring value) is a glvalue that denotes object whose resources can
/// be reused. glvalue ("generalized" lvalue) is an expression whose evaluation
/// determines the identity of an object, or function.
//------------------------------------------------------------------------------
using std::move;
using std::string;

namespace Hierarchy
{

namespace RuleOf5
{

A::A(int n) :
  n_{n}
{}

// User-defined copy constructor.
A::A(const A& a) :
  n_{a.n_}
{}

RuleOf5Object::RuleOf5Object():
  s_data_{},
  int_data_{0},
  destruction_flag_{0},
  copy_ctr_counter_{0},
  copy_assign_counter_{0},
  move_ctr_counter_{0},
  move_assign_counter_{0}
{}

RuleOf5Object::RuleOf5Object(const string& input_s, const int input_int):
  s_data_{input_s},
  int_data_{input_int},
  destruction_flag_{0},
  copy_ctr_counter_{0},
  copy_assign_counter_{0},
  move_ctr_counter_{0},
  move_assign_counter_{0}
{}

// Copy constructor; user-defined
RuleOf5Object::RuleOf5Object(const RuleOf5Object& object):
  s_data_{object.s_data_},
  int_data_{object.int_data_},
  destruction_flag_{object.destruction_flag_},
  copy_ctr_counter_{object.copy_ctr_counter_ + 1},
  copy_assign_counter_{object.copy_assign_counter_},
  move_ctr_counter_{object.move_ctr_counter_},
  move_assign_counter_{object.move_assign_counter_}  
{}

// Copy assignment, non-copy-and-swap assignment
RuleOf5Object& RuleOf5Object::operator=(const RuleOf5Object& object)
  // Only ctors take member initializers
  //s_data_{move(object.s_data_)},
  // ...
{
  s_data_ = object.s_data_;
  int_data_ = object.int_data_;
  destruction_flag_ = object.destruction_flag_;
  copy_ctr_counter_ = object.copy_ctr_counter_;
  copy_assign_counter_ = object.copy_assign_counter_ + 1,
  move_ctr_counter_ = object.move_ctr_counter_;
  move_assign_counter_ = object.move_assign_counter_;

  return *this;
}

// Move constructor
RuleOf5Object::RuleOf5Object(RuleOf5Object&& object):
  s_data_{move(object.s_data_)},
  int_data_{exchange(object.int_data_, 0)},
  destruction_flag_{move(object.destruction_flag_)},
  copy_ctr_counter_{move(object.copy_ctr_counter_)},
  copy_assign_counter_{move(object.copy_assign_counter_)},
  move_ctr_counter_{++object.move_ctr_counter_},
  move_assign_counter_{move(object.move_assign_counter_)}
{}

// Move assignment.
RuleOf5Object& RuleOf5Object::operator=(RuleOf5Object&& object)
  // Only ctors take member initializers
  //s_data_{move(object.s_data_)},
  // ...
{
  s_data_ = move(object.s_data_);
  int_data_ = move(object.int_data_);
  destruction_flag_ = move(object.destruction_flag_);
  copy_ctr_counter_ = move(object.copy_ctr_counter_);
  copy_assign_counter_ = move(object.copy_assign_counter_);
  move_ctr_counter_ = move(object.move_ctr_counter_),
  move_assign_counter_ = ++object.move_assign_counter_;

  return *this;
}


} // namespace RuleOf5
} // namespace Hierarchy
