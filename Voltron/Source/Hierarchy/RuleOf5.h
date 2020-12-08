//------------------------------------------------------------------------------
/// \file RuleOf5.h
/// \author Ernest Yeung
/// \brief Classes demonstrating copy and move constructors, assignments,
///   destructors.
///-----------------------------------------------------------------------------
#ifndef HIERARCHY_RULE_OF_5_H
#define HIERARCHY_RULE_OF_5_H

#include <string>

namespace Hierarchy
{
namespace RuleOf5
{

struct A
{
  int n_;

  // Constructors.
  A(int n = 1);
  A(const A& a);
};

class RuleOf5Object
{
  public:

    RuleOf5Object();

    RuleOf5Object(const std::string& input_s, const int input_int);

    // Copy constructor
    RuleOf5Object(const RuleOf5Object&);

    //--------------------------------------------------------------------------
    /// \brief Copy assignment, when copy-and-swap idiom isn't used.
    /// TODO: Add more to data member such as a unique_ptr to int array, for
    /// demonstration.
    /// cf. https://en.cppreference.com/w/cpp/language/copy_assignment
    //--------------------------------------------------------------------------
    RuleOf5Object& operator=(const RuleOf5Object&);

    // Move constructor
    RuleOf5Object(RuleOf5Object&&);

    // Move assignment.
    RuleOf5Object& operator=(RuleOf5Object&&);

    // Accessors

    std::string s_data() const
    {
      return s_data_;
    }

    int int_data() const
    {
      return int_data_;
    }

    int destruction_flag() const
    {
      return destruction_flag_;
    }

    int copy_constructor_counter() const
    {
      return copy_ctr_counter_;
    }

    int copy_assign_counter() const
    {
      return copy_assign_counter_;
    }

    int move_constructor_counter() const
    {
      return move_ctr_counter_;
    }

    int move_assign_counter() const
    {
      return move_assign_counter_;
    }

  private:

    std::string s_data_;
    int int_data_;
    int destruction_flag_;
    int copy_ctr_counter_;
    int copy_assign_counter_;
    int move_ctr_counter_;
    int move_assign_counter_;
};

} // namespace RuleOf5
} // namespace Hierarchy

#endif // HIERARCHY_RULE_OF_5_H