//------------------------------------------------------------------------------
/// \file RuleOf5.h
/// \author Ernest Yeung
/// \brief Classes demonstrating copy and move constructors, assignments,
///   destructors.
///-----------------------------------------------------------------------------
#ifndef _HIERARCHY_RULE_OF_5_H_
#define _HIERARCHY_RULE_OF_5_H_

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

} // namespace RuleOf5
} // namespace Hierarchy

#endif // _HIERARCHY_RULE_OF_5_H_