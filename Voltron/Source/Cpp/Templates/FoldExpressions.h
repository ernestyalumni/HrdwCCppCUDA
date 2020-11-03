//------------------------------------------------------------------------------
/// \file FoldExpressions.h
/// \author 
/// \brief 
/// \ref C++17 The Complete Guide by Nicolai M. Josuttis. (2019)
///-----------------------------------------------------------------------------
#ifndef CPP_TEMPLATES_FOLD_EXPRESSIONS_H
#define CPP_TEMPLATES_FOLD_EXPRESSIONS_H

namespace Cpp
{
namespace Templates
{
namespace FoldExpressions
{

template <typename... T>
auto fold_sum(T... args)
{
  return (... + args); // ((arg1 + arg2) + arg3) ...
}

} // FoldExpressions
} // namespace Templates
} // namespace Cpp

#endif // CPP_TEMPLATES_FOLD_EXPRESSIONS_H