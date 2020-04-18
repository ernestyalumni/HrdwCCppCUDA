//------------------------------------------------------------------------------
/// \file ReaderMonad.h
/// \author Ernest Yeung
/// \brief
///-----------------------------------------------------------------------------
#ifndef CATEGORIES_MONADS_READER_MONAD_H
#define CATEGORIES_MONADS_READER_MONAD_H

#include <string>
#include <type_traits>

namespace Categories
{
namespace Monads
{
namespace ReaderMonad
{

// What Haskell/Functional Programming calls "return"
template <typename X, typename E>
auto unit(const X& x)
{
  return [&x](const E& e)
  {
    return x;
  };
}

// Lambda version of unit, using return
auto return_ = [](auto x)
{
  return [x](auto e)
  {
    return x;
  };
};

template <typename E>
E ask(const E& environment)
{
  return environment;
}

/*
// Haskell/Functional Programming Programmers may simply call this the
// ReaderMonad when it is actually only the endomorphism.
template <typename X, typename TX, typename E>
X evaluate_endomorphism(TX tx, const E& environment)
{
  return tx(environment);
}
*/

auto runReader = [](auto ra, auto environment)
{
  return ra(environment);
};

auto bind = [](auto ra, auto f)
{
  return [ra, f](auto environment)
  {
    return runReader(f(runReader(ra, environment)), environment);
  };
};

} // namespace ReaderMonad
} // namespace Monads
} // namespace Categories

#endif // CATEGORIES_MONADS_READER_MONAD_H