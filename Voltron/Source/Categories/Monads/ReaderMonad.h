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
auto unit(const X x)
{
  return [x](const E& e)
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

// TODO: Make this work

template <
  typename E,
  typename EToX
  >
auto apply_morphism(EToX e_to_x, const E& environment)
{
  return e_to_x(environment);
}

template <
  typename E,
  typename EToX,
  typename Morphism
  >
auto bind(EToX e_to_x, Morphism f)
{
  return [e_to_x, f](const E& environment)
  {
    return apply_morphism<E, EToX>(
      f(apply_morphism<E, EToX>(e_to_x, environment)),
      environment);
  };
}

// "runReader" as Haskell/Functional Programming programmers would call it.
auto map_morphism = [](auto e_to_x, auto environment)
{
  return e_to_x(environment);
};

auto bind_ = [](auto e_to_x, auto f)
{
  return [e_to_x, f](auto environment)
  {
    return map_morphism(f(map_morphism(e_to_x, environment)), environment);
  };
};


} // namespace ReaderMonad
} // namespace Monads
} // namespace Categories

#endif // CATEGORIES_MONADS_READER_MONAD_H