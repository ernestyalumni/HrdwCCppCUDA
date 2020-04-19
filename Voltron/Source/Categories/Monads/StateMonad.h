//------------------------------------------------------------------------------
/// \file StateMonad.h
/// \author Ernest Yeung
/// \brief
///-----------------------------------------------------------------------------
#ifndef CATEGORIES_MONADS_STATE_MONAD_H
#define CATEGORIES_MONADS_STATE_MONAD_H

#include <utility>

namespace Categories
{
namespace Monads
{
namespace StateMonad
{

// This is what Haskell/Functional Programming calls an Algebraic Data Type,
// the product type. Mathematically, this is an element of the object
// T(X) = X \times S, where X \times S is the Cartesian product of X and S
// for types X, and states S.
template <typename X, typename S>
class StateObject
{
  public:

    StateObject(const X& x, const S s):
      inputs_{x},
      state_{s}
    {}

    explicit StateObject(const X& x):
      inputs_{x},
      state_{}
    {}

    X inputs() const
    {
      return inputs_;
    }

    S state() const
    {
      return state_;
    }

  private:

    X inputs_;
    S state_;
};

template <typename X, typename S>
class StateObjectAsPair : public std::pair<X, S>
{
  public:

    using std::pair<X, S>::pair;

//    explicit StateObject(const X& x):
//{}
};

template <typename X, typename S>
StateObject<X, S> unit(const X& x)
{
  return StateObject<X, S>{x};
}

template <typename X, typename S>
StateObject<X, S> multiplication_component(
  const std::pair<StateObject<X, S>, S>& ttx)
{
  return ttx.first;
}

} // namespace StateMonad
} // namespace Monads
} // namespace Categories

#endif // CATEGORIES_MONADS_STATE_MONAD_H