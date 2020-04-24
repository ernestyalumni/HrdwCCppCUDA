//------------------------------------------------------------------------------
/// \file ContinuationMonad.h
/// \author Ernest Yeung
/// \brief
///-----------------------------------------------------------------------------
#ifndef CATEGORIES_MONADS_CONTINUATION_MONAD_H
#define CATEGORIES_MONADS_CONTINUATION_MONAD_H

namespace Categories
{
namespace Monads
{
namespace ContinuationMonad
{

template <typename T>
T evaluate(const T& value)
{
  return value;
}

// The endomorphism T
template <typename InternalHom, typename X>
auto apply_endomorphism(InternalHom internal_hom, const X& x)
{
  return internal_hom(x);
}

// cf. https://ncatlab.org/nlab/show/internal+hom
// Internal hom is in category theory what function types are in type theory.

template<typename X>
auto unit(X x)
{
  return [x](auto internal_hom)
  {
    return internal_hom(x);
  };
}

/* overload is ambiguous for functions*/
/*
template<typename X>
auto unit(X& x)
{
  return [&x](auto internal_hom)
  {
    return internal_hom(x);
  };
}
*/

namespace AsLambdas
{

auto eval = [](auto value)
{
  return value;
};

auto runContinuation = [](auto ca, auto continuation)
{
  return ca(continuation);
};

// cf. https://github.com/Iasi-C-CPP-Developers-Meetup/presentations-code-samples/tree/master/radugeorge
auto return_ = [](auto x)
{
  return [x](auto continuation)
  {
    return continuation(x);
  };
};

auto bind = [](auto ca, auto f)
{
  return [ca, f](auto continuation)
  {
    return ca([f, continuation](auto x)
    {
      return runContinuation(f(x), continuation);
    });
  };
};

} // namespace AsLambdas

} // namespace ContinuationMonad
} // namespace Monads
} // namespace Categories

#endif // CATEGORIES_MONADS_CONTINUATION_MONAD_H