//------------------------------------------------------------------------------
/// \file OptionalMonad.h
/// \author Ernest Yeung
/// \brief
/// \ref Ivan Čukić, Functional Programming in C++,  Manning Publications;
/// 1st edition (November 19, 2018). ISBN-13: 978-1617293818
///-----------------------------------------------------------------------------
#ifndef CATEGORIES_MONADS_OPTIONAL_MONAD_H
#define CATEGORIES_MONADS_OPTIONAL_MONAD_H

#include <optional>

namespace Categories
{
namespace Monads
{
namespace OptionalMonad
{

// Note that the endomorphism T : X \to T(X) maps any type T to the type T(X),
// which in this case is T(X) = std::optional<T>

// In this case for the OptionalMonad, the endofunctor T is std::optional.
// The type X is an object in the objects for the syntactic category. The
// objects for this category are the types for this program.
// What Čukić calls transform is the morphism map for the endofunctor.
// Indeed, if morphism f : X \to T(Y), then the morphism map for endomorphism T
// Tf: T(X) \to T(Z) corresponds to this function. Z = T(Y) in this case.
// \param std::optional<X>& object_element is of type T(X)
// object_element.value() is of type X.
template <typename X, typename Morphism>
auto endomorphism_morphism_map(
  const std::optional<X>& object_element,
  Morphism f)
  -> decltype(std::make_optional(f(object_element.value())))
{
  if (object_element)
  {
    return std::make_optional(f(object_element.value()));
  }
  else
  {
    return {};
  }
}

// multiplication_component is from Category Theory being the component at X
// of the multiplication natural transformation.
// It is called "join" in Haskell/Functional Programming.
// In this case for the OptionalMonad, the endofunctor T is std::optional.
// The type X is an object in the objects for the syntactic category, the
// types for this program.
// Indeed, since \mu_X : T^2(X) \to T(X), \mu_X : x \mapsto y, with x \in T^2(X)
// and y \in T(X).
template <typename X>
std::optional<X> multiplication_component(
  const std::optional<std::optional<X>>& object_element)
{
  if (object_element)
  {
    // Return an instance of type T(X)
    return object_element.value();
  }
  else
  {
    return {};
  }
}

} // namespace OptionalMonad
} // namespace Monads
} // namespace Categories

#endif // CATEGORIES_MONADS_OPTIONAL_MONAD_H