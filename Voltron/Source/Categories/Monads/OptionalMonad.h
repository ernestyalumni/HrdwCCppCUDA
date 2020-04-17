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

// In this case for the OptionalMonad, the endofunctor T is std::optional.
// The type X is an object in the objects for the syntactic category. The
// objects for this category are the types for this program.
// What Čukić calls transform is the morphism map for the endofunctor.
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
template <typename X>
std::optional<X> multiplication_component(
  const std::optional<std::optional<X>>& object_element)
{
  if (object_element)
  {
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