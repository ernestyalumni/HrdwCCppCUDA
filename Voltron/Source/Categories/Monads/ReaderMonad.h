//------------------------------------------------------------------------------
/// \file ReaderMonad.h
/// \author Ernest Yeung
/// \brief Reader Monad
/// \ref https://www.slideshare.net/ovidiuf/monadic-computations-in-c14
///-----------------------------------------------------------------------------
#ifndef CATEGORIES_MONADS_READER_MONAD_H
#define CATEGORIES_MONADS_READER_MONAD_H

#include <string>
#include <type_traits>
#include <utility> // std::forward

namespace Categories
{
namespace Monads
{
namespace ReaderMonad
{

template <typename X>
class Unit
{
  public:

    Unit(X& x):
      x_{std::forward<X>(x)}
    {}

    Unit(X&& x):
      x_{std::forward<X>(x)}
    {}

    X x() const
    {
      return x_;
    }

    // E is the environment type.
    template <typename E>
    X operator()(E&& environment)
    {
      return x_;
    }

  private:

    X x_;
};

template <typename Morphism>
class MultiplicationEndomorphismComposed
{
  public:

    MultiplicationEndomorphismComposed(Morphism f):
      f_{f}
    {}

    template <typename InternalHom>
    class MorphismFunctor
    {
      public:

        MorphismFunctor(Morphism f, InternalHom r):
          f_{f},
          r_{r}
        {}

        template <typename E>
        auto operator()(E&& environment)
        {
          return (f_(r_(std::forward<E>(environment))))(
            std::forward<E>(environment));
        }

      private:

        Morphism f_;
        InternalHom r_;
    };

    template <typename InternalHom>
    MorphismFunctor<InternalHom> operator()(InternalHom r)
    {
      return MorphismFunctor<InternalHom>{f_, r};
    }

    // Doesn't work.
    /*
    template <typename InternalHom, typename E>
    auto operator()(InternalHom r)
    {
      // Capture of non-variable f_
      return [f_, &r](E&& environment)
      {
        return (f_(r(std::forward<E>(environment))))(
          std::forward<E>(environment));
      };
    }
    */

  private:

    Morphism f_;
};

template <typename Endomorphism>
class Local
{
  public:

    Local(Endomorphism e):
      e_{e}
    {}

    template <typename InternalHom>
    class ComposeInternalHom
    {
      public:

        ComposeInternalHom(InternalHom f):
          f_{f}
        {}

        template <typename E>
        auto operator()(E&& environment)
        {
          return f_(e_(std::forward<E>(environment)));
        }

      private:

        InternalHom f_;
    };

    template <typename InternalHom>
    ComposeInternalHom<InternalHom> operator()(InternalHom f)
    {
      return ComposeInternalHom<InternalHom>{f};
    }

    // Doesn't work
    /*
    template <typename Morphism, typename E>
    auto operator()(Morphism f)
    {
      // error: capture of non-variable
      return [&e_, &f](E&& environment)
      {
        return f(e(std::forward<E>(environment)));
      };
    }
    */

  private:

    Endomorphism e_;
};


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