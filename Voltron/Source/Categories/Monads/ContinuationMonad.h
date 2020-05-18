//------------------------------------------------------------------------------
/// \file ContinuationMonad.h
/// \author Ernest Yeung
/// \brief
///-----------------------------------------------------------------------------
#ifndef CATEGORIES_MONADS_CONTINUATION_MONAD_H
#define CATEGORIES_MONADS_CONTINUATION_MONAD_H

#include <utility>

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

template <typename X>
class Unit
{
  public:

    Unit(X& input):
      input_{std::forward<X>(input)}
    {}

    Unit(X&& input):
      input_{std::forward<X>(input)}
    {}

    X input() const
    {
      return input_;
    }

    // InternalHom maps X to Y
    template <typename InternalHom>
    auto operator()(InternalHom f)
    {
      return f(input_);
    }

  private:

    X input_;
};

// Morphism \equiv Mor{C} \ni f : X \to T(Y) = [[Y, Z], Z]
template <typename Morphism>
class Bind
{
  public:

    Bind(Morphism f):
      f_{f}
    {}

    // \mu_Y \circ Tf
    template <typename TX>
    class MultiplicationComposedFunctor
    {
      public:

        MultiplicationComposedFunctor(Morphism f, TX tx):
          f_{f},
          tx_{tx}
        {}

        template <typename InternalHom>
        auto operator()(InternalHom k)
        {
          auto morphism = [&f_, &k](auto x)
          {
            return (f_(x))(k);
          }

          return tx_(morphism);
        }

      private:

        Morphism f_;
        TX tx_;
    };

    template <typename TX>
    MultiplicationComposedFunctor<TX> operator()(TX tx)
    {
      return MultiplicationComposedFunctor<TX>{f_, tx};
    }

  private:

    Morphism f_;
};

template <typename X>
class Endomorphism
{
  public:

    auto operator()(X& x)
    {
      return [&x](auto f)
      {
        return f(x);
      };
    }

    auto operator()(X&& x)
    {
      return [x](auto f)
      {
        return f(x);
      };
    }
};

/*
template <typename Morphism>
class CallCC
{
  public:

    CallCC(Morphism f):
      f_{f}
    {}

    template <typename InternalHom>
    class CurrentContinuation
    {
      public:

        CurrentContinuation(Morphism f, InternalHom k):
          f_{f},
          k_{k}
        {}

        template <typename X>
        class Call
        {
          public:

            Call(Morphism f, InternalHom k, X x):
              f_{f},
              k_{k},
              x_{x}
            {}

            {

            }

          private:

            Morphism f_;
            InternalHom k_;
            X x_;
        };


        auto operator()()
        {
          auto morphism = [&k_](auto x)
          {
            return [&k_, &&x](auto)
            {
              return k
            }
          }
        }

      private:

        Morphism f_;
        InternalHom k_;
    };

    template <typename InternalHom>
    CurrentContinuation<InternalHom> operator()(InternalHom k)
    {
      return CurrentContinuation<InternalHom>{f_, k};
    }

  private:

    Morphism f_;
};
*/

template <typename X>
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

// ca \equiv c \in T(X) = [[X, Z], Z], f \in X \to T(Y) = [[Y, Z], Z]
auto bind = [](auto ca, auto f)
{
  // continuation \equiv k \in [Y, Z]
  // (\mu_Y \circ T)(f)(c)(k) \in Z
  // (\mu_Y \circ T)(f)(c) \in T(Y) =[[Y, Z], Z]
  return [ca, f](auto continuation)
  {
    return
      // ca \equiv c \in T(X) = [[X, Z], Z]
      ca(
        [f, continuation](auto x)
        {
          // continuation \equiv k
          // x \to f(x)(k) \equiv f(x)(continuation)
          return runContinuation(f(x), continuation);
        });
  };
};

// call_cc f = Cont $ \k -> runCont (f (\t -> Cont $ \_ -> k a)) k
// where runCont is runContinuation
// cf. https://github.com/Iasi-C-CPP-Developers-Meetup/presentations-code-samples/blob/master/radugeorge/monadic/main.cpp
auto call_cc = [](auto f)
{
  return 
    // continuation can be written as k and k \in [X, Z]
    [f](auto continuation)
    {
      return runContinuation(
        f(
          [continuation](auto x)
          {
            return [continuation, x](auto)
            {
              return continuation(x);
            };
          }),
        continuation);
    };
};

} // namespace AsLambdas

} // namespace ContinuationMonad
} // namespace Monads
} // namespace Categories

#endif // CATEGORIES_MONADS_CONTINUATION_MONAD_H