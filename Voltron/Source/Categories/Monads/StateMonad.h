//------------------------------------------------------------------------------
/// \file StateMonad.h
/// \author Ernest Yeung
/// \brief
///-----------------------------------------------------------------------------
#ifndef CATEGORIES_MONADS_STATE_MONAD_H
#define CATEGORIES_MONADS_STATE_MONAD_H

#include <utility> // std::forward, std::pair

namespace Categories
{
namespace Monads
{
namespace StateMonad
{

//------------------------------------------------------------------------------
/// \class Unit
/// \ref https://en.cppreference.com/w/cpp/utility/forward
/// \details 1) Forwards lvalues as either lvalues or as rvalues, depending on T
/// Forwards rvalues as rvalues and prohibits forwarding of rvalues as lvalues.
//------------------------------------------------------------------------------
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

    // W is the mutable state type.
    template <typename W>
    std::pair<W, X> operator()(W&& state)
    {
      return std::make_pair<W, X>(
        std::forward<W>(state),
        std::forward<X>(input_));
    }

  private:

    X input_;
};

template <typename MorphismTY, typename MorphismTZ>
auto compose(MorphismTZ g, MorphismTY f)
{
  return [g, f](auto x)
  {
    return [g, f, x](auto s)
    {
      auto fxs = f(x)(s);
      return g(fxs.second)(fxs.first);
    };
  };
}

template <typename MorphismTY, typename MorphismTZ>
class Compose
{
  public:

    Compose(MorphismTZ g, MorphismTY f):
      g_{g},
      f_{f}
    {}

    template <typename X>
    class ComposedMorphisms
    {
      public:

        ComposedMorphisms(MorphismTZ g, MorphismTY f, X& input):
          g_{g},
          f_{f},
          input_{std::forward<X>(input)}
        {}

        ComposedMorphisms(MorphismTZ g, MorphismTY f, X&& input):
          g_{g},
          f_{f},
          input_{std::forward<X>(input)}
        {}

        template <typename State>
        auto operator()(State&& state)
        {
          auto fxs = f_(input_)(std::forward<State>(state));
          return g_(fxs.second)(fxs.first);
        }

        template <typename State>
        auto operator()(State& state)
        {
          auto fxs = f_(input_)(std::forward<State>(state));
          return g_(fxs.second)(fxs.first);
        }

      private:

        MorphismTZ g_;
        MorphismTY f_;
        X input_;
    };

    template <typename X>
    ComposedMorphisms<X> operator()(X input)
    {
      return ComposedMorphisms<X>{g_, f_, input};
    }

  private:

    MorphismTZ g_;
    MorphismTY f_;
};

namespace AsLambdas
{

auto unit = [](auto x)
{
  return [x](auto state)
  {
    return std::pair{state, x};
  };
};

auto bind = [](auto g, auto f)
{
  return [g, f](auto x)
  {
    return [g, f, x](auto s)
    {
      auto fxs = f(x)(s);
      return g(fxs.second)(fxs.first);
    };
  };
};

} // namespace AsLambdas

} // namespace StateMonad
} // namespace Monads
} // namespace Categories

#endif // CATEGORIES_MONADS_STATE_MONAD_H