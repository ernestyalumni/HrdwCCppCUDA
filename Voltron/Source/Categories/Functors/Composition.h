//------------------------------------------------------------------------------
/// \file Composition.h
/// \author Ernest Yeung
/// \brief Composition law for Functors.
///-----------------------------------------------------------------------------
#ifndef _CATEGORIES_COMPOSITION_H_
#define _CATEGORIES_COMPOSITION_H_

namespace Categories
{
namespace Functors
{

//------------------------------------------------------------------------------
/// \class Composition
/// \brief Composition law for Functors
/// \details 
/// \ref https://gist.github.com/splinterofchaos/3994038
//------------------------------------------------------------------------------

template <class F, class ... G>
struct Composition;

template <class F, class G>
struct Composition<F, G>
{
  F f_;
  G g_;

  template <class H, class I>
  constexpr Composition(H&& h, I&& i) :
    f_{std::forward<H>(h)},
    g_{std::forward<I>(i)}
  {}

  template <class X, class ...Y>
  constexpr decltype(f_(g_(std::declval<X>()), std::declval<Y>()...))
    operator() (X&& x, Y&& y)
  {
    return f_(g_(std::forward<X>(x)), std::forward<Y>(y)...);
  }

  constexpr decltype(f_(g_())) operator()()
  {
    return f_(g_());
  }
};

template <class F, class G, class ...H>
struct Composition<F, G, H...> : Composition<F, Composition<G, H...>>
{
  using Comp = Composition<G, H...>;


}

} // namespace Functors
} // namespace Categories

#endif // _CATEGORIES_FUNCTORS_COMPOSITION_H_