//------------------------------------------------------------------------------
/// \file Functor.h
/// \author Ernest Yeung
/// \brief Functors.
///-----------------------------------------------------------------------------
#ifndef _CATEGORIES_FUNCTORS_FUNCTOR_H_
#define _CATEGORIES_FUNCTORS_FUNCTOR_H_

namespace Categories
{
namespace Functors
{

//------------------------------------------------------------------------------
/// \class Functor
/// \brief Functors
/// \details 
/// \ref https://gist.github.com/splinterofchaos/3960343
//------------------------------------------------------------------------------

struct SequenceTag
{};

struct PointerTag
{};

template <class X>
X category(...);

template <class S>
auto category(const S& s) -> decltype(std::begin(s), SequenceTag{});

template <class Ptr>
auto category(const Ptr& p) -> decltype(*p, p=nullptr, PointerTag{});

template <class T>
struct Category
{
  using Type = decltype(category<T>(std::declval<T>()));
};

template <class R, class ... X>
struct Category<R(&)(X...)>
{
  using Type = R(&)(X...);
}

template <class T>
using CategoryType = typename Category<T>::type;

template <class ... >
struct Functor;

template <class F, class FX, class Function=Functor<CategoryType<FX>>>
auto fmap(F&& f, FX&& fx) ->
  decltype(Function::fmap(std::declval<F>(), std::declval<FX>()))
{
  return Function::fmap(std::forward<F>(f), std::forward<FX>(fx));
}

template <class F, class G>
struct Composition
{
  F f_;
  G g_;

  template <class X>
  auto operator()(X&& x) -> decltype(f_(g_(std::declval<X>)))
}

//template <class Function>
//struct Functor<Function>
//{
//  template <class F, class G, class C =
//}

} // namespace Functors
} // namespace Categories

#endif // _CATEGORIES_FUNCTORS_FUNCTOR_H_