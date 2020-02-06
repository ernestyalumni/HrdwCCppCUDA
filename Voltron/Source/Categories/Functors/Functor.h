//------------------------------------------------------------------------------
/// \file Functor.h
/// \author Ernest Yeung
/// \brief Functors.
///-----------------------------------------------------------------------------
#ifndef _CATEGORIES_FUNCTORS_FUNCTOR_H_
#define _CATEGORIES_FUNCTORS_FUNCTOR_H_

#include <functional>
#include <optional>

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
};

template <class T>
using CategoryType = typename Category<T>::type;

namespace OtherAlternatives
{

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
  auto operator()(X&& x) -> decltype(f_(g_(std::declval<X>)));
}; 

} // namespace OtherAlternatives

// cf. https://nalaginrut.com/archives/2019/10/31/8%20essential%20patterns%20you%20should%20know%20about%20functional%20programming%20in%20c%2B%2B14

template <class From, class To>
class Functor
{
	public:

		Functor(std::function<To(From)> operation) :
			operation_{operation}
		{}

		~Functor()
		{};

		template <class T>
		T operator()(T c)
		{
			std::transform(c.begin(), c.end(), c.begin(), operation_);
			return c;
		}

	private:
		std::function<To(From)> operation_;
};

// pp. 201, Cukic, Ivan. Functional Programming in C++. 10.1.1 Handling optional
// values, Listing 10.1 Defining the transform function for std::optional
template <typename T1, typename F>
auto transform(const std::optional<T1>& opt, F f)
  // Specify the return type, because you're returning just {} when there's no
  // value.
  -> decltype(std::make_optional(f(opt.value())))
{
  if (opt)
  {
    return std::make_optional(f(opt.value())); 
  }
  else
  {
    // If no value, returns an empty instance of std::optional
    return {};
  }
}

//template <class Function>
//struct Functor<Function>
//{
//  template <class F, class G, class C =
//}

} // namespace Functors
} // namespace Categories

#endif // _CATEGORIES_FUNCTORS_FUNCTOR_H_