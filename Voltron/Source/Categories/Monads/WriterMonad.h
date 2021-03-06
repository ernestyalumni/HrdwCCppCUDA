//------------------------------------------------------------------------------
/// \file WriterMonad.h
/// \author Ernest Yeung
/// \brief
/// \ref Ivan Čukić, Functional Programming in C++,  Manning Publications;
/// 1st edition (November 19, 2018). ISBN-13: 978-1617293818
///-----------------------------------------------------------------------------
#ifndef CATEGORIES_MONADS_WRITER_MONAD_H
#define CATEGORIES_MONADS_WRITER_MONAD_H

#include <string>
#include <type_traits>

namespace Categories
{
namespace Monads
{
namespace WriterMonad
{

// cf. Sec. 10.6 Handling state with monads, pp. 216, Ch. 10 Monads of Čukić

// Instead of handling failures, keep a debugging log of operations performed.
// This log is state you want to change.

template <typename T>
class WithLog
{
  public:

    WithLog(T value, std::string& log):
      value_{value},
      log_{log}
    {}

    explicit WithLog(T value):
      value_{value},
      log_{}
    {}


    T value() const
    {
      return value_;
    }

    std::string log() const
    {
      return log_;
    }

  private:

    T value_;
    std::string log_;
};

// Redefine user_full_name and to_html functions to return values along with
// log.
//
// WithLog<std::string> user_full_name(const std::string& login);
//
// WithLog<std::string> to_html(const std::string& text);

// Composition

// Takes instance of WithLog<T>, val, which contains value and current log
// (state),
// function F f that transforms value and returns transformed value with new
// logging information.
// mbind needs to return new result along with new logging information appended
// to old log.
template <
  typename T,
  typename F,
  typename Ret = typename std::result_of<F(T)>::type
  >
Ret mbind(const WithLog<T>& val, F f)
{
  // Transforms given value with f, which yields resulting value and log string
  // that f generated.
  const auto result_with_log = f(val.value());

  // You need to return result value, but not just log that f returned;
  // concatenate it with previous log.
  return Ret(
    result_with_log.value(),
    val.log() + result_with_log.log());
}

// This is the Cartesian product of W and X, W x X
template <typename X, typename W>
class WriterMonadEndomorphism
{
  public:

    WriterMonadEndomorphism(const X value, const W& log) :
      value_{value},
      log_{log}
    {}

    explicit WriterMonadEndomorphism(const X value) :
      value_{value},
      log_{W{}}
    {}

    X value() const
    {
      return value_;
    }

    W log() const
    {
      return log_;
    }

  private:

    X value_;
    W log_;
};

// T is the endomorphism
template <typename X, typename T>
T unit(const X x)
{
  return T{x};
}

template <typename Morphism, typename T>
T bind(const T& tx, Morphism f)
{
  const auto ty = f(tx.value());

  return T{ty.value(), tx.log() + ty.log()};
}

} // namespace WriterMonad
} // namespace Monads
} // namespace Categories

#endif // CATEGORIES_MONADS_WRITER_MONAD_H