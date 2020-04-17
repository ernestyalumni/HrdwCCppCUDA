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

namespace Categories
{
namespace Monads
{

// cf. Sec. 10.6 Handling state with monads, pp. 216, Ch. 10 Monads of Čukić

// Instead of handling failures, keep a debugging log of operations performed.
// This log is state you want to change.

template <typename T>
class WithLog
{
  public:

    WithLog(T value, std::string& log = std::string{}) :
      value_{value},
      log_{log}
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

} // namespace Monads
} // namespace Categories

#endif // CATEGORIES_MONADS_WRITER_MONAD_H