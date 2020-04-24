//------------------------------------------------------------------------------
/// \file IOMonad.h
/// \author Ernest Yeung
/// \brief http://www.reanimator.ltd.uk/code/monads-in-cpp
///-----------------------------------------------------------------------------
#ifndef CATEGORIES_MONADS_IO_MONAD_H
#define CATEGORIES_MONADS_IO_MONAD_H

namespace Categories
{
namespace Monads
{
namespace IOMonad
{

class IO
{
  public:

    IO()
    {}

    template <typename Operation>
    IO do_operation(Operation op) const
    {
      op();
      return IO();
    }
};

} // namespace IOMonad
} // namespace Monads
} // namespace Categories

#endif // CATEGORIES_MONADS_IO_MONAD_H