//------------------------------------------------------------------------------
/// \file ProgramStateExample.h
/// \author Ernest Yeung
/// \brief Program state example from Ch. 9 of Cukic.
///
/// \details Example of (algebraic data type) sum types through inheritance.
///-----------------------------------------------------------------------------
#ifndef _CATEGORIES_ALGEBRAIC_DATA_TYPES_PROGRAM_STATE_EXAMPLE_H_
#define _CATEGORIES_ALGEBRAIC_DATA_TYPES_PROGRAM_STATE_EXAMPLE_H_

#include <cassert>
#include <memory> // std::make_unique
#include <variant>

namespace Categories
{
namespace AlgebraicDataTypes
{
namespace ProgramStates
{

namespace ThroughInheritance
{

// cf. 9.1.1 pp. 176 Cukic, Sum types through inheritance.
// Have a superclass representing the sum type and derived classes to represent
// summed types.
class StateT
{
  public:
    virtual ~StateT()
    {}

    int type_value() const
    { 
      return type_value_;
    }

  protected:

    // It shouldn't be possible to create instances of this class, so make the
    // constructor protected. It can be called only by classes that inherit from
    // StateT.
    // Each subclass should pass a different value for the type argument. You
    // can use it as an efficient replacement for dynamic_cast.
    //  
    StateT(int type_value) :
      type_value_{type_value}
    {}

  private:
    int type_value_;    
};

class InitialT : public StateT
{
  public:
    enum {id = 0};
    InitialT() :
      StateT{id}
    {}
};

class RunningT : public StateT
{
  public:
    enum { id = 1 };
    RunningT() :
      StateT{id}
    {}

    unsigned count() const
    { return count_; }

  private:
    // For the running state, you need a counter and the handler to the web page
    // whose words you want to count.
    unsigned count_ {0};
    // socket_t web_page_;
};

class FinishedT : public StateT
{
  public:
    enum { id = 2 };
    FinishedT(unsigned count) :
      StateT(id),
      count_(count)
    {}

    unsigned count() const
    { return count_; }

  private:
    unsigned count_;
};

// Original implementation from pp. 181 Ch. 9 of Cukic
/* 
class InitialStateT
{};

class RunningT
{
  public:

    unsigned count() const
    {
      return count_;
    }

  private:

    unsigned count_ {0};
};

class FinishedT
{
  public:
    FinishedT(const unsigned count) :
      count_{count}
    {}

    unsigned count() const
    {
      return count_;
    }

  private:
    unsigned count_;
};
*/

} // ThroughInheritance

namespace ThroughStdVariant
{

class InitialT
{};

// RunningT, FinishedT classes define only their state and nothing else.

class RunningT
{
  public:
    unsigned count() const
    {
      return count_;
    }
  private:
    unsigned count_ {0};
    // socket_t web_page_;
};

class FinishedT
{
  public:
    FinishedT(unsigned count) :
      count_{count}
    {}

    unsigned count() const
    {
      return count_;
    }
  private:
    unsigned count_;
};

} // namespace ThroughStdVariant

namespace MainPrograms
{

class ProgramTThroughInheritance
{
  public:

    using FinishedT = ProgramStates::ThroughInheritance::FinishedT;
    using InitialT = ProgramStates::ThroughInheritance::InitialT;
    using RunningT = ProgramStates::ThroughInheritance::RunningT;

    ProgramTThroughInheritance() :
      state_{std::make_unique<ProgramStates::ThroughInheritance::InitialT>()}
    {}

    void counting_finished()
    {
      assert(state_->type_value() ==
        ProgramStates::ThroughInheritance::RunningT::id);

      auto state =
        static_cast<ProgramStates::ThroughInheritance::RunningT*>(state_.get());

      state_ =
        std::make_unique<
          ProgramStates::ThroughInheritance::FinishedT>(state->count());
    }

  private:
    std::unique_ptr<ProgramStates::ThroughInheritance::StateT> state_;
};

class ProgramTThroughStdVariant
{
  public:

    using FinishedT = ProgramStates::ThroughStdVariant::FinishedT;
    using InitialT = ProgramStates::ThroughStdVariant::InitialT;
    using RunningT = ProgramStates::ThroughStdVariant::RunningT;

    // You initialized 
    ProgramTThroughStdVariant() :
      state_{InitialT{}}
    {}

    void counting_finished()
    {
      // Uses std::get_if to check whether there's a value of a specified type
      // in std::variant. You didn't pass it a pointer.
      //  - std::variant isn't based on dynamic polymorphism; it doesn't store
      //  pointers to objects allocated on the heap.
      // std::variant stores actual object in its own memory space, just as with
      // ordinary unions.
      // It automatically handles construction and destruction of objects stored
      // within, and knows actly type of object it contains at any given time.
      // std::get either returns value or throws
      // std::get_if returns ptr to contained value, or nullptr on error.
      auto* state = std::get_if<RunningT>(&state_);

      // Returns nullptr if the variant doesn't hold the value of the specified
      // type.
      assert(state != nullptr);

      state_ = FinishedT{state->count()};
    }

  private:
    std::variant<InitialT, RunningT, FinishedT> state_;
};

} // namespace MainPrograms

} // namespace ProgramStates

} // namespace AlgebraicDataTypes
} // namespace Categories

#endif // _CATEGORIES_ALGEBRAIC_DATA_TYPES_PROGRAM_STATE_EXAMPLE_H_
