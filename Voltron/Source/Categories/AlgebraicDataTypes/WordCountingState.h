//------------------------------------------------------------------------------
/// \file WordCountingStates.h
/// \author Ernest Yeung
/// \brief Program state example from Ch. 9 of Cukic.
/// \ref https://gitlab.com/manning-fpcpp-book/code-examples/blob/master/chapter-09/word-counting-states/main.cpp
/// \details Example of (algebraic data type) sum types through inheritance.
///-----------------------------------------------------------------------------
#ifndef _CATEGORIES_ALGEBRAIC_DATA_TYPES_WORK_COUNTING_STATES_H_
#define _CATEGORIES_ALGEBRAIC_DATA_TYPES_WORK_COUNTING_STATES_H_

#include <cassert>
#include <fstream>
#include <iterator>
#include <string>
#include <variant>

namespace Categories
{
namespace AlgebraicDataTypes
{

// A helper to create overloaded function objects
template <typename... FS>
struct overloaded : FS...
{
  using FS::operator()...;
};

template <typename... FS>
overloaded(FS...) -> overloaded<FS...>;

namespace WordCounting
{

namespace States
{

// The initial state does not need to contain anything.
class InitialT
{};

class RunningT
{
  public:
    RunningT(const std::string& file_name) :
      file_{file_name}
    {}

    void count_words()
    {
      // cf. https://stackoverflow.com/questions/7435713/simple-istream-iterator-question
      // "stream_iterator() which the program interprets as eof"
      // cf. https://en.cppreference.com/w/cpp/iterator/istream_iterator
      // The default-constructed std::istream_iterator is known as the
      // end-of-stream iterator. When a valid std::istream_iterator reaches the
      // end of the underlying stream, it becomes equal to the end-of-stream
      // iterator. Dereferencing or incrementing it further invokes undefined
      // behavior. 
      count_ =
        std::distance(
          std::istream_iterator<std::string>(file_),
          std::istream_iterator<std::string>());
    }

    unsigned count() const
    {
      return count_;
    }

  private:
    unsigned count_ {0};
    std::ifstream file_;
};

// The finished state contains only the final count
class FinishedT
{
  public:

    FinishedT(unsigned count = 0):
      count_{count}
    {}

    unsigned count() const
    {
      return count_;
    }

  private:
    unsigned count_;
};

} // namespace States

class ProgramT
{
  public:

    using FinishedT = States::FinishedT;
    using InitialT = States::InitialT;
    using RunningT = States::RunningT;

    ProgramT() :
      state_{InitialT{}}
    {}

    void count_words(const std::string& file_name)
    {
      assert(state_.index() == 0);
      state_ = RunningT{file_name};
      std::get<RunningT>(state_).count_words();
      counting_finished();
    }

    void counting_finished()
    {
      // One of the ways to work with variants (Section 9.1.2) is to use the
      // std::get_if function which returns a pointer to the data stored in the
      // variant if it contains a value of the requested type. Otherwise, it
      // returns nullptr
      const auto *state = std::get_if<RunningT>(&state_);

      assert(state != nullptr);

      state_= FinishedT{state->count()};
    }

    unsigned count() const
    {
      // Another approach is to use the std::visit function which executes a
      // given function on the value stored inside of the variant.
      //
      // The 'overloaded' helper function can be used to coming several
      // lambdas of different signatures into a single function
      // object.
      return std::visit(
        overloaded
        {
          [](InitialT)
          {
            return (unsigned) 0;
          },
          [](const RunningT& state)
          {
            return state.count();
          },
          [](const FinishedT& state)
          {
            return state.count();
          }
        }, state_);
    }

  private:
    std::variant<InitialT, RunningT, FinishedT> state_;
};

} // namespace WordCounting

} // namespace AlgebraicDataTypes
} // namespace Categories

#endif // _CATEGORIES_ALGEBRAIC_DATA_TYPES_WORK_COUNTING_STATES_H_
