#ifndef DATA_STRUCTURES_QUEUES_LEXICOGRAPHIC_PAIR_H
#define DATA_STRUCTURES_QUEUES_LEXICOGRAPHIC_PAIR_H

#include <cstddef>

namespace DataStructures
{
namespace Queues
{
namespace DWHarder
{


//------------------------------------------------------------------------------
/// \ref 7.01.Priority_queues.Questions.pdf, 2013 D.W. Harder. ECE 250. 7.1c
//------------------------------------------------------------------------------
template <typename T>
class LexicographicPair
{
  public:

    T priority_;

    LexicographicPair(const T& priority):
      priority_{priority},
      count_priority_{count_++}
    {}

    bool operator==(const LexicographicPair& other) const
    {
      return (priority_ == other.priority_) && (count_priority_ ==
        other.count_priority_);
    }

    bool operator!=(const LexicographicPair& other) const
    {
      return !operator==(other);
    }
    bool operator<(const LexicographicPair& other) const
    {
      return (priority_ < other.priority_) ? true :
        (priority_ == other.priority_) && (count_priority_ <
          other.count_priority_);
    }

    bool operator<=(const LexicographicPair& other) const
    {
      return operator<(other) || operator==(other);
    }

    bool operator>(const LexicographicPair& other) const
    {
      return other.operator<(*this);
    }

    bool operator>=(const LexicographicPair& other) const
    {
      return other.operator<=(*this);
    }

  protected:

    std::size_t count_priority_;
    static std::size_t count_;
};

template <typename T>
std::size_t LexicographicPair<T>::count_ = 0;

} // namespace DWHarder
} // namespace Queues
} // namespace DataStructures

#endif // DATA_STRUCTURES_QUEUES_LEXICOGRAPHIC_PAIR_H