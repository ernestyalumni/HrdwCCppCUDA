#ifndef DATA_STRUCTURES_LINKED_LISTS_NODES_NODE_H
#define DATA_STRUCTURES_LINKED_LISTS_NODES_NODE_H

#include <algorithm>

namespace DataStructures
{
namespace LinkedLists
{
namespace Nodes
{

template <typename T>
class Node
{
	public:

		Node(const T& value = T{}, Node* next = nullptr):
      value_{value},
      next_{next}
    {}

    Node(const Node& other):
      value_{other.value_},
      next_{other.next_}
    {}

    Node& operator=(Node other)
    {
      std::swap(value_, other.value_);
      std::swap(next_, other.next_);
    }

		T get_value() const
    {
      return value_;
    }

    void set_value(const T& value)
    {
      value_ = value;
    }

		Node* get_next() const
    {
      return next_;
    }

    void set_next(Node* next)
    {
      next_ = next;
    }

		// friend class SingleList<T>;
		// friend class CycleList<T>;
		// friend class SentinelList<T>;
		// friend class CycleSentinelList<T>;

	private:

		T value_;
		Node* next_;
};

} // namespace Nodes
} // namespace LinkedLists
} // namespace DataStructures

#endif // DATA_STRUCTURES_LINKED_LISTS_NODES_NODE_H