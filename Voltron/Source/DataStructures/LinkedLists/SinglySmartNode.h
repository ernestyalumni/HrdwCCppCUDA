#ifndef DATA_STRUCTURES_LINKED_LISTS_SINGLY_SMART_NODE_H
#define DATA_STRUCTURES_LINKED_LISTS_SINGLY_SMART_NODE_H

#include <memory>
#include <utility> // for std::forward()

namespace DataStructures
{

namespace Lists
{

namespace SinglyLinked
{

template <typename T>
class Node
{
	public:

		explicit Node(T&& value):
			value_{std::forward<T>(value)},
			next_{nullptr}
		{}

		// Without this constructor, lvalues cannot be inputted.
		explicit Node(T& value):
			value_{std::forward<T>(value)},
			next_{nullptr}
		{}

    // cf. https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines#f7-for-general-use-take-t-or-t-arguments-rather-than-smart-pointers
		Node(T&& value, Node<T>* next):
			value_{std::forward<T>(value)},
			next_{next}
		{}

		virtual ~Node() = default;

		T value() const
		{
			return value_;
		}

		// deferences pointer to the next object.
		Node<T>& get_next()
		{
			assert(next_);

			return *next_.get();
		}

		// Checks if next_ owns an object, i.e. whether get() != nullptr
		bool is_linked() const
		{
			return static_cast<bool>(next_);
		}

		// Releases ownership of managed object if any. Caller is responsible for
		// deleting object.
		Node<T>* release()
		{
			return next_.release();
		}

		void link_new(Node<T>* next)
		{
			next_.reset(next);
		}

	private:

		T value_;
		std::shared_ptr<Node<T>> next_;
};

} // namespace SinglyLinked

} // namespace Lists

} // namespace DataStructures

#endif // DATA_STRUCTURES_LINKED_LISTS_SINGLY_SMART_NODE_H