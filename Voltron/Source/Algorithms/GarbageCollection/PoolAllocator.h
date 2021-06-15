#ifndef ALGORITHMS_GARBAGE_COLLECTION_POOL_ALLOCATOR_H
#define ALGORITHMS_GARBAGE_COLLECTION_POOL_ALLOCATOR_H

#include <cstddef>

namespace Algorithms
{
namespace GarbageCollection
{

//------------------------------------------------------------------------------
/// \brief A chunk within a larger block
//------------------------------------------------------------------------------
struct Chunk
{
  //----------------------------------------------------------------------------
  /// \details When a chunk is free, the 'next' contains the address of the next
  /// chunk in a list.
  ///
  /// When it's allocated, this space is used by the user.
  //----------------------------------------------------------------------------
  Chunk* next_;
};

//------------------------------------------------------------------------------
/// \brief The allocator class.
///
/// \details FEatures:
///   - Parametrized by number of chunks per block
///   - Keeps track of the allocation pointer
///   - Bump-allocates chunks
///   - Requests a new larger block when needed
//------------------------------------------------------------------------------
class PoolAllocator
{
  public:

    PoolAllocator(const std::size_t chunks_per_block) :
      chunks_per_block_{chunks_per_block}
    {}

    void* allocate(const std::size_t size);

    void deallocate(void* ptr, std::size_t size);

  private:

    // Number of chunks per larger block.
    std::size_t chunks_per_block_;

    // Allocation pointer.
    Chunk* alloc_ {nullptr};

    //--------------------------------------------------------------------------
    /// \brief Allocates a larger block (pool) for chunks.
    //--------------------------------------------------------------------------
    Chunk* allocate_block(std::size_t chunk_size);
};

} // namespace GarbageCollection
} // namespace Algorithms

#endif // ALGORITHMS_GARBAGE_COLLECTION_POOL_ALLOCATOR_H