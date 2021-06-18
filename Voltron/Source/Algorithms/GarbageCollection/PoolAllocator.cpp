#include "PoolAllocator.h"

#include <cstddef>
#include <cstdlib> // malloc

namespace Algorithms
{
namespace GarbageCollection
{

Chunk* PoolAllocator::allocate_block(const std::size_t chunk_size)
{
  const std::size_t block_size {chunks_per_block_ * chunk_size};

  // The first chunk of the new block.
  Chunk* block_begin {reinterpret_cast<Chunk*>(malloc(block_size))};

  // Once the block is allocated, we need to chain all the chunks in this block:

  Chunk* chunk {block_begin};

  for (int i {0}; i < chunks_per_block_ - 1; ++i)
  {
    chunk->next_ =
      reinterpret_cast<Chunk*>(reinterpret_cast<char*>(chunk) + chunk_size);
    chunk = chunk->next_;
  }

  chunk->next_ = nullptr;

  return block_begin;
}

} // namespace GarbageCollection
} // namespace Algorithms
