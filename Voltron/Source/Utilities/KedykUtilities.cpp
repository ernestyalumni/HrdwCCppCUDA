#include "KedykUtilities.h"

namespace Utilities
{
namespace Kedyk
{

void raw_delete(void* array)
{
  ::operator delete(array);
}

} // namespace Kedyk
} // namespace Utilities