#ifndef DIGITAL_INTS_BITWIDTH_H
#define DIGITAL_INTS_BITWIDTH_H

#include <cassert>

namespace DigitalInts
{

template <std::size_t NBits, class Enable = void>
class UInt32BitWidth
{
	static_assert(NBits <= 32, "Uint32bitwidth must fit in a uint32_t");	
};

} // namespace DigitalInts

#endif // DIGITAL_INTS_BITWIDTH_H