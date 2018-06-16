# Endianness, Floating Point representations

cf. [`endian.h` from Linux Programmer's Manual](http://man7.org/linux/man-pages/man3/endian.3.html)

```
htobe16
htole16
be16toh
le16toh
htobe32
htole32
be32toh
le32toh
htobe64
be64toh
le64toh
```
convert values between host and big-/little-endian byte order.

```
#include <endian.h>

uint16_t htobe16(uint16_t host_16bits);

```
