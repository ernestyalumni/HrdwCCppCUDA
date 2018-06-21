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


https://github.com/google/sensei/blob/master/sensei/util/endian.h
in 
https://github.com/google/sensei


## Miscellaneous

`snprintf`

cf. http://www.cplusplus.com/reference/cstdio/snprintf/

``` 
int snprintf(char* s, size_t n, const char* format, ...);
``` 
**Write formatted output to sized buffer**.

Composes a string with the same text that would be printed if *format* was used on `printf`, but instead of being printed, the content is stored as a C *string* in the buffer pointed by `s` (taking `n` as the maximum buffer capacity to fill).

## References

https://github.com/CppCon/CppCon2017/blob/master/Presentations/Type%20Punning%20In%20C%2B%2B17%20-%20Avoiding%20Pun-defined%20Behavior/Type%20Punning%20In%20C%2B%2B17%20-%20Avoiding%20Pun-defined%20Behavior%20-%20Scott%20Schurr%20-%20CppCon%202017.pdf

https://benjaminjurke.com/content/articles/2015/loss-of-significance-in-floating-point-computations/

http://www.math.uconn.edu/~leykekhman/courses/MATH3795/Lectures/Lecture_2_Floating_point.pdf

* David Goldberg. *What Every Computer Scientist Should Know About Floating-Point Arithmetic*. March 1991, Computing Surveys.

[Download link from waterloo.ca](https://ece.uwaterloo.ca/~dwharder/NumericalAnalysis/02Numerics/Double/paper.pdf)
