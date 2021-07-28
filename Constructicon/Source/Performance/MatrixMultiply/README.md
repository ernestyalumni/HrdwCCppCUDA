## AddressSanitizer

AddressSanitizer is a quick memory error checker that uses compiler instrumentation and a runtime library.

Do

```
make clean
make ASAN=1

./matrix_multiply_testbed
```

=================================================================
==92777==ERROR: LeakSanitizer: detected memory leaks

Direct leak of 48 byte(s) in 3 object(s) allocated from:
    #0 0x4be5cf  (/home/topolo/PropD/HrdwCCppCUDA/Constructicon/Source/Performance/MatrixMultiply/matrix_multiply_testbed+0x4be5cf)
    #1 0x4f4449  (/home/topolo/PropD/HrdwCCppCUDA/Constructicon/Source/Performance/MatrixMultiply/matrix_multiply_testbed+0x4f4449)
    #2 0x7f320aed4081  (/lib64/libc.so.6+0x27081)

Indirect leak of 192 byte(s) in 12 object(s) allocated from:
    #0 0x4be5cf  (/home/topolo/PropD/HrdwCCppCUDA/Constructicon/Source/Performance/MatrixMultiply/matrix_multiply_testbed+0x4be5cf)
    #1 0x4f44d7  (/home/topolo/PropD/HrdwCCppCUDA/Constructicon/Source/Performance/MatrixMultiply/matrix_multiply_testbed+0x4f44d7)

Indirect leak of 96 byte(s) in 3 object(s) allocated from:
    #0 0x4be5cf  (/home/topolo/PropD/HrdwCCppCUDA/Constructicon/Source/Performance/MatrixMultiply/matrix_multiply_testbed+0x4be5cf)
    #1 0x4f4490  (/home/topolo/PropD/HrdwCCppCUDA/Constructicon/Source/Performance/MatrixMultiply/matrix_multiply_testbed+0x4f4490)
    #2 0x7f320aed4081  (/lib64/libc.so.6+0x27081)

SUMMARY: AddressSanitizer: 336 byte(s) leaked in 18 allocation(s).
