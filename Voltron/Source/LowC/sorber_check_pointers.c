/*
 * @url https://youtu.be/yM9zteeTCiI
 * @ref Jacob Sorber. How to Check Your Pointers at Runtime. Oct. 20, 2020
 * @details
 * Example Compiling:
 * gcc sorber_check_pointers.c -o sorber_check_pointers
 */

/* Segmentation Fault is when you try to access a spot in memory that has not
 * been mapped.
 * Process hasn't requested that memory, your process doesn't have access to
 * that memory space, but you try to access it anyway.
 */

#include <stdio.h>
#include <sys/types.h>
#include <sys/mman.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h> /* pipe() */
#include <sys/errno.h>
#include <stdint.h> /* uintptr_t */

int ismapped(const void *ptr, int bytes) {
  if (ptr == NULL) {
    return 0;
  }

  //create a pipe, create 2 endpoints - doesn't go anywhere
  /* int pipe(int pipefd[2])
   * pipefd[0] refers to read end of pipe
   * pipefd[1] refers to write end of pipe.
   */
  int fd[2];
  int valid = 1;

  pipe(fd);
  if (write(fd[1], ptr, bytes) < 0) {
    if (errno == EFAULT)
      valid = 0;
  }

  close(fd[0]);
  close(fd[1]);
  return valid;
}

/* bytes is number of bytes needed to specify range to access of p */
void testptr(void *p, int bytes, char *name) {
  /* We want a function that would say the byte is allocated, whether pointer is
   * safe. */

  printf("%s:\t%d\t%p\n", name, ismapped(p, bytes), p);
}

int main()
{
  /* Assign junk to something we know we can't access, like NULL */
  int *junk = NULL;
  /* make own custom address */
  int *junk2 = (int*)((uintptr_t)0x352342524a);
  int *p = malloc(50);
  int x = 5;
  /* pointer on the stack */
  int *px = &x;

  /* Uncomment out for Segmentation Fault. */
  /* *junk = 567; */

  testptr(junk, 1, "junk");
  testptr(junk2, 1, "junk2");
  testptr(px, sizeof(int), "px");
  testptr(p, 50, "p");
}