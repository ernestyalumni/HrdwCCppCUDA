#include <assert.h>
#include <string.h> // strlen

void swap_by_index(char str[], int i, int j)
{
  assert(i >= 0 && j >= 0);
  char temp = str[i];
  str[i] = str[j];
  str[j] = temp;
}

/* Space complexity: N (only the string) and 1 temporary char byte.
 *
 */
void reverse_string(char str[])
{
  // strlen does not count null-terminated byte.
  // strlen complexity: O(N) - checks with counter each char for null-terminated
  // character.
  int N = strlen(str);

  if (N == 0 || N == 1)
  {
    return;
  }

  
  if (N == 2)
  {
    swap_by_index(str, 0, 1);
    return;
  }

  int l = 0;
  int r = N - 1;
  // Time complexity. O(N/2) ~ O(N)
  while (l < r)
  {
    swap_by_index(str, l, r);
    ++l;
    --r;
  }

  return;
}