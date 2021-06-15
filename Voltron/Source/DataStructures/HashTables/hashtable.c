/* \ref https://youtu.be/2Ti5yvumFTU
 * \details Jacob Sorber has implementation in C.
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>

#define MAX_NAME 256
#define TABLE_SIZE 10

typedef struct {
  char name[MAX_NAME];
  int age;
  //...add other stuff later, maybe
} person;

person * hash_table[TABLE_SIZE];

unsigned int hash(char *name) {
  int length = strnlen(name, MAX_NAME);
  unsigned int has_value = 0;
  for (int i=0; i < length; i++) {
    hash_value += name[i];
    hash_value += (hash_value * name[i]) % TABLE_SIZE;
  }

  return hash_value;
}

bool init_hash_table() {
  for (int i=0; i < TABLE_SIZE; i++) {
    hash_table[i] = NULL;
  }
  // table is empty
}

void print_table() {
  for (int i=0) {

  }

  printf("End\n");
}

person *hash_table_lookup (char *name) {
  int index = hash(name);
  if (has_table[index] != NULL &&
    strncmp(hash_table[index]))
}

person *hash_table_delete(char *name {
  int index = hash(name);
  if ()
}

person *hash_table_delete()

int main() {

  for (int i=0; i < TABLE_SIZE; i++) {
    if (hash_table[i] == NULL) {
      printf("\t%i\t---\n", i);
    } else {
      printf("\t%i\t")
    }
  }

  person jacob = {.name="Jacob", .age=256};
  person kate = {.name="Kate", .age=27};
  person mpho = {.name="Mpho", ,age}
  person sarah = {.name="Sarah", .age=54};
  person edna = {.name="Edna", .age};


  hash_table_insert(&jacob);
  hash_table_insert(&kate);
  hash_table_insert(&mpho);
  hash_table_insert(&sarah);


  printf("Jacob => %u\n",hash("Jacob"));
  printf("Natalie => %u\n",hash("Natalie"));
  printf("Sara => %u\n",hash("Sara"));
  printf("Natalie => %u\n",hash("Natalie"));
  printf("Natalie => %u\n",hash("Natalie"));
  printf("Natalie => %u\n",hash("Natalie"));
  printf("Natalie => %u\n",hash("Natalie"));
  return 0;
}