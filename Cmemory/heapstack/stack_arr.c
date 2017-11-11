/**
 * 	@file 	stack_arr.c
 * 	@brief 	C implementation with arrays of stack 
 * 	@ref	http://www.geeksforgeeks.org/stack-data-structure-introduction-program/
 * 	@details 3 (or 4?) basic operations; 
 * 	- push, if stack is full, then it's said to be an Overflow condition
 * 	- pop, if stack empty, then it's said to be an Underflow condition 
 * 	- peek or top, returns top element of stack.  
 * 	- isEmpty, returns true if stack empty, else false
 * 
 * Pros; easy to implement, memory not saved as pointers; Cons; no dynamic, doesn't grow and shrink depending on runtime needs
 * 
 * COMPILATION TIP : -Wall warning all, -g debugger, gdb can step through 
 * gcc -Wall -g stack_arr.c -o stack_arr
 * */
#include <stdio.h> 
#include <stdlib.h> // malloc
#include <limits.h>  // INT_MIN

// A structure to represent a stack  
struct Stack
{
	int top;
	unsigned capacity;
	int* array;
};

// function to create a stack of given capacity.  It initializes size of 
// stack as 0
struct Stack* createStack(unsigned capacity) 
{
	// ironically, this "stack" is on heap memory
	struct Stack* stack = (struct Stack*) malloc (sizeof(struct Stack));
	stack->capacity = capacity;
	stack->top = -1;
	stack->array = (int*) malloc(stack->capacity * sizeof(int));
	return stack;
}

// Stack is full when top is equal to the last index
int isFull(struct Stack* stack)
{
	return stack->top == stack->capacity - 1; 
}

// Stack is empty when top is equal to -1
int isEmpty(struct Stack* stack)
{
	return stack->top == -1;
}

// Function to add an item to stack.  It increases top by 1
void push(struct Stack* stack, int item) 
{
	if (isFull(stack))
		return;
	stack->array[++stack->top] = item;
	printf("%d pushed to stack\n", item);
}

// Function to remove an item from stack.  It decreases top by 1
int pop(struct Stack* stack)
{
	if (isEmpty(stack))
		return INT_MIN;
	return stack->array[stack->top--];
}
// Driver program to test above functions
int main()
{
	struct Stack* stack = createStack(100);
	
	push(stack, 10);
	push(stack, 20);
	push(stack, 30);
	
	printf("%d popped from stack\n", pop(stack));
	
	return 0;
}
