/**
 * 	@file 	stack_stack_arr.c
 * 	@brief 	C implementation with arrays of stack on stack memory
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
 * gcc -Wall -g stack_stack_arr.c -o stack_stack_arr
 * */
#include <stdio.h> 
#include <stdlib.h> // malloc
#include <limits.h>  // INT_MIN

// struct to represent a stack with array
struct Stack {
	int n_top; // n_top = top 
	unsigned N; // N = capacity
	int* arr;
}; 

// function to create stack of given capacity N.  It initializes size of 
// stack as 0 
struct Stack* createStack(unsigned N) 
{
	struct Stack* stack;
	stack->N = N;
	stack->n_top = -1;
	int arr[N];
	stack->arr = arr;
}

// Stack is full when N_top = N-1 (0-based counting)
int isFull(struct Stack* stack)
{
	return stack->n_top == stack->N - 1;
}

// Stack is empty when n_top = -1
int isEmpty(struct Stack* stack)
{
	return stack->n_top == -1;
}

// Function to add an item to stack.  It increases n_top by 1
void push(struct Stack* stack, int item)
{
	if (isFull(stack))
		return;
	++stack->n_top;
	stack->arr[stack->n_top] = item;
	printf("%d pushed to stack\n", item);
}

// Function to remove an item from stack.  It decreases n_top by 1
int pop(struct Stack* stack)
{
	if (isEmpty(stack))
		return INT_MIN;
	return stack->arr[stack->n_top--];
}

// Driver program to test above functions
int main()
{
	struct Stack* stack=createStack(10);
	
	push(stack, 1);
	push(stack, 2);
	push(stack, 4);
	push(stack, 5);
	push(stack, 8);
	
	printf("%d popped from stack\n", pop(stack));
	
	return 0;
}
