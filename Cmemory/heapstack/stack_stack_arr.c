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
//	int arr[] ; // invalid use of flexible array member at assignment
}; 

// function to create stack of given capacity N.  It initializes size of 
// stack as 0 
struct Stack createStack(unsigned N) 
{
	struct Stack stack;
	stack.N = N;
	stack.n_top = -1;
	int arr[N];
	stack.arr = arr;
	return stack;
}


// Stack is full when N_top = N-1 (0-based counting)
int isFull(struct Stack stack)
{
	return stack.n_top == (stack.N - 1);
}

// Stack is empty when n_top = -1
int isEmpty(struct Stack stack)
{
	return stack.n_top == -1;
}

// Function to add an item to stack.  It increases n_top by 1
void push(struct Stack* ptr_stack, int item)
{
	if (isFull(*ptr_stack))
		return;
	
//	++ptr_stack->n_top;
	ptr_stack->n_top = ptr_stack->n_top + 1;

//	ptr_stack->arr[++ptr_stack->n_top] = item;
	(ptr_stack->arr)[ptr_stack->n_top] = item;


	printf("%d pushed to stack\n", item);

}


// Function to remove an item from stack.  It decreases n_top by 1
int pop(struct Stack* ptr_stack)
{
	if (isEmpty(*ptr_stack))
		return INT_MIN;
	
/*	int poppedval = stack.arr[stack.n_top];
	stack.n_top = stack.n_top - 1;
	
	return poppedval; 
	* */
	return ptr_stack->arr[ptr_stack->n_top--];
}



// Driver program to test above functions
int main()
{
//	struct Stack stack=createStack(10); SegFault access uninitialized ptr to struct Stack
	struct Stack stack = createStack(10);
	struct Stack* ptr_stack = &stack;

/*	
	push(stack, 1);
	push(stack, 2);
	push(stack, 4);
	push(stack, 5);
	push(stack, 18);
*/
	push(ptr_stack, 1);
	push(ptr_stack, 2);
	push(ptr_stack, 4);
	push(ptr_stack, 5);
	push(ptr_stack, 18);

	
	printf("%d popped from stack\n", pop(ptr_stack));
	
	return 0;
}
