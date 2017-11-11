/**
 * 	@file 	stack_more.c
 * 	@brief 	more on C implementation with arrays of stack 
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
#include <limits.h>    // INT_MIN

// struct to represent Stack of ints as arrays
struct iStack_arr {
	int N_top; 	// N_top = top  
	unsigned N; // N = capacity
	int* arr; 
};

// function to create a stack of given capacity.  It initializes size of 
// stack as 0
struct iStack_arr* createiStack_arr(unsigned N) {
	struct iStack_arr* stack = (struct iStack_arr*) malloc (sizeof(struct iStack_arr)) ;
	stack->N = N;
	stack->N_top = -1;
	stack->arr = (int*) malloc(N * sizeof(int));
}

// Stack is full when N_top = (N-1) (0-based counting)
int isFull(struct iStack_arr* stack)
{
	return stack->N_top == stack->N - 1;
}

// Stack is empty when N_top == -1
int isEmpty(struct iStack_arr* stack)
{
	return stack->N_top == -1;
}

// Function to add an item to stack.  It increases N_top by 1
void push(struct iStack_arr* stack, int item) 
{
	if (isFull(stack))
		return;
	++(stack->N_top);  
	stack->arr[	stack->N_top] = item;
	printf("%d pushed to stack\n", item);
}

// Function to remove an item from stack.  It decreases N_top by 1
int pop(struct iStack_arr* stack) 
{
	if (isEmpty(stack))
		return INT_MIN;
	return stack->arr[stack->N_top--];
}

/* =============== Linked list version of stack =============== */
struct iStackNode {
	int x; // x=data \in \mathbb{Z}, ZZ
	struct StackNode* next;
};

struct iStackNode* newNode(int x_in) // x_in is data to input
{
	struct iStackNode* stackNode = 
		(struct iStackNode*) malloc(sizeof (struct iStackNode)); 
		
	stackNode->x = x_in;
	stackNode->next = NULL;
	return stackNode;
}

/* error: conflicting types for ‘isEmpty’
int isEmpty(struct iStackNode *root)
{
	return !root;
}
*/

// Driver program to test above functions
int main()
{
	struct iStack_arr* stack = createiStack_arr(100);
	push(stack, 1);
	push(stack, 2);
	push(stack, 4);
	push(stack, 5);
	push(stack, 8);
	
	printf("%d popped from stack\n", pop(stack));
	
	return 0;
}
