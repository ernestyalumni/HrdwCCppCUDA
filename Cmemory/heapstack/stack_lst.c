/**
 * 	@file 	stack_lst.c
 * 	@brief 	C implementation with linked list of stack 
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
#include <limits.h>  

// A struct to represent a stack
struct StackNode
{
	int data;
	struct StackNode* next;
};

struct StackNode* newNode(int data) 
{
	struct StackNode* stackNode = 
		(struct StackNode*) malloc(sizeof(struct StackNode));
	stackNode->data = data;
	stackNode->next = NULL;
	return stackNode;
}

int isEmpty(struct StackNode *root) 
{
	return !root;
}

void push(struct StackNode** root, int data)
{
	struct StackNode* stackNode = newNode(data);
	stackNode->next = *root;
	*root = stackNode;
	printf("%d pushed to stack\n", data);
}

int pop(struct StackNode** root)
{
	if (isEmpty(*root))
		return INT_MIN;
	struct StackNode* temp = *root;
	*root = (*root)->next;
	int popped = temp->data;
	free(temp);
	
	return popped;
}

int peek(struct StackNode* root)
{
	if (isEmpty(root))
		return INT_MIN;
	return root->data;
}

int main()
{
	struct StackNode* root = NULL;
	
	push(&root, 10);
	push(&root, 20);
	push(&root, 30);
	
	printf("%d popped from stack\n", pop(&root));
	
	printf("Top element is %d\n", peek(root));
	
	return 0;
}
