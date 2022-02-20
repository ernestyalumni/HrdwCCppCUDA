#!/bin/sh

# cf. Lecture 02 - Shell Scripting, 15-123S09 CS CMU

# Run the program like: sh myscript.sh 2 + 3

case "$2"
in
  "+") ans=`expr $1 + $3`
      printf "%d %s %d = %d\n" $1 $2 $3 $ans
      ;; # Two ;;'s serve as the break
  "-") ans=`expr $1 - $3`
      printf "%d %s %d = %d\n" $1 $2 $3 $ans
      ;; # Two ;;'s serve as the break
  "\*") ans=`expr $1 * $3`
      printf "%d %s %d = %d\n" $1 $2 $3 $ans
      ;; # Two ;;'s serve as the break
  "/") ans=`expr $1 / $3`
      printf "%d %s %d = %d\n" $1 $2 $3 $ans
      ;; # Two ;;'s serve as the break

  # Notice this: the default case is a simple *
  *) printf "Don't know how to do that.\n"
      ;; # Two ;;'s serve as the break
  esac
