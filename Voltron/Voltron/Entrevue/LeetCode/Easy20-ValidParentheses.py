"""
@file Easy20-ValidParentheses.py

Log:
2020/10/13
21:04
23:54 Looked up solution.

2020/10/14
1:06 Implementation with stacks.
"""
example_input_1 = "()"
example_input_2 = "()[]{}"
example_input_3 = "(]" # false
example_input_4 = "([)]" # false
example_input_5 = "{[]}"

example_input_6 = "(){}}{"
example_input_7 = "(([]){})"
example_input_8 = "(("

matching_brackets = [
    ("(", ")"),
    ("[", "]"),
    ("{", "}")]

def check_first_element(s: str):
    """
    If s[0] is a "right" bracket immediately return false.
    If s[0] is a "left" bracket, it must be closed either for s[1] or on the
    last element of str. On the other hand,
    Proof: If the "right" bracket is not in any s[i], i > 0, then false.
    If "right" bracket is in any s[i], 1 < i < (|s| - 1), then that implies a
    "new" set of brackets beginning for j > i.
    """
    # O(1) Time (Zeit)
    if s[0] in [ele[1] for ele in matching_brackets]:
        return False
    # Assume input string is a "correct" input consisting of parentheses only.
    # (Constraint 2)

    def match_to_first_element(bracket_tuple):
        return bracket_tuple[0] == s[0]

    # O(1) Time (Zeit)
    matching_bracket = list(
        filter(match_to_first_element, matching_brackets))[0]

    # O(1) Time (Zeit) for the check.
    # Recursion check (if valid input), can take O(N/2) ~ O(N)
    # Could stop earlier for a false case.
    # Can't assume this.
    if (s[-1] == matching_bracket[1]):
        if (len(s) == 2):
            return True
        return check_first_element(s[1:-1])

    # This is the particular case that the "first" bracket was ended for a s[i]
    # such that 1 < i < (|s| - 1). So we'll need to treat it with another
    # procedure.
    return s

#def process_multiple_bracket_sets(s: str):
    """
    Assume this is a "multiple" set of brackets, so s[0] and s[-1] 
    """

def is_left_bracket(bracket):
    return bracket in [ele[0] for ele in matching_brackets]

def is_right_bracket(bracket):
    return bracket in [ele[1] for ele in matching_brackets]

def get_matching_bracket(bracket):
    if is_left_bracket(bracket):
        def match_to_left_bracket(bracket_tuple):
            return bracket_tuple[0] == bracket

        matching_bracket = list(
            filter(match_to_left_bracket, matching_brackets))[0]

        return matching_bracket[1]

    else:
        def match_to_right_bracket(bracket_tuple):
            return bracket_tuple[1] == bracket

        matching_bracket = list(
            filter(match_to_right_bracket, matching_brackets))[0]

        return matching_bracket[0]

def process_brackets(s: str):
    """
    A valid subexpression can be
    - composed of multiple valid subexpressions
    - can have multiple valid subexpressions within a matching set brackets
    "on the outside".

    For a valid subexpression, of length N,
    Suppose for 0 <= i < N,
    s[i] is a "left" bracket, and s[i + 1] is a "right" bracket. s[i + 1] must
    match s[i], and we can remove s[i], s[i + 1] from consideration. Otherwise,
    immediately return False.
    Consider s[i_1], s[i_2] on the "stack" in that order (implied in i_1, i_2).
    Then if subexpression is valid, there exist j_2, j_1 such that j_2 < j_1,
    and s[j_2] "matches" s[i_2] and s[j_1] "matches" s[i_1].

    By induction, this shows a valid or invalid subexpression.
    """

    stack = []
    s_as_list = list(s)
    if (len(s_as_list) == 0):
        return True
    # Odd number of parentheses cannot pair up.
    if (len(s_as_list) % 2 != 0):
        return False

    while (len(s_as_list) > 0):

        if is_left_bracket(s_as_list[0]):
            stack.append(s_as_list[0])
            s_as_list.pop(0)
        # ele is otherwise a right bracket
        else:
            if len(stack) == 0:
                return False
            
            matching_right_bracket = get_matching_bracket(stack[-1])
            if (matching_right_bracket == s_as_list[0]):
                stack.pop()
                s_as_list.pop(0)
            else:
                return False

    if len(stack) == 0 and len(s_as_list) == 0:
        return True
    elif len(stack) > 0:
        return False

    else:
        return stack, s_as_list


def process_element_pair(left_s: str, right_s: str):
    """
    Given an input string s, s[0], s[1],
    If s[0] is a "right" closing bracket, then return False. Otherwise,
    s[0], s[1] is either a pair of matching brackets, so return True, or
    s[1] is a  "right" closing bracket of wrong type to s[0], return False, or
    s[1] is a "left" "starting" bracket.

    Generalize this.
    """
    # O(1) Time (Zeit)
    #if left_s in [ele[1] for ele in matching_brackets]:
    if is_right_bracket(left_s):
        return False

    def match_to_left_element(bracket_tuple):
        return bracket_tuple[0] == left_s

    matching_bracket = list(
        filter(match_to_left_element, matching_brackets))[0]

    if (right_s == matching_bracket[1]):
        return True
#    elif (right_s in [ele[1] for ele in matching_brackets]):
    elif (is_right_bracket(right_s)):
        return False
    else:
        return [left_s, right_s]

def process_bracket_set(s: str):
    """
    Assume s is even length and s > 2.
    """
    left_brackets = []

    while(len(s) > 0):
        if is_left_bracket(s[0]):
            left_brackets.append(s[0])
            s = s[1:]
        else:
            break

    if len(left_brackets) > len(s):
        return False

    while (len(left_brackets) > 0):
        result = process_element_pair(left_brackets[-1], s[0])
        if result == True:
            left_brackets.pop()
            s = s[1:]
        else:
            return False

    return s


def process_brackets_set(s: str):
    """
    Given an input string s,
    If s[0] is a "right" closing bracket, then return False. Otherwise,
    s[0], s[1] is either a pair of matching brackets, so return True, or
    s[1] is a  "right" closing bracket of wrong type to s[0], return False, or
    s[1] is a "left" "starting" bracket.

    """
    while (len(s) > 0):

        # Odd length strings can never be valid.
        if (len(s) % 2 != 0):
            return False

        if len(s) == 2:
            result = process_element_pair(s[0], s[1])
            if result == True:
                return True
            elif result == False:
                return False
            else:
                return False

        # Process brackets one by one until a "right" "closing" bracket is
        # found.
        # while (len(s) > 0):
        s = process_bracket_set(s)
        if s == False:
            return False
        if len(s) == 0:
            return True

    # if len(s) == 0:
    return True



class Solution:
    def isValid(self, s: str) -> bool:

        return process_brackets(s)

if __name__ == "__main__":

    print(is_left_bracket("["))
    print(is_left_bracket("{"))
    print(is_left_bracket("("))
    print(is_left_bracket(")"))
    print(is_left_bracket("}"))
    print(is_left_bracket("]"))
    print(is_right_bracket("["))
    print(is_right_bracket("{"))
    print(is_right_bracket("("))
    print(is_right_bracket(")"))
    print(is_right_bracket("}"))
    print(is_right_bracket("]"))

    """
    print(process_bracket_set(example_input_1))
    print(process_bracket_set(example_input_2))
    print(process_bracket_set(example_input_3))
    print(process_bracket_set(example_input_4))
    print(process_bracket_set(example_input_5))
    print(process_bracket_set(example_input_6))

    print(process_brackets_set(example_input_1))
    print(process_brackets_set(example_input_2))
    print(process_brackets_set(example_input_3))
    print(process_brackets_set(example_input_4))
    print(process_brackets_set(example_input_5))
    print(process_brackets_set(example_input_6))
    """
    print(process_brackets(example_input_1)) # True
    print(process_brackets(example_input_2)) # True
    print(process_brackets(example_input_3)) # False
    print(process_brackets(example_input_4)) # False
    print(process_brackets(example_input_5)) # True
    print(process_brackets(example_input_6)) # False
    print(process_brackets(example_input_7)) # True
    print(process_brackets(example_input_8)) # True
