"""
@file Recursion-CrosswordPuzzle.py

Log:
2020/10/15
02:44 looked up Python solution, copied.
"""

# get possible locations
def get_possible_locations(board, word):
    possible_locations = []
    length = len(word)

    grid_dimension = 10

    # horizontal possible location
    for i in range(grid_dimension):
        for j in range(grid_dimension - length + 1):
            good = True
            if j > 0:
                if board[i][j - 1] != '+':
                    good = False
                    #break
            for k in range(len(word)):
                if board[i][j + k] not in ['-', word[k]]:
                    good = False
                    break
            if (j + len(word)) < grid_dimension:
                if board[i][j + len(word)] != '+':
                    good = False
                    #break
            if good:
                yield (i, j, 0) # 0 is axis indicator

    # vertical possible location
    for i in range(grid_dimension - length + 1):
        for j in range(grid_dimension):
            good = True
            if i > 0:
                if board[i - 1][j] != '+':
                    good = False
                    #break
            for k in range(len(word)):
                if board[i + k][j] not in ['-', word[k]]:
                    good = False
                    break
            if (i + len(word)) < grid_dimension:
                if board[i + len(word)][j] != '+':
                    good = False
                    #break
            if good:
                yield (i, j, 1) # 1 is axis indicator


# revert move
def revert(board, word, start_location):
    i, j, axis = start_location
    if axis == 0: # axis 0 is horizontal
        for k in range(len(word)):
            board[i][j + k] = '-'
    else: # axis 1 is vertical
        for k in range(len(word)):
            board[i + k][j] = '-'


# Write the word on board at a specified location.
def move(board, word, start_location):
    i, j, axis = start_location
    if axis == 0: # axis 0 is horizontal
        for k in range(len(word)):
            board[i][j + k] = word[k]
    else: # axis 1 is vertical
        for k in range(len(word)):
            board[i + k][j] = word[k]


def solve(board, words):
    global solved

    # base case.
    if len(words) == 0:
        if not solved:
            print(board)
        solved = True

        return board

    word = words.pop()

    possible_locations = [
        location for location in get_possible_locations(board, word)]

    for location in possible_locations:
        move(board, word, location)
        solve(board, words)

        # backtracking, recursion technique to search other directions.
        revert(board, word, location)

    words.append(word)


example_input_board = [
list("++++++++++"),
list("+------+++"),
list("+++-++++++"),
list("+++-++++++"),
list("+++-----++"),
list("+++-++-+++"),
list("++++++-+++"),
list("++++++-+++"),
list("++++++-+++"),
list("++++++++++")]

example_input_words = "POLAND;LHASA;SPAIN;INDIA"

if __name__ == "__main__":

    print(list(get_possible_locations(example_input_board, "POLAND")))
    print(list(get_possible_locations(example_input_board, "LHASA")))
    print(list(get_possible_locations(example_input_board, "SPAIN")))
    print(list(get_possible_locations(example_input_board, "INDIA")))

    solved = False
    example_solved_board = solve(
        example_input_board,
        example_input_words.split(";"))

    print(example_solved_board)