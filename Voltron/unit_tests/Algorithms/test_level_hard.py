#from Voltron.Algorithms.level_hard import (
    # In order of usage or appearance.

def create_boggle_board_samples():
    
    board = [
        ["t", "h", "i", "s", "i", "s", "a"],
        ["s", "i", "m", "p", "l", "e", "x"],
        ["b", "x", "x", "x", "x", "e", "b"],
        ["x", "o", "g", "g", "l", "x", "o"],
        ["x", "x", "x", "D", "T", "r", "a"],
        ["R", "E", "P", "E", "A", "d", "x"],
        ["x", "x", "x", "x", "x", "x", "x"],
        ["N", "O", "T", "R", "E", "-", "P"],
        ["x", "x", "D", "E", "T", "A", "E"]]

    words = [
        "this",
        "is",
        "not",
        "a",
        "simple",
        "boggle",
        "board",
        "test",
        "REPEATED",
        "NOTRE-PEATED"]

    output = ["this", "is", "a", "simple", "boggle", "board", "NOTRE-PEATED"]

    return board, words, output

