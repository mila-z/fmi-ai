import random

def conflicts(board, row, col):
    count = 0

    for r in range(len(board)):
        c = board[r]
        if r == row:
            continue

        if c == col:
            count += 1

        if abs(r - row) == abs(c - col):
            count += 1

    return count

def min_conflicts(n, max_steps=100000):
    if n == 2 or n == 3:
        return -1
    
    board = [random.randrange(n) for _ in range(n)]

    for step in range(max_steps):
        conflicted = []
        for row in range(n):
            if conflicts(board, row, board[row]) > 0:
                conflicted.append(row)
        
        if not conflicted:
            return board
        
        row = random.choice(conflicted)

        min_conf = n + 1
        best_cols = []
        for col in range(n):
            c = conflicts(board, row, col)
            if c < min_conf:
                min_conf = c
                best_cols = [col]
            elif c == min_conf:
                best_cols.append(col)

        board[row] = random.choice(best_cols)

    return None

n = int(input("Enter n:"))
print(min_conflicts(n))