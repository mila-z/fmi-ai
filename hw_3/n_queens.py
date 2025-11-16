import random
import time

def print_board(queen_pos):
    """
    print the board
    """
    n = len(queen_pos)
    for r in range(n):
        row_str = ["*" if queen_pos[r] == c else "_" for c in range(n)]
        print("".join(row_str))

def initialize(n):
    """
    initialize the board with n queens
    """
    # initially every queen is in the first column of every row
    queen_pos = [0] * n

    # counts the conflicts for each column 
    col_confl = [0]*n
    # counts the conflict for positive diagonals -> /
    pos_diag_confl = [0]*(2*n - 1)
    # counts the conflict for negative diagonals -> \
    neg_diag_confl = [0]*(2*n - 1)

    return queen_pos, col_confl, pos_diag_confl, neg_diag_confl

def place_random(n, queen_pos, col_confl, pos_diag_confl, neg_diag_confl):
    """
    places the queens on the board in a random matter
    """
    for row in range(n):
        # geneate a random column
        col = random.randint(0, n - 1)
        # update the position of the queens and the conflicts accordingly
        queen_pos[row] = col
        col_confl[col] += 1
        pos_diag_confl[row + col] += 1
        neg_diag_confl[n - 1 - row + col] += 1

def place_pattern(n, queen_pos, col_confl, pos_diag_confl, neg_diag_confl):
    """
    places the queens on the board in a pattern - with two columns space inbetween
    often has fewer conflicts than a completely random one
    """
    # start with the first column
    col = 1
    for row in range(n):
        # place the queen
        queen_pos[row] = col
        # update the conflicts
        col_confl[col] += 1
        pos_diag_confl[row+col] += 1
        neg_diag_confl[n - 1 - row + col] += 1

        # update the column
        col+=2
        if col >= n:
            col = 0

def confl_count(row, col, n, col_confl, pos_diag_confl, neg_diag_confl):
    """
    returns the total number of queens attacking position (row, col) including the queen that might already be there
    """
    return (
        col_confl[col]
        + pos_diag_confl[row+col]
        + neg_diag_confl[n - 1 - row+col]
    )

def row_max_confl(n, queen_pos, col_confl, pos_diag_confl, neg_diag_confl):
    """
    returns the row with maximum conflics by which we decide which queen to move
    """
    # list of rows that currently have the maximum conflicts
    worst_rows = []
    # current highest conflict count
    max_confl = -1

    for row in range(n):
        # get the current column of the queen
        col = queen_pos[row]
        # get the number of conflicts - includes it 3 times, so we subtract 3
        c = confl_count(row, col, n, col_confl, pos_diag_confl, neg_diag_confl) - 3

        # if this row more conflicts than what we have seen so far, update and reset
        if c > max_confl:
            max_confl = c
            worst_rows = [row]
        # otherwise its an amigo so append
        elif c == max_confl:
            worst_rows.append(row)

    # no queen is attacked so its a solution
    if max_confl == 0:
        return -1
    
    return random.choice(worst_rows)

def col_min_confl(row, n, queen_pos, col_confl, pos_diag_confl, neg_diag_confl):
    """
    returns to col with minimum conflicts by which we decide which position to move the queen
    """
    # columns achieving the current min
    best_cols = []
    # the current minimum
    min_confl = float("inf")

    for col in range(n):
        # get the number of conflicts for (row, col)
        c = confl_count(row, col, n, col_confl, pos_diag_confl, neg_diag_confl)

        # if the queen is there, subtract 3 (we count it 3 times)
        if queen_pos[row] == col:
            c -= 3

        # if this is the least so far, update and reset
        if c < min_confl:
            min_confl = c
            best_cols = [col]
        # otherwise its an amigo
        elif c == min_confl:
            best_cols.append(col)

    return random.choice(best_cols)

def update_confl(n, row, new_col, queen_pos, col_confl, pos_diag_confl, neg_diag_confl):
    """
    update the conflict arrays after moving a queen to pos (row, new_col)
    """
    # get the old column
    old_col = queen_pos[row]

    # update the arrays for the old column
    col_confl[old_col] -=1
    pos_diag_confl[row + old_col] -=1
    neg_diag_confl[n - 1 - row + old_col] -=1

    # update the pos of the queen
    queen_pos[row] = new_col

    # update the arrays for the new column
    col_confl[new_col] +=1
    pos_diag_confl[row + new_col] +=1
    neg_diag_confl[n - 1 - row + new_col] +=1

def n_queens(n, max_steps=10000, restarts=50):
    """
    run the algorithm
    """
    # only one way to put 1 queen
    if n == 1:
        return [0]
    # no solution for 2 or 3 queens
    if n in (2, 3):
        return -1
    
    # try up to restarts times 
    for attempt in range(restarts):
        # initialize the board
        queen_pos, col_confl, pos_diag_confl, neg_diag_confl = initialize(n)

        # for each restart try with different placing to escape bad local minima
        if attempt % 2 == 0:
            place_pattern(n, queen_pos, col_confl, pos_diag_confl, neg_diag_confl)
        else:
            place_random(n, queen_pos, col_confl, pos_diag_confl, neg_diag_confl)

        for _ in range(max_steps):
            # get row with maximum conflicts
            r = row_max_confl(n, queen_pos, col_confl, pos_diag_confl, neg_diag_confl)

            # we have found a solution
            if r == -1:
                return queen_pos

            # get the best position in the current configuration
            best_col = col_min_confl(r, n, queen_pos, col_confl, pos_diag_confl, neg_diag_confl)
            # move the queen
            update_confl(n, r, best_col, queen_pos, col_confl, pos_diag_confl, neg_diag_confl)
    
    return -1


if __name__ == "__main__":
    n = int(input().strip())
    # start_time = time.time()
    result = n_queens(n)
    # print("Time:", round(time.time() - start_time, 3), "seconds")
    print(result)