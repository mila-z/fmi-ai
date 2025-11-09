import math
import time

def solvable(board, size, zero_pos):
    """
    checks whether a given sliding configuration is solvable

    inversion - an inversion occurs when a high-number tile appears before a low-number tile (i.e. 132)
    
    - size is odd (i.e. 3x3, 5x5) then the solvability depends on the number of inversions being even
    - size is even (i.e. 4x4, 6x6) then the solvability depends on the number of inversions summed with the row where the blank tile is being odd
    """
    tiles = [x for x in board if x != 0] # 0 is not counted in inversions
    inversions = 0
    for i in range(len(tiles)): 
        for j in range(i + 1, len(tiles)):
            if tiles[i] > tiles[j]:
                inversions += 1

    if size%2 == 0:
        zero_row = zero_pos // size 
        return (inversions + zero_row) % 2 == 1
    else:
        return inversions % 2 == 0
    
def manhattan(board, goal_pos, size):
    """
    calculates the manhattan euristic, i.e. how far the board is from the goal
    does that by summing how many grid moves each tile needs to get to its correct place - uses distance and not tiles

    manhattan distance between 2 points (x1, y1) and (x2, y2) is |x1 - x2| + |y1 - y2|
    """
    total = 0
    for i, tile in enumerate(board):
        if tile == 0:
            continue
        goal_i = goal_pos[tile]
        # 5 // 3 = 1 -> row and 5 % 3 = 2 -> column
        x1, y1 = divmod(i, size) 
        x2, y2 = divmod(goal_i, size)
        total += abs(x1 - x2) + abs(y1 - y2)
    return total

def neighbors(zero_pos, size):
    """
    returns the position of the empty tile and the step that has been taken for each possible move from the current board 
    with position of the empty tile at zero_pos
    """
    moves = [
        (-1, 0, "down"),
        (1, 0, "up"),
        (0, -1, "right"),
        (0, 1, "left")
    ]

    zero_x, zero_y = divmod(zero_pos, size)
    neighbors = []

    for x, y, move in moves:
        # calculate new position of blank tile
        new_zero_x, new_zero_y = zero_x + x, zero_y + y
        # check if it is valid
        if 0 <= new_zero_x < size and 0 <= new_zero_y < size:
            new_zero_pos = new_zero_x * size + new_zero_y
            neighbors.append((new_zero_pos, move))
    return neighbors

def dfs(path, g, bound, zero_pos, size, goal, goal_pos, curr_h):
    """
    path = list of boards from start to curr
    g = actual cost
    bound  = cuurent ida* cutoff
    zero_pos = where the blank tile is
    size = size of the board
    goal = target board
    goal_pos = mapping tile -> goal index
    current_h = current heuristic
    """
    # get the current state which is the last in the path
    board = path[-1]
    # compute the f-cost
    f = g + curr_h
    # if f is more than the bound, prune the branch
    if f > bound:
        return f, None
    # if we have found the goal
    if board == goal:
        return True, []
    
    # store the smallest f-cost that exceeds the limit on this branch
    min_over = float('inf')

    for new_zero, move in neighbors(zero_pos, size):
        # generate the new_board
        new_board = list(board)
        # get the tile that is in the new_zero_pos
        moved_tile = new_board[new_zero]
        # swap
        new_board[zero_pos], new_board[new_zero] = new_board[new_zero], new_board[zero_pos]
        # transform back to tuple
        new_board = tuple(new_board)

        # avoid a step back
        if len(path) > 1 and new_board == path[-2]:
            continue

        # calculate new heuristic
        # get the goal index of the moved tile 
        goal_i = goal_pos[moved_tile]
        # get its coordinates
        goal_x, goal_y = divmod(goal_i, size)
        # get old and new positions of moved tile
        old_x, old_y = divmod(new_zero, size)
        new_x, new_y = divmod(zero_pos, size)
        # get distances
        old_dist = abs(old_x - goal_x) + abs(old_y - goal_y)
        new_dist = abs(new_x - goal_x) + abs(new_y - goal_y)
        # new heuristic
        new_h = curr_h - old_dist + new_dist

        # add board to path
        path.append(new_board)
        # call dfs with the updated path, updated cost, new empty tile pos, goal board, goal tile positions, new heuristic
        result, moves = dfs(path, g + 1, bound, new_zero, size, goal, goal_pos, new_h)
        path.pop()

        # found a solution, build the path
        if result is True:
            return True, [move] + moves
        
        # get the smallest f 
        if result < min_over:
            min_over = result

    return min_over, None
    
def ida_star(start, goal, size, goal_pos, zero_pos):
    """
    run the ida_star alg
    start = start state
    goal = goal_state
    size = size of the board
    goal_pos = positions of the tiles in the goal_state
    zero_pos = position of the empty tile
    """
    h0 = manhattan(start, goal_pos, size)
    bound = h0
    path = [start]

    while True:
        result, moves = dfs(path, 0, bound, zero_pos, size, goal, goal_pos, h0)
        if result is True:
            return moves
        if result == float('inf'):
            return -1
        bound = result


def main():
    # get the input
    n = int(input().strip())
    i = int(input().strip())
    size = int(math.sqrt(n+1))

    tiles = []
    for _ in range(size):
        row = list(map(int, input().split()))
        tiles.extend(row)

    start = tuple(tiles)

    # get the goal
    if i == -1:
        goal = tuple(list(range(1, n + 1)) + [0])
    else:
        goal = tuple(list(range(1, i + 1)) + [0] + list(range(i + 1, n + 1)))

    # get the positions of the tiles in the goal and the current position of the empty tile
    goal_pos = {tile: j for j, tile in enumerate(goal)}
    zero_pos = start.index(0)

    if not solvable(start, size, zero_pos):
        print(-1)
        return 
    
    # start = time.time()
    result = ida_star(start, goal, size, goal_pos, zero_pos)

    if result == -1:
        print(-1)
    else:
        print(len(result))
        for move in result:
            print(move)

    # print(f"Time: {time.time() - start:.4f} s")

if __name__ == "__main__":
    main()