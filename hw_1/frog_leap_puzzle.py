def is_goal_state(board, zero_state):
    return zero_state == len(board)//2 and not('>' in board[:zero_state]) and not('<' in board[(zero_state + 1):])

def moves(board, zero_state):
    moves = []
    board = list(board)
    board_len = len(board)

    # think of better var names
    right = zero_state - 1
    double_right = zero_state - 2
    left  = zero_state + 1
    double_left = zero_state + 2

    if right >= 0 and board[right] == '>':
        board_right = board.copy()
        board_right[right], board_right[zero_state] = board_right[zero_state], board_right[right]
        moves.append((tuple(board_right), right))
    
    if double_right >= 0 and board[double_right] == '>':
        board_double_right = board.copy()
        board_double_right[double_right], board_double_right[zero_state] = board_double_right[zero_state], board_double_right[double_right]
        moves.append((tuple(board_double_right), double_right))

    if left < board_len and board[left] == '<':
        board_left = board.copy()
        board_left[left], board_left[zero_state] = board_left[zero_state], board_left[left]
        moves.append((tuple(board_left), left))

    if double_left < board_len and board[double_left] == '<':
        board_double_left = board.copy()
        board_double_left[double_left], board_double_left[zero_state] = board_double_left[zero_state], board_double_left[double_left]
        moves.append((tuple(board_double_left), double_left))

    return moves
    

def dfs(board, zero_state, visited=None, path=None):
    if visited is None:
        visited = set()
    if path is None:
        path = []

    if board in visited:
        return None
    visited.add(board)

    if is_goal_state(board, zero_state):
        return path + [''.join(board)]
    
    for move, new_state in moves(board, zero_state):
        result = dfs(move, new_state, visited, path + [''.join(board)])
        if result:
            return result
        

    return None

def get_board(n):
    return tuple(['>']*n + ['_'] + ['<']*n)

n = int(input())
zero_state = n
board = get_board(n)
# board_tuple = tuple(board)
# print(board_tuple)
# print(len(board_tuple))
# print(moves(board, zero_state))
# print(is_goal_state(('<', '<', '_', '>', '>'), 2))

sol = dfs(board, zero_state)
if sol:
    for step in sol:
        print(step)