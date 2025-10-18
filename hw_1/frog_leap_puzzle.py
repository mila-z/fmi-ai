def is_goal_state(board, zero_state):
    mid = len(board)//2
    return (
        zero_state == mid and 
        '>' not in board[:mid] and 
        '<' not in board[mid + 1:])

def moves(board, zero_state):
    res = []
    n = len(board)

    def swap(new_pos):
        temp = list(board)
        temp[zero_state], temp[new_pos] = temp[new_pos], temp[zero_state]
        return tuple(temp)

    # think of better var names
    for step in (1, 2):
        jump = zero_state - step
        if jump >= 0 and board[jump] == '>':
            res.append((swap(jump), jump))

        jump = zero_state + step
        if jump < n and board[jump] == '<':
            res.append((swap(jump), jump))

    return res
    

def dfs(board, zero_state, visited=None, path=None):
    if visited is None:
        visited = set()
    if path is None:
        path = []

    if board in visited:
        return None
    visited.add(board)
    path.append(''.join(board))

    if is_goal_state(board, zero_state):
        return path.copy() #why a copy
    
    for move, new_state in moves(board, zero_state):
        result = dfs(move, new_state, visited, path)
        if result:
            return result
        
    path.pop()
    return None

def get_board(n):
    return tuple(['>']*n + ['_'] + ['<']*n)

n = int(input())
# zero_state = n
board = get_board(n)
# board_tuple = tuple(board)
# print(board_tuple)
# print(len(board_tuple))
# print(moves(board, zero_state))
# print(is_goal_state(('<', '<', '_', '>', '>'), 2))

sol = dfs(board, n)
if sol:
    for step in sol:
        print(step)