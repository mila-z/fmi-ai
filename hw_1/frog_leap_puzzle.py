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
        return ''.join(temp)

    # think of better var names
    for step in (1, 2):
        left = zero_state - step
        if left >= 0 and board[left] == '>':
            res.append((swap(left), left))

        right = zero_state + step
        if right < n and board[right] == '<':
            res.append((swap(right), right))

    return res

def is_dead_state(board, zero_state):
    mid = len(board) // 2
    if zero_state < mid and '<' not in board[zero_state+1:]:
        return True
    
    if zero_state > mid and '>' not in board[:zero_state]:
        return True
    
    return False
    

def dfs(board, zero_state, visited=None, path=None):
    if visited is None:
        visited = set()
    if path is None:
        path = []

    if board in visited or is_dead_state(board, zero_state):
        return None
    
    visited.add(board)
    path.append(board)

    if is_goal_state(board, zero_state):
        return path.copy() 
    
    for move, new_state in moves(board, zero_state):
        result = dfs(move, new_state, visited, path)
        if result:
            return result
        
    path.pop()
    return None

def get_board(n):
    return '>'*n + '_' + '<'*n

n = int(input())
board = get_board(n)

sol = dfs(board, n)
if sol:
    for step in sol:
        print(step)