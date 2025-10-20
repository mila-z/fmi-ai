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

# def is_dead_state(board, zero_state):
#     mid = len(board) // 2
#     if zero_state < mid and '<' not in board[zero_state+1:]:
#         return True
    
#     if zero_state > mid and '>' not in board[:zero_state]:
#         return True
    
#     return False
    

def dfs_rec(board, zero_state, goal, visited, path):
    # if visited is None:
    #     visited = set()
    # if path is None:
    #     path = []

    # if board in visited or is_dead_state(board, zero_state):
    #     return None
    
    visited.add(board)
    path.append(board)

    if board == goal:
        return path.copy() 
    
    for move, new_state in moves(board, zero_state):
        result = dfs_rec(move, new_state, goal, visited, path)
        if result:
            return result
        
    path.pop()
    return None

def dfs_it(start, zero_state, goal):
    stack = [(start, zero_state, [start], {start})]

    while stack:
        board, z, path, visited = stack.pop()
        if board == goal:
            return path
        
        for move, next_z in moves(board, z):
            if move not in visited:
                stack.append((move, next_z, path + [move], visited | {move}))

    return None


def dls(start, zero_state, goal, limit):
    def dls_rec(state, zero_state, depth, path, visited):
        if depth > limit:
            return None
        
        path.append(state)
        visited.add(state)

        if state == goal:
            return path.copy()
        
        for neighbour, next_zero_state in moves(state, zero_state):
            if neighbour not in visited:
                result = dls_rec(neighbour, next_zero_state, depth+1, path, visited)
                if result is not None:
                    return result
            
        path.pop()
        visited.remove(state)
        return None

    return dls_rec(start, zero_state, 0, [], set())

def ids(start, zero_state, goal, max_depth):
    for depth in range(max_depth + 1):
        result = dls(start, zero_state, goal, depth)
        if result is not None:
            return result
        
    return None

def get_board(n):
    return '>'*n + '_' + '<'*n

def get_goal(n):
    return '<'*n + '_' + '>'*n

n = int(input())
max_depth = n*(n+2)
board = get_board(n)
goal = get_goal(n)

# sol = dfs_rec(board, n, goal, set(), [])
# sol = ids(board, n, goal, max_depth)
sol = dfs_it(board, n, goal)
if sol:
    for step in sol:
        print(step)