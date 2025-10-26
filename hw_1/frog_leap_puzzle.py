def moves(board, zero_state):
    res = []
    n = len(board)

    def swap(new_pos):
        temp = list(board)
        temp[zero_state], temp[new_pos] = temp[new_pos], temp[zero_state]
        return ''.join(temp)

    for step in (1, 2):
        right = zero_state - step
        if right >= 0 and board[right] == '>':
            res.append((swap(right), right))

        left = zero_state + step
        if left < n and board[left] == '<':
            res.append((swap(left), left))

    return res

def dfs_it(start, zero_state, goal):
    stack = [(start, zero_state, [start])]
    visited = set([start])

    while stack:
        board, z, path = stack.pop()
        if board == goal:
            return path
        
        for move, next_z in moves(board, z):
            if move not in visited:
                visited.add(move)
                stack.append((move, next_z, path + [move]))

    return None

def get_board(n):
    return '>'*n + '_' + '<'*n

def get_goal(n):
    return '<'*n + '_' + '>'*n

n = int(input('Enter n:'))
board = '>'*n + '_' + '<'*n
goal = '<'*n + '_' + '>'*n
sol = dfs_it(board, n, goal)
if sol:
    for step in sol:
        print(step)