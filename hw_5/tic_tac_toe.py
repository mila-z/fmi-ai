import sys
import math

def parse_input():
    lines = [line.rstrip('\n') for line in sys.stdin]
    mode = lines[0]

    turn = lines[1].split()[1]
    if turn == "0":
        turn = "O"

    board = []
    for line in lines[2:]:
        if '|' in line:
            cells = [c.strip() for c in line.split('|')[1:4]]
            board.append(cells)
    
    return mode, turn, board

def is_winner(board, player):
    lines = (
        board +
        [list(col) for col in zip(*board)] +
        [[board[i][i] for i in range(3)],
         [board[i][2-i] for i in range(3)]]
    )
    return any(all(cell == player for cell in line) for line in lines)

def is_terminal(board):
    if is_winner(board, 'X') or is_winner(board, 'O'):
        return True
    return all(cell != '_' for row in board for cell in row)

def minimax(board, alpha, beta, is_maximizing, ai, opp, depth):
    if is_winner(board, ai):
        return 10 - depth
    if is_winner(board, opp):
        return depth - 10
    if is_terminal(board):
        return -depth
    
    if is_maximizing:
        best = -math.inf
        for r in range(3):
            for c in range(3):
                if board[r][c] == '_':
                    board[r][c] = ai
                    score = minimax(board, alpha, beta, False, ai, opp, depth + 1)
                    board[r][c] = '_'
                    best = max(best, score)
                    alpha = max(alpha, best)
                    if beta <= alpha:
                        return best
        return best
    else:
        best = math.inf
        for r in range(3):
            for c in range(3):
                if board[r][c] == '_':
                    board[r][c] = opp
                    score = minimax(board, alpha, beta, True, ai, opp, depth + 1)
                    board[r][c] = '_'
                    best = min(best, score)
                    beta = min(beta, best)
                    if beta <= alpha:
                        return best
        return best
    
def best_move(board, turn):
    ai = turn
    opp = 'O' if ai == 'X' else 'X'

    best_score = -math.inf if ai == 'X' else math.inf
    move = None

    for r in range(3):
        for c in range(3):
            if board[r][c] =='_':
                board[r][c] = ai
                score = minimax(
                    board, 
                    -math.inf,
                    math.inf,
                    False,
                    ai,
                    opp,
                    1
                )
                board[r][c] = '_'

                if ai == 'X' and score > best_score:
                    best_score = score
                    move = (r, c)
                if ai == 'O' and score < best_score:
                    best_score = score
                    move = (r, c)

    return move

def main():
    mode, turn, board = parse_input()

    if is_terminal(board):
        print(-1)
        return 
    
    move = best_move(board, turn)
    if move is None:
        print(-1)
        return
    
    r, c = move
    print(r + 1, c + 1)

if __name__ == "__main__":
    main()