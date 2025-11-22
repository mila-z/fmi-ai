import sys
import math

def parse_board():
    board = []
    while len(board) < 3:
        line = sys.stdin.readline()
        if not line:
            break
        line = line.rstrip('\n')
        if '|' in line:
            cells = [c.strip() for c in line.split('|')[1:4]]
            board.append(cells)
    
    return board

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
        return 0
    
    if is_maximizing:
        best = -math.inf
        for r in range(3):
            for c in range(3):
                if board[r][c] == '_':
                    board[r][c] = ai
                    score = minimax(board, alpha, beta, False, ai, opp, depth + 1)
                    board[r][c] = '_'
                    if score > best:
                        best = score
                    if best > alpha:
                        alpha = best
                    if beta <= alpha:
                        break
        return best
    else:
        best = math.inf
        for r in range(3):
            for c in range(3):
                if board[r][c] == '_':
                    board[r][c] = opp
                    score = minimax(board, alpha, beta, True, ai, opp, depth + 1)
                    board[r][c] = '_'
                    if score < best:
                        best = score
                    if best < beta:
                        beta = best
                    if beta <= alpha:
                        break
        return best
    
def best_move(board, turn):
    ai = turn
    opp = 'O' if ai == 'X' else 'X'

    best_score = -math.inf 
    move = None

    for r in range(3):
        for c in range(3):
            if board[r][c] == '_':
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

                if score > best_score:
                    best_score = score
                    move = (r, c)
    return move


def print_board(board):
    sep = "+---+---+---+"
    print(sep)
    for row in board:
        print("|" + " | ".join(row) + " |")
        print(sep)

def other(player):
    return 'O' if player == 'X' else 'X'

def current_player_from_board(first, board):
    moves = sum(1 for row in board for cell in row if cell != '_')
    if moves % 2 == 0:
        return first
    else:
        return other(first)

def run_judge():
    turn_line = sys.stdin.readline().strip()
    turn_raw = turn_line.split()[1]
    turn = 'O' if turn_raw == '0' else turn_raw

    board = parse_board()

    if is_terminal(board):
        print(-1)
        return 
    
    move = best_move(board, turn)
    if move is None:
        print(-1)
        return
    
    r, c = move
    print(r + 1, c + 1)

def run_game():
    first_line = sys.stdin.readline().strip()
    human_line = sys.stdin.readline().strip()

    first_raw = first_line.split()[1]
    human_raw = human_line.split()[1]

    first = 'O' if first_raw == '0' else first_raw
    human = 'O' if human_raw == '0' else human_raw
    ai = other(human)

    board = parse_board()

    current = current_player_from_board(first, board)

    while not is_terminal(board):
        if current == human:
            line = input()
            if not line:
                break
            parts = line.split()
            if len(parts) != 2:
                continue
            row = int(parts[0]) - 1
            col = int(parts[1]) - 1
            if 0 <= row < 3 and 0 <= col < 3 and board[row][col] == '_':
                board[row][col] = human
        else:
            move = best_move(board, current)
            if move is None:
                break
            r, c = move
            board[r][c] = current
        
        print_board(board)
        current = other(current)

    if is_winner(board, 'X'):
        print("WINNER: X")
    elif is_winner(board, 'O'):
        print("WINNER: O")
    else:
        print("DRAW")

def main():
    mode = sys.stdin.readline().strip()

    if mode == "JUDGE":
        run_judge()
    elif mode == "GAME":
        run_game()
    else:
        pass

if __name__ == "__main__":
    main()