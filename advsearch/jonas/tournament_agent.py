from typing import Tuple
from ..othello.gamestate import GameState
from ..othello.board import Board
from .othello_minimax_mask import evaluate_mask
from .minimax import minimax_move_tournament
from .minimax import SearchTimeout
import time

# Voce pode criar funcoes auxiliares neste arquivo
# e tambem modulos auxiliares neste pacote.
#
# Nao esqueca de renomear 'your_agent' com o nome
# do seu agente.

TIME_LIMIT = 4.7 

EVAL_TEMPLATE = [
    [100, -30, 6, 2, 2, 6, -30, 100],
    [-30, -50, 1, 1, 1, 1, -50, -30],
    [  6,   1, 1, 1, 1, 1,   1,   6],
    [  2,   1, 1, 3, 3, 1,   1,   2],
    [  2,   1, 1, 3, 3, 1,   1,   2],
    [  6,   1, 1, 1, 1, 1,   1,   6],
    [-30, -50, 1, 1, 1, 1, -50, -30],
    [100, -30, 6, 2, 2, 6, -30, 100]
]

def make_move(state : GameState) -> Tuple[int, int]:
    start_time = time.time()
    best_move = None

    for depth in range(1, 20):  # teto alto; o tempo vai parar antes
        elapsed = time.time() - start_time
        if elapsed >= TIME_LIMIT:
            break
        try:
            move = minimax_move_tournament(state, depth, evaluate_custom, 
                                start_time=start_time)
            best_move = move
        except SearchTimeout:
            break

        print(depth)


    return best_move

def heuristica_cantos(board, player):
    opponent = board.opponent(player)

    cantos_player = 0
    cantos_opp = 0

    for x, y in [(0,0), (0,7), (7,0), (7,7)]:
        piece = board.tiles[x][y]

        if piece == player:
            cantos_player += 1
        elif piece == opponent:
            cantos_opp += 1

    return 100 * (cantos_player - cantos_opp)

def heuristica_mobilidade(state, player):
    board = state.get_board()

    player_moves = len(board.legal_moves(player))
    opponent_moves = len(board.legal_moves(board.opponent(player)))

    total = player_moves + opponent_moves

    if total == 0:
        return 0

    return 100 * (player_moves - opponent_moves) / total


def evaluate_custom(state, player:str) -> float:
    """
    Evaluates an othello state from the point of view of the given player. 
    If the state is terminal, returns its utility. 
    If non-terminal, returns an estimate of its value based on your custom heuristic
    :param state: state to evaluate (instance of GameState)
    :param player: player to evaluate the state for (B or W)
    """
    #return 0    # substitua pelo seu codigo

    board = state.get_board()

    opponent = board.opponent(player)

    ans_canto = 0


    if (board.num_pieces(player) + board.num_pieces(opponent)) >= 52:######################
        return board.num_pieces(player) - board.num_pieces(opponent)

    ans_canto = heuristica_cantos(board, player)
            
    ans_mobilidade = heuristica_mobilidade(state, player)

    return (90 * ans_canto) + (10 * ans_mobilidade) #+ (10 * ans_mask)