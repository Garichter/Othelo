from typing import Tuple
from ..othello.gamestate import GameState
from ..othello.board import Board
from .othello_minimax_mask import evaluate_mask
from .minimax_custom import minimax_move
from .minimax_custom import SearchTimeout
import time

# Voce pode criar funcoes auxiliares neste arquivo
# e tambem modulos auxiliares neste pacote.
#
# Nao esqueca de renomear 'your_agent' com o nome
# do seu agente.

TIME_LIMIT = 4.7 

def make_move(state : GameState) -> Tuple[int, int]:
    start_time = time.time()
    best_move = None

    for depth in range(1, 20):  # teto alto; o tempo vai parar antes
        elapsed = time.time() - start_time
        if elapsed >= TIME_LIMIT:
            break
        try:
            move = minimax_move(state, depth, evaluate_custom, 
                                start_time=start_time)
            best_move = move
        except SearchTimeout:
            break

    return best_move


def evaluate_custom(state : GameState, player:str) -> float:
    phase = game_phase(state)

    
    mob        = mobility_score(state, player)
    pos        = evaluate_mask(state, player)
    pdiff      = piece_difference_score(state.board, player)

    if phase == 'early':
        return (
            100 * mob +
             30 * pos +
              5 * pdiff
        )
    elif phase == 'mid':
        return (
             70 * mob +
             50 * pos +
             15 * pdiff
        )
    else:
        return (
             30 * mob +
             20 * pos +
            100 * pdiff
        )


def mobility_score(state : GameState, player):
    """Mobilidade relativa: movimentos meus vs do oponente"""
    my_moves = len(state.board.legal_moves(player))

    opponent = 'W' if player == 'B' else 'B'
    
    opponent_moves = len(state.board.legal_moves(opponent))
    

    if my_moves + opponent_moves == 0:
        return 0
    
    return 100 * (my_moves - opponent_moves) / (my_moves + opponent_moves + 1)

def piece_difference_score(board : Board, player):
    player_pieces = board.piece_count[player]
    opponent = 'W' if player == 'B' else 'B'
    opp_pieces = board.piece_count[opponent]
    total = player_pieces + opp_pieces
    if total == 0:
        return 0
    return 100 * (player_pieces - opp_pieces) / total

def game_phase(state):
    total = state.board.piece_count['W'] + state.board.piece_count['B']
    
    if total < 20:
        return 'early'
    elif total < 50:
        return 'mid'
    else:
        return 'late'
