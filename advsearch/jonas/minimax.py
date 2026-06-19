import random
from typing import Tuple, Callable
import time
from typing import Callable, Tuple
#from .othello_minimax_mask import EVAL_TEMPLATE


TIME_LIMIT = 4.7
CHECK_INTERVAL = 200

class SearchTimeout(Exception):
    pass

_node_count: int = 0

def minimax_move(state, max_depth: int, eval_func: Callable,###minimax_move usado para os testes entre heurísticas, possui apenas aprofundamento iterativo e alpha-beta
                 start_time: float = None) -> Tuple[int, int]:
    global _node_count
    _node_count = 0

    if start_time is None:
        start_time = time.time()

    root_player = state.player
    legal = list(state.legal_moves())

    if not legal:
        return None

    best_move = legal[0]
    alpha = -float('inf')
    beta = float('inf')

    for move in legal:
        new_state = state.next_state(move)
        v = _minimo(new_state, max_depth, eval_func, 1, alpha, beta,
                    root_player, start_time)
        if v > alpha:
            alpha = v
            best_move = move

    return best_move


def _maximo(state, max_depth, eval_func, depth, alpha, beta, root_player, start_time):
    global _node_count
    _node_count += 1
    if _node_count % CHECK_INTERVAL == 0:
        if time.time() - start_time >= TIME_LIMIT:
            raise SearchTimeout()

    if state.is_terminal() or depth == max_depth:
        return eval_func(state, root_player)

    v = -float('inf')
    for move in state.legal_moves():
        new_state = state.next_state(move)
        v = max(v, _minimo(new_state, max_depth, eval_func, depth + 1,
                           alpha, beta, root_player, start_time))
        alpha = max(alpha, v)
        if alpha >= beta:
            break
    
    return v


def _minimo(state, max_depth, eval_func, depth, alpha, beta, root_player, start_time):
    global _node_count
    _node_count += 1
    if _node_count % CHECK_INTERVAL == 0:
        if time.time() - start_time >= TIME_LIMIT:
            raise SearchTimeout()

    if state.is_terminal() or depth == max_depth:
        return eval_func(state, root_player)

    v = float('inf')
    for move in state.legal_moves():
        new_state = state.next_state(move)
        v = min(v, _maximo(new_state, max_depth, eval_func, depth + 1,
                           alpha, beta, root_player, start_time))
        beta = min(beta, v)
        if beta <= alpha:
            break
    
    return v


def minimax_move_ordenado(state, max_depth: int, eval_func: Callable,###usado no agente do torneio, ordena os nodos para chegar mais rapido nos cortes alfa-beta
                 start_time: float = None) -> Tuple[int, int]:
    global _node_count
    _node_count = 0

    if start_time is None:
        start_time = time.time()

    root_player = state.player
    legal = list(state.legal_moves())

    if not legal:
        return None

    best_move = legal[0]
    alpha = -float('inf')
    beta = float('inf')

    for move in _order_moves(legal):
        new_state = state.next_state(move)
        v = _minimo_ordenado(new_state, max_depth, eval_func, 1, alpha, beta,
                    root_player, start_time)
        if v > alpha:
            alpha = v
            best_move = move

    return best_move


def _maximo_ordenado(state, max_depth, eval_func, depth, alpha, beta, root_player, start_time):
    global _node_count
    _node_count += 1
    if _node_count % CHECK_INTERVAL == 0:
        if time.time() - start_time >= TIME_LIMIT:
            raise SearchTimeout()

    if state.is_terminal() or depth == max_depth:
        return eval_func(state, root_player)

    v = -float('inf')
    for move in _order_moves(state.legal_moves()):
        new_state = state.next_state(move)
        v = max(v, _minimo_ordenado(new_state, max_depth, eval_func, depth + 1,
                           alpha, beta, root_player, start_time))
        alpha = max(alpha, v)
        if alpha >= beta:
            break

    return v


def _minimo_ordenado(state, max_depth, eval_func, depth, alpha, beta, root_player, start_time):
    global _node_count
    _node_count += 1
    if _node_count % CHECK_INTERVAL == 0:
        if time.time() - start_time >= TIME_LIMIT:
            raise SearchTimeout()

    if state.is_terminal() or depth == max_depth:
        return eval_func(state, root_player)

    v = float('inf')
    for move in _order_moves(state.legal_moves()):
        new_state = state.next_state(move)
        v = min(v, _maximo_ordenado(new_state, max_depth, eval_func, depth + 1,
                           alpha, beta, root_player, start_time))
        beta = min(beta, v)
        if beta <= alpha:
            break

    return v

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


def _order_moves(moves):
    return sorted(moves, key=lambda m: EVAL_TEMPLATE[m[0]][m[1]], reverse=True)

def minimax_move_simples(state, max_depth:int, eval_func:Callable) -> Tuple[int, int]:######minimax com poda alpha-beta simples
    """
    Returns a move computed by the minimax algorithm with alpha-beta pruning for the given game state.
    :param state: state to make the move (instance of GameState)
    :param max_depth: maximum depth of search (-1 = unlimited)
    :param eval_func: the function to evaluate a terminal or leaf state (when search is interrupted at max_depth)
                    This function should take a GameState object and a string identifying the player,
                    and should return a float value representing the utility of the state for the player.
    :return: (int, int) tuple with x, y coordinates of the move (remember: 0 is the first row/column)
    """

    rootPlayer = state.player
    movimentos_legais = state.legal_moves()
    if not movimentos_legais:
        return None
    best_move = next(iter(movimentos_legais))
    v = -float('inf')
    alpha = -float('inf')
    beta = float('inf')

    for s in state.legal_moves():
        newState = state.next_state(s)

        v_linha = minimo(newState, max_depth, eval_func, 1, alpha, beta, rootPlayer)
        if v_linha > v:
            v = v_linha
            best_move = s

        alpha = max(alpha, v)

    return best_move


def maximo(state, max_depth, eval_func, depth, alpha, beta, rootPlayer):
    if state.is_terminal() or depth == max_depth:
        return eval_func(state, rootPlayer)
    
    v = -float('inf')

    for s in state.legal_moves():
        newState = state.next_state(s)

        v_linha = minimo(newState, max_depth, eval_func, depth+1, alpha, beta, rootPlayer)
        v = max(v,v_linha)

        alpha = max(alpha, v)
        if alpha >= beta:
            break

    return v

def minimo(state, max_depth, eval_func, depth, alpha, beta, rootPlayer):
    if state.is_terminal() or depth == max_depth:
        return eval_func(state, rootPlayer)
    
    v = float('inf')

    for s in state.legal_moves():
        newState = state.next_state(s)

        v_linha = maximo(newState, max_depth, eval_func, depth+1, alpha, beta, rootPlayer)
        v = min(v,v_linha)

        beta = min(beta, v)
        if beta <= alpha:
            break

    return v