import time
from typing import Callable, Tuple
from .othello_minimax_mask import EVAL_TEMPLATE

TIME_LIMIT = 4.7
CHECK_INTERVAL = 200

class SearchTimeout(Exception):
    pass

_node_count: int = 0


def minimax_move(state, max_depth: int, eval_func: Callable,
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
    for move in _order_moves(state.legal_moves()):
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
    for move in _order_moves(state.legal_moves()):
        new_state = state.next_state(move)
        v = min(v, _maximo(new_state, max_depth, eval_func, depth + 1,
                           alpha, beta, root_player, start_time))
        beta = min(beta, v)
        if beta <= alpha:
            break

    return v


def _order_moves(moves):
    return sorted(moves, key=lambda m: EVAL_TEMPLATE[m[0]][m[1]], reverse=True)