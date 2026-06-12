import random
from typing import Tuple, Callable



def minimax_move(state, max_depth:int, eval_func:Callable) -> Tuple[int, int]:
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

