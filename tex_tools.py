"""
Created on Wed Aug 12 01:37:16 2015

@author: richard
"""

import re

VAR_RE = '[a-z][\d_]*'

def matrix_to_tex(matrix_):
    lines = []
    for line in matrix_:
        lines.append(' & '.join(map(str, line)) + ' \\\\')
    return '\n'.join(lines)

def _var_repler(var):
    var = var.group()
    if len(var) == 1:
        return var[0]
    var_letter, subscript = var[0], var[1:]
    if subscript[0] == '_':
        subscript = subscript[1:]
    subscript = subscript.replace('_', '')
    return '{}_{{{}}}'.format(var_letter, subscript)


def var_to_tex(var):
    return re.sub(VAR_RE, _var_repler, str(var).replace('_', ''))

def expr_to_tex(expr):
    expr = str(expr).replace(' ', '').replace('**1.0', '')

    tex = re.sub(VAR_RE, _var_repler, expr).replace('*', '')
    return tex

def eqn_to_tex(eqn):
    eqn = str(eqn).replace(' ', '')

    expr1, expr2 = eqn.split('==')

    tex = '{} &= {}'.format(expr_to_tex(expr1), expr_to_tex(expr2))
    return tex


def eqns_to_tex(eqns):
    ''' To convert to array environment, copy the output into a lyx LaTeX cell,
        then copy this entire cell into an eqnarray of sufficient size
    '''
    return '\\\\'.join(map(eqn_to_tex, eqns))