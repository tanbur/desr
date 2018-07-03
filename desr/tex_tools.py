"""
Created on Wed Aug 12 01:37:16 2015

Author: Richard Tanburn (richard.tanburn@gmail.com)
"""

import re
import sympy
from sympy.abc import _clash1

VAR_RE = '[A-Za-z*][\d_]*'

def matrix_to_tex(matrix_):
    '''
    Given a matrix, write out the TeX.

    Args:
        matrix_ (sympy.Matrix): Matrix to turn into TeX
    Returns:
        str

    Printing is the correct way to use this function, but the docstring looks a bit odd.
    >>> print matrix_to_tex(sympy.eye(2))
    1 & 0 \\\\
    0 & 1 \\\\
    '''
    lines = []
    for line in matrix_.tolist():
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
    """
    Given a sympy variable, write out the TeX.

    Args:
        var (sympy.Symbol): Variable to turn into TeX
    Returns:
        str

    >>> print map(var_to_tex, sympy.symbols('x y_1 Kw_3 z_{3} k_m1'))
    ['x', 'y_{1}', 'Kw_{3}', 'z_{3}', 'k_{-1}']
    """
    return expr_to_tex(var)

def expr_to_tex(expr):
    """
    Given a sympy expression, write out the TeX.

    Args:
        expr (sympy.Expression): Expression to turn into TeX
    Returns:
        str

    >>> print map(expr_to_tex, map(sympy.sympify, ['(x + y - 1.5)**2', '(x + y_m1)**1', 'k_m1*t']))
    ['\\\\left(x + y - 1.5\\\\right)^{2}', 'x + y_{-1}', 'k_{-1} t']
    """
    tex = sympy.latex(expr)
    # Substitute _{m...} for _{-...}
    tex = re.sub(r'\_\{?m([^\}]+)\}?', r'_{-\1}', tex)
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

def tex_to_sympy(tex):
    """
    Given some TeX, turn it into sympy expressions and equations.
    Each line is parsed seperately.

    Args:
        tex (str): LaTeX

    Returns:
        list

    >>> lines = ['\\frac{dE}{dt} &= - k_1 E S + k_{-1} C + k_2 C \\\\',
    ... '\\frac{dS}{dt} &= - k_1 E S + k_{-1} C \\\\',
    ... '\\frac{dC}{dt} &= k_1 E S - k_{-1} C - k_2 C \\\\',
    ... '\\frac{dP}{dt} &= k_2 C']
    >>> sym = tex_to_sympy('\\n'.join(lines))
    >>> for s in sym: print s
    Eq(Derivative(E, t), C*k_2 + C*k_m1 - E*S*k_1)
    Eq(Derivative(S, t), C*k_m1 - E*S*k_1)
    Eq(Derivative(C, t), -C*k_2 - C*k_m1 + E*S*k_1)
    Eq(Derivative(P, t), C*k_2)
    """
    # Parse each line individually
    split_tex = tex.split('\n')
    if len(split_tex) > 1:
        return map(tex_to_sympy, split_tex)

    # Remove alignment characters
    tex = tex.strip('\s').replace('&', '').replace('\\\\', '')

    # If equality, return a sympy.Eq
    sides = tex.split('=')
    if len(sides) == 2:
        return sympy.Eq(*map(tex_to_sympy, sides))
    elif len(sides) != 1:
        raise ValueError('Too many = in {}.'.format(tex))

    # Turn \frac{d }{d } into sympy.Derivatives
    diff_match = re.match('\s*\\frac\{d(.+)\}\{d(.+)\}', tex)
    if diff_match:
        return sympy.Derivative(*sympy.symbols(' '.join(diff_match.groups())))

    # Turn \frac into ratios
    tex = re.sub(r'\frac{(.*)}{(.*)}', '((\\1) / (\\2))', tex)
    # Turn spaces between variables into *. Do this by matching anything that isn't an operation
    # Use a lookahead assertion to get overlapping instances.
    tex = re.sub('([^+\-*\s/(]+)\s+(?=[^+\-*\s/)]+)', '\\1 * ', tex)

    # Change minuses in the subscripts to m's
    tex = re.sub('([a-zA-Z]+)_{(.*)\-(.+)}', '\\1_\\2m\\3 ', tex)

    # We want to use all available variables, so sympify with the _clash local dictionary
    return sympy.sympify(tex, _clash1)


if __name__ == '__main__':
    import doctest
    doctest.testmod()