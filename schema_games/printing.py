"""
Print using terminal colors.
"""


def red(text):
    return '\033[{:d}m{}\033[0m'.format(31, text)


def green(text):
    return '\033[{:d}m{}\033[0m'.format(32, text)


def yellow(text):
    return '\033[{:d}m{}\033[0m'.format(33, text)


def blue(text):
    return '\033[{:d}m{}\033[0m'.format(34, text)


def purple(text):
    return '\033[{:d}m{}\033[0m'.format(35, text)


def cyan(text):
    return '\033[{:d}m{}\033[0m'.format(36, text)


def white(text):
    return '\033[{:d}m{}\033[0m'.format(37, text)
