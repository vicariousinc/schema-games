import argparse
import numpy as np
from collections import defaultdict

from gym.utils.play import play
import pygame

from schema_games.breakout import games
from schema_games.printing import blue

ZOOM_FACTOR = 5
DEFAULT_DEBUG = True
DEFAULT_CHEAT_MODE = False


def play_game(environment_class,
              cheat_mode=DEFAULT_CHEAT_MODE,
              debug=DEFAULT_DEBUG,
              fps=30):
    """
    Interactively play an environment.

    Parameters
    ----------
    environment_class : type
        A subclass of schema_games.breakout.core.BreakoutEngine that represents
        a game. A convenient list is included in schema_games.breakout.games.
    cheat_mode : bool
        If True, player has an infinite amount of lives.
    debug : bool
        If True, print debugging messages and perform additional sanity checks.
    fps : int
        Frame rate per second at which to display the game.
    """
    print blue("-" * 80)
    print blue("Starting interactive game. "
               "Press <ESC> at any moment to terminate.")
    print blue("-" * 80)

    env_args = {
        'return_state_as_image': True,
        'debugging': debug,
    }

    if cheat_mode:
        env_args['num_lives'] = np.PINF

    env = environment_class(**env_args)
    keys_to_action = defaultdict(lambda: env.NOOP, {
            (pygame.K_LEFT,): env.LEFT,
            (pygame.K_RIGHT,): env.RIGHT,
        })

    play(env, fps=fps, keys_to_action=keys_to_action, zoom=ZOOM_FACTOR)


if __name__ == '__main__':
    """
    Command line interface.
    """
    parser = argparse.ArgumentParser(
        description='Play interactively one of breakout game variants.',
        usage='play.py [<Game>] [--debug]')

    parser.add_argument(
        'game',
        default='StandardBreakout',
        type=str,
        help="Game variant specified as class name."
    )

    parser.add_argument(
        '--debug',
        dest='debug',
        default=DEFAULT_DEBUG,
        action='store_true',
        help="Print debugging messages and perform additional sanity checks."
    )

    parser.add_argument(
        '--cheat_mode',
        dest='cheat_mode',
        default=DEFAULT_CHEAT_MODE,
        action='store_true',
        help="Cheat mode: infinite lives."
    )

    options = parser.parse_args()
    variant = options.game
    debug = options.debug
    cheat_mode = options.cheat_mode

    play_game(getattr(games, variant), debug=debug, cheat_mode=cheat_mode)
