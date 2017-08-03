"""
The constants `_MAX_SPEED` and `_BALL_SHAPE` are absolutely immutable (i.e.,
the game engine was built assuming these values to be immutable). Some of the
remaining constants may be changed with caution. Proceed at your own risk.
"""

import numpy as np


###############################################################################
#
#   DO NOT CHANGE !!!
#   -----------------
#   These constants determine the maximum radius up to which graceful dynamics
#   are guaranteed without too many visual artifacts. Recommended value: 2.
#
_MAX_SPEED = 2
_BALL_SHAPE = np.array([1, 1])
###############################################################################

###############################################################################
# Maximum number of pixels per entity. Knowing this in advance allows us to
# optimize object creation and tracking.
###############################################################################
MAX_NZIS_PER_ENTITY = 100

ALLOW_BOUNCE_AGAINST_PHYSICS = False
BOUNCE_STOCHASTICITY = 0.25
CORRUPT_RENDERED_IMAGE = False
DEBUGGING = False
DEFAULT_BRICK_REWARD = 1
DEFAULT_BRICK_SHAPE = np.array([8, 4])
DEFAULT_NUM_BRICKS_COLS = 11
DEFAULT_NUM_BRICKS_ROWS = 6
DEFAULT_WALL_THICKNESS = 3
DEFAULT_WIDTH = (DEFAULT_WALL_THICKNESS * 2 +
                 DEFAULT_NUM_BRICKS_COLS *
                 DEFAULT_BRICK_SHAPE[0])
DEFAULT_HEIGHT = int(1.25 * DEFAULT_WIDTH)
DEFAULT_PADDLE_SHAPE = np.array([int(DEFAULT_WIDTH * .25), 4])
EXCLUDED_VELOCITIES = frozenset(
    {(u*(v+1), 0) for u in (-1, 1) for v in range(_MAX_SPEED)})
NUM_BALLS = 1
NUM_LIVES = 3  # Inf is useful for effortless debugging
PADDLE_SPEED = 2
PADDLE_SPEED_DISTRIBUTION = np.zeros((2 * PADDLE_SPEED + 1,))
PADDLE_SPEED_DISTRIBUTION[-1] = 0.90
PADDLE_SPEED_DISTRIBUTION[-2] = 0.10
PADDLE_STARTING_POSITION = (None, None)
REWARD_UPON_BALL_LOSS = -1
REWARD_UPON_NO_BRICKS_LEFT = 0
STARTING_BALL_MOVEMENT_RADIUS = 1

###############################################################################
#   Color constants
###############################################################################
CLASSIC_BACKGROUND_COLOR = (0, 0, 0)
CLASSIC_BALL_COLOR = (200, 72, 73)
CLASSIC_BRICK_COLORS = [(66, 72, 200), (72, 160, 72), (163, 162, 42),
                        (180, 122, 48), (198, 108, 58), (200, 72, 73)]
CLASSIC_PADDLE_COLOR = (200, 72, 73)
CLASSIC_WALL_COLOR = (142, 142, 142)
DEFAULT_PADDLE_COLOR = CLASSIC_BALL_COLOR
