"""
Game variants for Breakout. As a general rule, game changes, rule variations,
etc. that occur in the subclass' `__init__` method are run only once, when the
game is being constructed, whereas those occurring in `layout` are being run
every new episode (i.e., every time the game is lost or won and the layout of
the game reinitialized). This is important to keep in mind if we care about
randomizing some game properties across episodes or not.
"""

import numpy as np
import random
from itertools import product

from schema_games.breakout.core import BreakoutEngine
from schema_games.breakout.objects import \
    AcceleratorBrick, Brick, \
    HorizontallyMovingObstacle, ResetterBrick, Wall, WallOfPunishment
from schema_games.breakout.events import BallAcceleratesEvent
from schema_games.breakout.constants import \
    _MAX_SPEED, CLASSIC_BACKGROUND_COLOR, CLASSIC_PADDLE_COLOR, \
    CLASSIC_BALL_COLOR, CLASSIC_BRICK_COLORS, CLASSIC_WALL_COLOR, \
    DEFAULT_BRICK_REWARD, DEFAULT_BRICK_SHAPE, DEFAULT_NUM_BRICKS_ROWS


class StandardBreakout(BreakoutEngine):
    """
    Default layout with standard bricks.
    """
    def __init__(self,
                 num_bricks_rows=DEFAULT_NUM_BRICKS_ROWS,
                 brick_shape=DEFAULT_BRICK_SHAPE,
                 brick_reward=DEFAULT_BRICK_REWARD,
                 increment_ball_speed_at_hits=(4, 8, 12),
                 include_accelerator_bricks=True,
                 **kwargs
                 ):
        """
        Parameters
        ----------
        num_bricks_rows : int
            Number of brick rows in the upper half of the environment.
        brick_shape : (int, int)
            Shape of each brick.
        brick_reward : float
            Reward when a common brick gets destroyed.
        increment_ball_speed_at_hits : [int]
            When we hit the brick these many times, the speed of the ball
            increases until we hit the maximum speed allowed.
        include_accelerator_bricks : bool
            One of the layers of bricks permanently accelerates the ball upon
            collision.
        """

        super(StandardBreakout, self).__init__(**kwargs)

        # Attributes specific to standard layout
        self.num_bricks_rows = num_bricks_rows
        self._brick_shape = brick_shape
        self.brick_reward = brick_reward
        self.include_accelerator_bricks = include_accelerator_bricks
        self.increment_ball_speed_at_hits = increment_ball_speed_at_hits

        error_msg = "Physics algorithm will fail"
        assert self._brick_shape[1] >= _MAX_SPEED, error_msg

        # Make sure brick layout does not have any 'gaps'
        assert ((self.width - 2 * self.wall_thickness) %
                self._brick_shape[0] == 0)

        # Setup classic colors
        shape = (self.width, self.height, 3)
        self.background_image = np.zeros(shape, dtype=np.uint8)
        self.background_image[:] = CLASSIC_BACKGROUND_COLOR

        for wall in self.walls:
            if isinstance(wall, WallOfPunishment):
                wall.color = CLASSIC_BACKGROUND_COLOR
            elif isinstance(wall, Wall):
                wall.color = CLASSIC_WALL_COLOR

        # Define in __init__ because they should not be removed
        self.conditional_events = [
            BallAcceleratesEvent(num_brick_hits)
            for num_brick_hits in self.increment_ball_speed_at_hits
        ]

    def layout(self):
        """
        API method. Sets up layers of bricks.
        """
        self.paddle.color = CLASSIC_PADDLE_COLOR

        for ball in self.balls:
            ball.color = CLASSIC_BALL_COLOR

        brick_xs, brick_ys = StandardBreakout.brick_wall_coordinates(
            height=self.height,
            width=self.width,
            num_bricks_rows=self.num_bricks_rows,
            brick_shape=self._brick_shape,
            wall_thickness=self.wall_thickness)

        brick_colors = Brick.brick_colors_classic(len(brick_ys))

        self.bricks = []

        for x in brick_xs:
            for k, (y, col) in enumerate(zip(brick_ys, brick_colors)):
                kwargs = {
                    'reward':   self.brick_reward,
                    'shape':    self._brick_shape,
                    'color':    col,
                }

                if k == self.num_bricks_rows - 4 and \
                   self.include_accelerator_bricks:
                    self.bricks += [AcceleratorBrick((x, y), **kwargs)]
                else:
                    self.bricks += [Brick((x, y), **kwargs)]

    @staticmethod
    def brick_wall_coordinates(height,
                               width,
                               num_bricks_rows,
                               brick_shape,
                               wall_thickness):
        """
        Helper method that returns brick coordinates under the standard
        implementation of the game.
        """
        upper_row = (height - 2 - wall_thickness)
        lower_row = (upper_row - num_bricks_rows * brick_shape[1])
        x = range(wall_thickness, width - wall_thickness, brick_shape[0])
        y = range(lower_row, upper_row, brick_shape[1])

        return x, y


class OffsetPaddleBreakout(StandardBreakout):
    """
    Paddle with a random vertical offset.
    """
    def __init__(self,
                 min_paddle_height=None,
                 max_paddle_height=None,
                 *args, **kwargs):
        """
        Parameters
        ----------
        min_paddle_height : int or None
            Minimum paddle height. If None, zero.
        max_paddle_height : int or None
            Maximum paddle height. If None, a fourth of the game height.
        """
        super(OffsetPaddleBreakout, self).__init__(*args, **kwargs)

        self.min_paddle_height = min_paddle_height
        self.max_paddle_height = max_paddle_height

        if self.min_paddle_height is None:
            if self.bottom_wall_of_punishment:
                self.min_paddle_height = self.wall_thickness
            else:
                self.min_paddle_height = 0

        if self.max_paddle_height is None:
            self.max_paddle_height = self.height // 4

    def layout(self):
        """
        API method. Sets up layers of bricks as usual, and changes the height
        of the paddle randomly.
        """
        paddle_height = random.randrange(self.min_paddle_height,
                                         self.max_paddle_height)
        self.paddle_starting_position = (None, paddle_height)
        super(OffsetPaddleBreakout, self).layout()


class HalfNegativeBreakout(StandardBreakout):
    """
    Color determines score. Divide the screen into two halves s.t. all
    bricks on one half have the same color. One color always gives +1
    reward, while the other color gives -1 reward.
    """
    def __init__(self, *args, **kwargs):
        super(HalfNegativeBreakout, self).__init__(*args, **kwargs)

    def layout(self):
        """
        API method. Sets up layers of bricks, and divides them into right and
        left half-domains with opposite reward signs.
        """
        super(HalfNegativeBreakout, self).layout()

        rewards = [-1, 1]
        brick_colors = random.sample(CLASSIC_BRICK_COLORS, 2)
        random.shuffle(rewards)

        for brick in self.bricks:
            x, _ = brick.position

            if x < self.width // 2:
                brick.color = brick_colors[0]
                brick.reward = rewards[0]
            else:
                brick.color = brick_colors[1]
                brick.reward = rewards[1]


class MiddleWallBreakout(StandardBreakout):
    """
    Left/right/middle part of the game is bordered by an unbreakable wall,
    located under the lowest row of bricks.
    """
    def __init__(self, wall_location='middle', *args, **kwargs):
        """
        Parameters
        ----------
        wall_location : one of ('left', 'middle', 'right', None)
            Location of unbreakable wall. If None, randomize across episodes.
        """
        assert wall_location in ('left', 'middle', 'right', None)
        self.wall_location = wall_location

        if self.wall_location is None:
            self.wall_location = random.choice(('left', 'middle', 'right'))

        super(MiddleWallBreakout, self).__init__(*args, **kwargs)

    def layout(self):
        """
        API method. Sets up layers of bricks and additionaly horizontal wall.
        Note that we put it in the `self.miscellaneous` container and under the
        `layout` method to potentially randomize wall position across episodes.
        """
        super(MiddleWallBreakout, self).layout()

        _, brick_ys = StandardBreakout.brick_wall_coordinates(
            height=self.height,
            width=self.width,
            num_bricks_rows=self.num_bricks_rows,
            brick_shape=self._brick_shape,
            wall_thickness=self.wall_thickness)

        obstacle_shape = (self._brick_shape[0], self.wall_thickness)
        obstacle_y = min(brick_ys) - obstacle_shape[1]

        if self.wall_location == 'left':
            obstacle_xs = range(self.wall_thickness,
                                self.width // 2,
                                obstacle_shape[0])
        elif self.wall_location == 'right':
            obstacle_xs = range(self.width // 2,
                                self.width - self.wall_thickness,
                                obstacle_shape[0])
        elif self.wall_location == 'middle':
            obstacle_xs = range(self.width // 4,
                                self.width - self.width // 4,
                                obstacle_shape[0])

        for x in obstacle_xs:
            self.miscellaneous += [Wall((x, obstacle_y), shape=obstacle_shape)]


class JugglingBreakout(StandardBreakout):
    """
    Small window with just a paddle and one or more balls. Goal is to keep the
    ball(s) from falling off the screen. You may need to constrain this layout
    more so that the game is solvable (i.e., that it may be physically possible
    to not lose a single ball).
    """
    def __init__(self, **kwargs):
        kwargs.setdefault('num_balls', 3)
        kwargs.setdefault('starting_ball_movement_radius', _MAX_SPEED)
        super(JugglingBreakout, self).__init__(**kwargs)

    def layout(self):
        """
        API method. No bricks are present in the game.
        """
        self.bricks = []

    def all_good_bricks_destroyed(self):
        """
        Don't end the game because there are no bricks.
        """
        return False


class RandomTargetBreakout(StandardBreakout):
    """
    A single target appears and respawns each time it is hit.
    """
    def __init__(self,
                 num_bricks=(2, 4),
                 *args, **kwargs):
        """
        Parameters
        ----------
        num_bricks : (int, int)
            Dimensions of the target, specified as the number of bricks that
            compose it along the x- and y-coordinates.
        """
        kwargs.setdefault('starting_ball_movement_radius', _MAX_SPEED)
        super(RandomTargetBreakout, self).__init__(**kwargs)
        self.num_bricks = num_bricks
        self.x_inf = self.wall_thickness
        self.x_sup = (self.width - self.wall_thickness -
                      self.num_bricks[0] * self._brick_shape[0] + 1)

        _, self.brick_ys = StandardBreakout.brick_wall_coordinates(
            height=self.height,
            width=self.width,
            num_bricks_rows=self.num_bricks_rows,
            brick_shape=self._brick_shape,
            wall_thickness=self.wall_thickness)

        self.allowed_ys = self.brick_ys[:-self.num_bricks[1] + 1]

    def layout(self):
        """
        API method. Sets up a single block of bricks.
        """
        self.bricks, self.block_x, self.block_y = self.get_block_of_bricks()

    def get_block_of_bricks(self):
        """
        Helper method. Creates a block of bricks at the desired location.

        Returns
        -------
        block_of_bricks : [ResetterBrick]
        block_x, block_y : (int, int)
            Position of the target.
        """
        m, n = self.num_bricks
        block_x = random.randrange(self.x_inf, self.x_sup)
        block_y = random.choice(self.allowed_ys)
        block_of_bricks = []
        color_index = self.brick_ys.index(block_y)

        for i, j in product(xrange(m), xrange(n)):
            color = CLASSIC_BRICK_COLORS[color_index + j]
            position = (block_x + i * self._brick_shape[0],
                        block_y + j * self._brick_shape[1])
            block_of_bricks += [ResetterBrick(position,
                                              shape=self._brick_shape,
                                              color=color)]

        return block_of_bricks, block_x, block_y

    def position_overlaps(self, ball_position, brick_block_position):
        """
        Help setting up initial brick / target and balls positions without
        overlap.

        Parameters
        ----------
        ball_position : (int, int)
        brick_block_position : (int, int)

        Returns
        -------
        bool
            Whether the ball position overlaps with the target.
        """
        x = ball_position[0]
        y = ball_position[1]
        bx0 = brick_block_position[0]
        by0 = brick_block_position[1]
        dx = self.num_bricks[0] * self._brick_shape[0]
        dy = self.num_bricks[1] * self._brick_shape[1]

        return bx0 <= x < bx0 + dx and by0 <= y < by0 + dy

    def randomize_bricks_positions(self):
        """
        Helper method to respawn the target (block of bricks) elsewhere. Will
        be called by the target's `_collision_effect`.

        Mutates
        -------
        self.bricks : [Brick]
            Mutates the environment bricks to respawn them elsewhere.
        self.block_x, self.block_y : (int, int)
            Keep track of the position of the target.
        """
        bxy = self.balls[0].position
        starting_position = (self.block_x, self.block_y)

        while True:
            if not self.position_overlaps(bxy, (self.block_x, self.block_y)):
                if starting_position != (self.block_x, self.block_y):
                    break

            self.bricks, self.block_x, self.block_y = \
                self.get_block_of_bricks()


class MovingObstaclesBreakout(StandardBreakout):
    """
    Moving obstacles bounce back-and-forth, horizontally, between the paddle
    and the bricks, making it more difficult to aim.
    """
    def __init__(self,
                 obstacles_speeds=(1, -2, 2),
                 obstacles_heights=(10, 18, 26),
                 *args, **kwargs):
        """
        Parameters
        ----------
        obstacles_speeds : [int]
            Horizontal speeds of the obstacles. There are as many obstacles as
            the length of this parameter.
        obstacles_heights : [int]
            Heights of the obstacles. There are as many obstacles as the length
            of this parameter.
        """
        self.obstacles_speeds = obstacles_speeds
        self.obstacles_heights = obstacles_heights

        super(MovingObstaclesBreakout, self).__init__(*args, **kwargs)

    def layout(self):
        """
        API method. Adds bricks and moving obstacles.
        """
        for sgn, y in zip(self.obstacles_speeds, self.obstacles_heights):
            position = (self.width//2 - self._brick_shape[0]//2, y)
            velocity = (self.ball_movement_radius * sgn, 0)

            self.miscellaneous += [
                HorizontallyMovingObstacle(position=position,
                                           velocity=velocity,
                                           shape=self._brick_shape),
            ]

        super(MovingObstaclesBreakout, self).layout()
