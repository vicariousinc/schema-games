import copy
import numpy as np
import random
import warnings
from itertools import product

import gym
from gym.envs.classic_control import rendering

from schema_games.printing import red, blue, yellow, green, cyan, purple
from schema_games.utils import blockedrange, offset_nzis_from_position
from schema_games.breakout.objects import (
    BreakoutObject,
    Ball,
    Paddle,
    Wall,
    PaddleShrinkingWall,
    WallOfPunishment,
    MoveableObject,
    MomentumObject,
)
from schema_games.breakout.constants import (
    _MAX_SPEED,
    ALLOW_BOUNCE_AGAINST_PHYSICS,
    CLASSIC_BACKGROUND_COLOR,
    BOUNCE_STOCHASTICITY,
    CORRUPT_RENDERED_IMAGE,
    DEBUGGING,
    DEFAULT_HEIGHT,
    DEFAULT_PADDLE_SHAPE,
    DEFAULT_WALL_THICKNESS,
    DEFAULT_WIDTH,
    EXCLUDED_VELOCITIES,
    NUM_BALLS,
    NUM_LIVES,
    PADDLE_SPEED,
    PADDLE_SPEED_DISTRIBUTION,
    PADDLE_STARTING_POSITION,
    REWARD_UPON_BALL_LOSS,
    REWARD_UPON_NO_BRICKS_LEFT,
    STARTING_BALL_MOVEMENT_RADIUS,
)


class ResetHasNeverBeenCalledError(RuntimeError):
    """
    Raised when trying to interact with an uninitialized environment.
    """

    pass


###############################################################################
# Game environment
###############################################################################


class BreakoutEngine(gym.Env):
    """
    Base class for the Breakout game. It is a thin wrapper: in addition to the
    methods required by the Gym API, we require the `layout` method to be
    overridden in subclasses.
    """

    ACTIONS = NOOP, LEFT, RIGHT = range(3)
    action_space = gym.spaces.Discrete(len(ACTIONS))
    metadata = {"render.modes": ["human", "rgb_array"]}
    reward_range = (-1, 1)

    ###########################################################################
    # Game setup
    ###########################################################################

    def __init__(
        self,
        # -- general parameters --
        width=DEFAULT_WIDTH,
        height=DEFAULT_HEIGHT,
        num_lives=NUM_LIVES,
        wall_thickness=DEFAULT_WALL_THICKNESS,
        debugging=DEBUGGING,
        # -- ball parameters --
        num_balls=NUM_BALLS,
        starting_ball_movement_radius=STARTING_BALL_MOVEMENT_RADIUS,
        excluded_velocities=EXCLUDED_VELOCITIES,
        bounce_stochasticity=BOUNCE_STOCHASTICITY,
        allow_bounce_against_physics=ALLOW_BOUNCE_AGAINST_PHYSICS,
        # -- paddle parameters --
        paddle_speed=PADDLE_SPEED,
        paddle_shape=DEFAULT_PADDLE_SHAPE,
        paddle_starting_position=PADDLE_STARTING_POSITION,
        paddle_speed_distribution=PADDLE_SPEED_DISTRIBUTION,
        # -- reward parameters --
        reward_upon_ball_loss=REWARD_UPON_BALL_LOSS,
        reward_upon_no_bricks_left=REWARD_UPON_NO_BRICKS_LEFT,
        # -- state reporting parameters --
        corrupt_rendered_image=CORRUPT_RENDERED_IMAGE,
        report_nzis_as_entities="none",
        report_outer_walls_as_entities=False,
        bottom_wall_of_punishment=True,
        return_state_as_image=False,
    ):
        """
        General Parameters
        ------------------
        width : int
            Width of the world.
        height : int
            Height of the world.
        num_lives : int
            Number of lives before player is defeated.
        wall_thickness : int
            Thickness of environment border walls.
        debugging : bool
            If True, print a bunch of debugging messages, and perform more
            double-checking assertions.

        Ball Parameters
        ---------------
        num_balls : int
            Number of balls in play at the start of the game.
        starting_ball_movement_radius : int
            Starting radius of ball movement. The set of all the new positions
            of the ball corresponds to a circle in the L-infinity norm with a
            radius of self.ball_movement_radius. Since that radius may increase
            during the game, this parameter sets a starting value for it.
        excluded_velocities : {(int, int)}
            Set of ball velocities that will never be considered or proposed.
            This is useful to prevent games where the ball perpetually bounces
            between two parallel walls, but may be set to the empty set in more
            general physics simulations subclassing this class. By default, we
            exclude purely horizontal velocities.
        bounce_stochasticity : int
            Probability that the ball will slightly deviate from the angle or
            speed prescribed by the laws of physics during a bounce.
        allow_bounce_against_physics : bool
            Whether to allow the ball momentum along the vertical axis to have
            a small chance to reverse during a bounce. Game might become
            extremely difficult if True.

        Paddle Parameters
        -----------------
        paddle_speed : int
            Amount of pixels by which the paddle may move at each timestep
        paddle_shape : (int, int)
            Shape of the paddle.
        paddle_starting_position : (int or None, int or None)
            If a coordinate is an integer, use it to set the paddle position
            along the corresponding axis. If it is None, a coordinate will be
            randomly drawn from the domain of valid values for that axis.
        paddle_speed_distribution : [ float ]
            Probability that the paddle will move by -n, ..., -1, 0, 1, ..., n
            pixels, where n is the paddle speed, given that the desired action
            is to move it by +n pixels.

        Reward Parameters
        -----------------
        reward_upon_ball_loss : float
            Reward when a ball bounces past the paddle into oblivion. Generally
            negative (punishment).
        reward_upon_no_bricks_left : float
            Reward upon completion of the game. Generally positive.

        State Reporting Parameters
        --------------------------
        corrupt_rendered_image : bool
            Whether to add noise to the rendered picture of the environment.
        report_nzis_as_entities : str
            If 'all', individual pixels that compose an object are reported as
            separate entities in debug_info. If 'edges', only report nzis that
            compose the edge of the object. If 'none', do not report any other
            nzis than (0, 0) (i.e., object position).
        report_outer_walls_as_entities : bool
            If False, only report the walls bordering the interior of the game
            as entities. Setting to True might slow down the simulation.
        bottom_wall_of_punishment : bool
            If True, then the bottom-most rows of the environment is composed
            of virtual walls. Useful when training an entity-based agent.
        return_state_as_image : bool
            If True, the state returned as every step is the image of the game.
            Otherwise, it's a sparsified form of the image (states dictionary).
        """

        self.width = width
        self.height = height
        self.num_lives = num_lives
        self.num_balls = num_balls
        self.wall_thickness = wall_thickness
        self._ball_movement_radius = starting_ball_movement_radius
        self.paddle_speed = paddle_speed
        self.initial_paddle_shape = paddle_shape
        self.paddle_starting_position = paddle_starting_position
        self.paddle_speed_distribution = paddle_speed_distribution
        self.bounce_stochasticity = bounce_stochasticity
        self.allow_bounce_against_physics = allow_bounce_against_physics
        self.corrupt_rendered_image = corrupt_rendered_image
        self.reward_upon_ball_loss = reward_upon_ball_loss
        self.reward_upon_no_bricks_left = reward_upon_no_bricks_left
        self.report_nzis_as_entities = report_nzis_as_entities
        self.report_outer_walls_as_entities = report_outer_walls_as_entities
        self.bottom_wall_of_punishment = bottom_wall_of_punishment
        self.return_state_as_image = return_state_as_image
        self.debugging = debugging
        self.reset_has_never_been_called = True

        # Gym-specific attributes
        #####################################################################
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.height, self.width, 3)
        )
        self.viewer = None
        #####################################################################

        BreakoutObject.unique_entity_id = 0
        BreakoutObject.unique_object_id = 0

        # The protected mutable attributes (keys below) are reset to their
        # initial values (values below) at each call of `_reset`. This is to
        # ensure we can safely mutate them during a game and revert to their
        # initial values as the beginning of the following game.
        #####################################################################
        self.reset_mutables = {
            "num_lives": copy.copy(num_lives),
            "_ball_movement_radius": copy.copy(starting_ball_movement_radius),
        }

        # Special attributes
        #####################################################################
        self.walls = []
        self.conditional_events = []
        self._memoized_index_to_velocity = {}
        self.excluded_velocities = frozenset(excluded_velocities)

        # Sanity checks
        #####################################################################
        for velocity in self.excluded_velocities:
            assert len(velocity) == 2
            assert 0 <= abs(velocity[0]) <= _MAX_SPEED
            assert 0 <= abs(velocity[1]) <= _MAX_SPEED

        assert self.report_nzis_as_entities in ("all", "edges", "none")
        assert len(self.paddle_speed_distribution) == 2 * self.paddle_speed + 1
        assert np.isclose(np.sum(self.paddle_speed_distribution), 1.0)
        assert 0 <= self.bounce_stochasticity <= 1

        # Physics algorithm works only if these hold true
        error_msg = "Physics algorithm will fail"
        assert self.wall_thickness >= _MAX_SPEED, error_msg
        assert self.initial_paddle_shape[1] >= _MAX_SPEED, error_msg

        # Environment border walls: not really customizable by default
        #######################################################################
        for y in range(self.height):
            for w in range(self.wall_thickness):
                is_entity = (
                    w == self.wall_thickness - 1
                    and y <= self.height - self.wall_thickness
                ) or self.report_outer_walls_as_entities

                self.walls += [
                    Wall((w, y), is_entity=is_entity),
                    Wall((self.width - w - 1, y), is_entity=is_entity),
                ]

        for x in range(self.width):
            for w in range(self.wall_thickness):
                is_entity = (
                    w == self.wall_thickness - 1
                    and x >= self.wall_thickness - 1
                    and x <= self.width - self.wall_thickness
                ) or self.report_outer_walls_as_entities

                self.walls += [
                    PaddleShrinkingWall((x, self.height - w - 1), is_entity=is_entity)
                ]
                if self.bottom_wall_of_punishment:
                    self.walls += [WallOfPunishment((x, w), is_entity=True)]

    ###########################################################################
    # API methods: Gym API methods + our `layout` method
    ###########################################################################

    def render(self, mode="human", close=False):
        return self._render(mode, close)

    def _render(self, mode="human", close=False):
        if self.reset_has_never_been_called:
            raise ResetHasNeverBeenCalledError

        if mode == "rgb_array":
            return self._get_image()

        elif mode == "human":
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()

            self.viewer.imshow(self._get_image())

    def reset(self):
        return self._reset()

    def _reset(self):
        """
        Resets the bricks and ball to start a new game.

        Returns
        -------
        observation : np.array
            The current image of the environment.
        debug_info : dict
            Contains useful information about the environment for debugging
            purposes only.
        """
        for attribute, initial_value in self.reset_mutables.items():
            setattr(self, attribute, initial_value)

        # Set up game objects (balls and paddle: position/velocity do not
        # matter, will be reset below).
        #######################################################################
        px = self.width // 2 - self.initial_paddle_shape[0] // 2
        py = self.wall_thickness if self.bottom_wall_of_punishment else 0
        bx = self.width // 2
        by = self.height // 4

        self.miscellaneous = []
        self.bricks = []
        self.lost_balls = []
        self.balls = [Ball(position=(bx, by)) for _ in range(self.num_balls)]
        print("shape", self.initial_paddle_shape, type(self.initial_paddle_shape))
        self.paddle = Paddle((px, py), shape=self.initial_paddle_shape)

        self.layout()
        self.layout_sanity_check()
        self.randomize_paddle_position()
        self.randomize_ball_position_and_velocity()

        # Build unique mapping of color -> ID at the beginning of the game
        unique_colors = {obj.color for obj in self.objects}
        self.standard_color_map = {c: i for i, c in enumerate(unique_colors)}

        if self.debugging:
            print(
                blue(
                    "Detected the following unique "
                    "colors: {}".format(self.standard_color_map)
                )
            )

        # Counter variables updated at each timestep
        self.current_episode_frame = -1
        self.brick_hit_counter = 0
        self.reward = None  # reward at current timestep

        # Initially observed state
        if self.return_state_as_image:
            state = self._get_image()
        else:
            state = self.get_entity_states()

        self.reset_has_never_been_called = False

        return state

    def step(self, action):
        return self._step(action)

    def _step(self, action):
        """
        Parameters
        ----------
        action : int
            The action to perform.

        Returns
        -------
        observation : np.array
            The current image of the environment.
        reward : float
            The reward received from the current action-state pair.
        done : bool
            Indicates whether the environment has reached a termination state.
        debug_info : dict
            Contains useful information about the environment for debugging
            purposes only. Not used for training or evaluation.
        """
        if self.reset_has_never_been_called:
            raise ResetHasNeverBeenCalledError

        if not self.action_space.contains(action):
            raise ValueError("Invalid action: {}".format(action))

        self.reward = 0

        # Important: reset at every time step!
        self.hit_objects = set()

        # Invalid value to make sure it is set before step() returns!
        self.done = None

        self.debugprint_header()

        # Step 1: Update ball position and check where we landed
        #######################################################################
        for ball in self.balls:
            self._resolve_ball_physics(ball)

        # Step 2: Resolve object collision + destruction
        #######################################################################
        #
        #  If strictly more than one object gets hit in one timestep, multiple
        #  reward sources may thus be entangled, confusing the agent. Prevent
        #  this with a hard check. Of course, if there are more than one ball,
        #  then there will be more than one hit object per timestep in general,
        #  and entanglement is inevitable.
        #
        if len(self.balls) == 1:
            error_msg = red("Multiple collisions detected!")
            indirect_hit_already = False

            for hit in self.hit_objects:
                if not hit.indirect_collision_effects:
                    assert not indirect_hit_already, error_msg
                    indirect_hit_already = True

        for collided_object in self.hit_objects:

            # Call all the collision-triggered effects
            self.debugprint_line("collision")
            collided_object._collision_effect(self)

            # Call all the destruction-triggered effects
            if collided_object.hitpoints == 0:
                self.debugprint_line("destruction")
                collided_object._destruction_effect(self)

        # Step 3: Update moving obstacles positions
        #######################################################################
        for obstacle in self.miscellaneous:
            if isinstance(obstacle, MomentumObject):
                ox, oy = obstacle.position
                vx, vy = obstacle.velocity

                new_nzis_pos = set(
                    offset_nzis_from_position(obstacle.nzis, (ox + vx, oy + vy))
                )
                new_nzis_neg = set(
                    offset_nzis_from_position(obstacle.nzis, (ox - vx, oy - vy))
                )

                occupied_positions = self.occupied_by(exclude=([obstacle] + self.balls))

                # Excluding ball and paddle!
                if not (new_nzis_pos & set(occupied_positions)):
                    obstacle.position = ox + vx, oy + vy
                elif not (new_nzis_neg & set(occupied_positions)):
                    obstacle.position = ox - vx, oy - vy
                    obstacle.velocity = -vx, -vy
                else:
                    pass

        # Step 4: Update paddle position
        #######################################################################
        self.update_paddle_position(action)

        # Step 5: resolve conditional events
        #######################################################################
        for event in self.conditional_events:
            if event.happens(self):
                self.debugprint_line("conditional event", event)
                event.trigger(self)

        # Step 6: Cleanup and return
        #######################################################################
        self.current_episode_frame += 1
        self.end_game_manager()  # should set self.done!
        assert self.done is not None

        debug_info = {
            "entity_states": self.get_entity_states(),
        }

        if self.return_state_as_image:
            state = self._get_image()
        else:
            state = debug_info["entity_states"]

        # Constrain reward to be in {-1, 0, 1}
        self.reward = np.clip(self.reward, -1, 1)
        self.debugprint_line("reward")

        return state, self.reward, self.done, debug_info

    def layout(self):
        """
        Method that sets up the brick, called by BreakoutEngine.reset. Game
        layout routines should not be in reset, because we need to make sure
        that reset calls some caretaker functions after the environment has
        been laid out, which is not certain if reset is overloaded in a
        subclass.
        """
        raise NotImplementedError

    ###########################################################################
    # Core methods and properties
    ###########################################################################

    def layout_sanity_check(self):
        """
        Check that no two objects overlap
        """
        error_msg = "Several objects are overlapping!"
        considered_objects = self.bricks + self.miscellaneous
        all_occupied_nzis = [
            nzi for obj in considered_objects for nzi in obj.offset_nzis if obj.visible
        ]

        assert len(set(all_occupied_nzis)) == len(all_occupied_nzis), error_msg

        # This is an assumption that is kind of True, better be safe
        error_msg = "BreakoutEngine.walls must contain only border walls!"
        for wall in self.walls:
            pos = wall.position
            assert (
                pos[0] < self.wall_thickness
                or pos[0] > self.width - self.wall_thickness - 1
                or pos[1] < self.wall_thickness
                or pos[1] > self.height - self.wall_thickness - 1
            ), error_msg

        # Assuming all the moving objects are paddle, ball, or in miscellaneous
        error_msg = "Moveable objects have to be in miscellaneous only!"
        for obj in self.bricks + self.walls:
            assert not isinstance(obj, MoveableObject)

    @property
    def ball_movement_radius(self):
        return self._ball_movement_radius

    @ball_movement_radius.setter
    def ball_movement_radius(self, value):
        assert 0 < value <= _MAX_SPEED
        self._ball_movement_radius = value

    @property
    def index_to_velocity(self):
        """
        Velocities with adjacent index are considered sort of similar, so we
        can stochastically add or remove 1 to its index in ball velocity-
        randomizing functions such as `randomize_velocity`, so that the ball
        slowly drifts to other directions.

        Returns
        -------
        (int, int)
            Velocity in the (dx, dy) format.
        """
        try:
            return self._memoized_index_to_velocity[self.ball_movement_radius]
        except KeyError:
            unit_square = []
            coordinates = range(
                -self.ball_movement_radius, self.ball_movement_radius + 1
            )

            for dx, dy in product(coordinates, coordinates):
                if max(abs(dx), abs(dy)) != self.ball_movement_radius:
                    continue
                elif (dx, dy) in self.excluded_velocities:
                    continue
                else:
                    unit_square += [(dx, dy)]

            self._memoized_index_to_velocity[self.ball_movement_radius] = dict(
                enumerate(unit_square)
            )
        finally:
            return self._memoized_index_to_velocity[self.ball_movement_radius]

    @property
    def velocity_to_index(self):
        """
        Generate the mapping that converts a velocity vector back to an index
        representing that velocity.

        Returns
        -------
        velocity_to_index : {(int, int): int}
            Mapping from velocity in the (dx, dy) format to a single index.
        """
        velocity_to_index = {}

        for k, v in self.index_to_velocity.items():
            velocity_to_index[v] = k

        return velocity_to_index

    @property
    def objects(self):
        """
        Access all the objects present in the game.

        Returns
        -------
        [BreakoutObject]
            All the objects present in the game.
        """
        return (
            self.walls + self.bricks + self.miscellaneous + self.balls + [self.paddle]
        )

    def occupied_by(self, objects=None, exclude=None):
        """
        Determine the pixel locations (as non-zero indices) occupied by any
        object, subject to additional filters or considering only a subset of
        objects.

        Parameters
        ----------
        objects : [BreakoutObject] or None
            If not None, return locations occupied by this set of objects.
            Otherwise, perform the search over the set of all objects in the
            game.
        exclude : [BreakoutObject] or None
            If not None, exclude these objects from the search.

        Returns
        -------
        {(int, int)}
            Set of occupied pixels in the game.
        """
        objects = self.objects if objects is None else objects
        exclude = [] if exclude is None else exclude

        return {
            nzi
            for obj in objects
            for nzi in obj.offset_nzis
            if obj not in exclude and obj.visible
        }

    @property
    def accessible_domain(self):
        """
        Convenience property that returns the range of valid horizontal
        positions for the paddle, which depends (inter alia) on the size of the
        paddle (which may have changed during play).

        Returns
        -------
        (int, int)
            Range of positions accessible to the paddle, independently of its
            current position.
        """
        return (
            self.wall_thickness,
            self.width - self.wall_thickness - self.paddle.shape[0],
        )

    ###########################################################################
    # Rendering methods
    ###########################################################################

    def render_object(self, image, obj):
        """
        Renders the image of an object in the environment image in-place.

        Parameters
        ----------
        image : numpy.ndarray
            Image to paint in-place. Shaped (self.width, self.height, 3)
        obj : BreakoutObject
        """
        if not obj.visible:
            return
        elif obj.is_rectangular:  # easy optimization
            s_x = np.s_[
                obj.position[0]
                + obj.nzis_min[0] : obj.position[0]
                + obj.nzis_max[0]
                + 1
            ]
            s_y = np.s_[
                obj.position[1]
                + obj.nzis_min[1] : obj.position[1]
                + obj.nzis_max[1]
                + 1
            ]
            image[s_x, s_y, :] = np.array(obj.color).reshape(1, 3)
        else:
            nzis_x, nzis_y = list(zip(*obj.offset_nzis))
            image[nzis_x, nzis_y, :] = np.array(obj.color).reshape(1, 3)

    def _get_image(self):
        """
        Helper method that returns the rendered image of the game.

        Returns
        -------
        image : numpy.ndarray[:, :, :] (dtype=numpy.uint8)
            RGB image in the (r, c, 3) unsigned 8-byte integer format, with
            color values in (0, 255).
        """
        image = self.get_background_image()

        for obj in self.walls + self.bricks + self.miscellaneous:
            if not isinstance(obj, MomentumObject):
                self.render_object(image, obj)

        for obj in self.miscellaneous:
            if isinstance(obj, MomentumObject):
                self.render_object(image, obj)

        # Render the paddle
        if self.paddle.visible:
            self.render_object(image, self.paddle)

        # Render the balls (always last in order to always be visible)
        for ball in self.balls:
            self.render_object(image, ball)

        image = image[:, ::-1, :].transpose(1, 0, 2)

        return image

    def get_background_image(self):
        """
        Gets the background image this environment uses. Useful to keep as a
        separate method.

        Returns
        -------
        image : numpy.ndarray[:, :, :] (dtype=numpy.uint8)
            RGB image in the (r, c, 3) unsigned 8-byte integer format, with
            color values in (0, 255).
        """
        try:
            return self.background_image.copy()
        except AttributeError:
            image = np.zeros((self.width, self.height, 3), dtype=np.uint8)
            image[:] = CLASSIC_BACKGROUND_COLOR
            return image

    ###########################################################################
    # State reporting-method.
    ###########################################################################

    def parse_object_into_pixels(self, breakout_object):
        """
        Parse an object into pixels, each of which becomes a separate entity
        with position, color, and shape (the dimensions of the shape the pixel
        appears to be part of). This is a helper method to do so across
        different objects.

        Note that the states reported by this method are object-agnostic: we do
        not report the state of the `breakout_object` itself, but of each of
        its non-zero indices (nzis, or pixel coordinates) instead, as so many
        separate, atomic "entities".

        This is a quick proxy for a primitive computer vision system that would
        parse the image of the game into different pixel-sized entities, with
        only basic information about shapes, their colors and their positions.

        Parameters
        ----------
        breakout_object : BreakoutObject

        Returns
        -------
        parsed_pixels : [(
            {
                ('position', (int, int)): 0.0,
                ('shape', (int, int)): 0.0,
                ('color', (int, int, int)): 0.0,
            }, int)]

            List of entity states parsed from a single object. Each entity
            state is reported as a (state, unique entity ID) pair. The latter
            is included for tracking correctly each pixel, and the former is a
            dictionary of attributes and their log-probability. This format is
            used to allow replacing this function which a more evolved
            probabilistic vision system; for now, we use a simple deterministic
            proxy.
        """
        parsed_pixels = []

        # Filter valid NZIs
        if self.report_nzis_as_entities == "all":
            reported_nzis = breakout_object.offset_nzis
        elif self.report_nzis_as_entities == "edges":
            reported_nzis = breakout_object.offset_edge_nzis
        elif self.report_nzis_as_entities == "none":
            dr, dc = breakout_object.shape
            r, c = breakout_object.position
            reported_nzis = [(r + dr // 2, c + dc // 2)]
        else:
            raise ValueError("Invalid parameter: %s" % self.report_nzis_as_entities)

        eid = breakout_object.entity_id

        # Report each valid NZI as a separate entity
        for dx, dy in reported_nzis:
            r, c = self.xy2rc((dx, dy))
            du = dx - breakout_object.position[0]
            dv = dy - breakout_object.position[1]
            color = breakout_object.color

            if self.debugging:
                assert 0 <= du < breakout_object.shape[0]
                assert 0 <= dv < breakout_object.shape[1]

            state = {
                ("position", (r, c)): 0.0,
                ("shape", (du, dv)): 0.0,
                ("color", color): 0.0,
            }

            parsed_pixels.append((state, eid))
            eid += 1  # for each nzi in the object

        return parsed_pixels

    def get_entity_states(self):
        """
        Entity states that we may resonably expect a computer vision system to
        parse. We report each contiguous shape visible in the scene, with the
        relative position of each of its component pixels as well as their
        colors. See also the docstring of `parse_object_into_pixels` for more
        details.

        Returns
        -------
        entity_states : {int: {
                ('position', (int, int)): 0.0,
                ('shape', (int, int)): 0.0,
                ('color', (int, int, int)): 0.0,
            }}

            Entity states (see docstring of `parse_object_into_pixels`) for
            more details.
        """
        entity_states = {}

        # Ball ################################################################
        for ball in self.balls:
            if ball.is_entity:
                for state, eid in self.parse_object_into_pixels(ball):
                    entity_states[eid] = state

        # Paddle ##############################################################
        if self.paddle.is_entity:
            for state, eid in self.parse_object_into_pixels(self.paddle):
                entity_states[eid] = state

        # Miscellaneous objects ###############################################
        for obj in self.miscellaneous:
            if obj.is_entity:
                for state, eid in self.parse_object_into_pixels(obj):
                    entity_states[eid] = state

        # Walls and bricks ####################################################
        for wall in self.walls:
            if wall.is_entity:
                for state, eid in self.parse_object_into_pixels(wall):
                    entity_states[eid] = state

        for brick in self.bricks:
            if brick.is_entity:
                for state, eid in self.parse_object_into_pixels(brick):
                    entity_states[eid] = state

        return entity_states

    ###########################################################################
    # Game dynamics
    ###########################################################################

    def _resolve_ball_physics(self, ball):
        """
        Physics resolution for a specific Ball object. To make things simpler,
        Balls are assumed to have zero surface, e.g., two Balls can be in the
        same exact position. Also, we assume no ordering between collisions,
        i.e., two incoming Balls will bounce against and destroy a Brick at the
        same exact time. Also, we assume all balls share the same
        ball_velocity_radius.

        This function should only update ball positions and register collisions
        and destructions, and update the list of objects hit by balls over the
        past time step.

        Corner case: balls have priority over the paddle under the current
        implementation, i.e., the paddle position is updated after the balls'.
        This begets the following corner case: if the ball occupied an empty
        position and is about to bounce on the paddle, and if the paddle moves
        towards the ball, the current collision resolution will "trap" the ball
        inside the paddle, in an endless loop, until the paddle moves again. A
        way to preclude this from happening is to force the ball to exit the
        paddle if the former is inside the latter.

        Parameters
        ----------
        ball : Ball

        Mutates
        -------
        ball : Ball
            This function changes the two attributes `ball.velocity_index` and
            `ball.position`.
        """
        assert isinstance(ball, Ball)

        bx, by = ball.position
        vx, vy = self.index_to_velocity[ball.velocity_index]
        orig_vx, orig_vy = vx, vy

        # Address the corner case mentioned in the docstring: if the ball
        # starts in the paddle, ignore everything and make it travel in a
        # straight line.
        if self.paddle.contains_position((bx, by)):
            self.debugprint_line("ball inside paddle")
            ball.position = vx + bx, vy + by
            return

        # Step A: Update ball position.
        #######################################################################
        vx_after_paddle_bounce = self.get_ball_vx_after_paddle_bounce(bx, by, vx, vy)

        intangible = [
            obj for obj in self.walls if isinstance(obj, WallOfPunishment)
        ] + self.balls
        occupied_positions = self.occupied_by(exclude=intangible)

        self.hit_objects |= self.get_collision_elements((bx + vx, by + vy))

        # [Emptiness]
        if (
            bx + vx,
            by + vy,
        ) not in occupied_positions and vx_after_paddle_bounce is None:

            self.debugprint_line("ball physics", 0, vx_after_paddle_bounce)

            # Easiest case! Destination is empty. Notice that this assumes
            # there are no walls that are 1 pixel thick. If that was the case,
            # this suddenly becomes a nightmare.
            ball.position = vx + bx, vy + by
            # Velocity remains the same.

        # [Bounce, paddle]
        elif vx_after_paddle_bounce is not None:
            self.debugprint_line("ball physics", 1, vx_after_paddle_bounce)

            vx = vx_after_paddle_bounce
            vy *= -1

            # Don't let the ball slow down by a paddle bounce!
            if abs(vy) < self.ball_movement_radius and abs(vx) <= 1:
                vy = self.ball_movement_radius

            ball.velocity_index = self.velocity_to_index[(vx, vy)]
            ball.position = vx + bx, vy + by

            if (vx + bx, vy + by) in occupied_positions:
                vx *= -1
                ball.velocity_index = self.velocity_to_index[(vx, vy)]
                ball.position = vx + bx, vy + by

        # [Bounce, brick or wall]
        elif (bx + vx, by) in occupied_positions:
            self.debugprint_line("ball physics", 2, vx_after_paddle_bounce)

            vx *= -1
            ball.velocity_index = self.velocity_to_index[(vx, vy)]
            ball.velocity_index = self.randomize_velocity(ball.velocity_index)
            vx, vy = self.index_to_velocity[ball.velocity_index]
            ball.position = vx + bx, vy + by

        # [Bounce, brick or wall]
        elif (bx, by + vy) in occupied_positions:
            self.debugprint_line("ball physics", 3, vx_after_paddle_bounce)

            vy *= -1
            ball.velocity_index = self.velocity_to_index[(vx, vy)]
            ball.velocity_index = self.randomize_velocity(ball.velocity_index)
            vx, vy = self.index_to_velocity[ball.velocity_index]
            ball.position = vx + bx, vy + by

        else:
            self.debugprint_line("ball physics", 4, vx_after_paddle_bounce)

            vx, vy = -vx, -vy
            ball.velocity_index = self.velocity_to_index[(vx, vy)]
            ball.position = bx + vx, by + vy

        # Step B: Check where we landed, manage any higher-order collisions
        #######################################################################
        vx, vy = self.index_to_velocity[ball.velocity_index]
        if tuple(ball.position) in occupied_positions:
            self.debugprint_line("higher-order collision")

            # -----------------------------------------------------------------
            # Here we flag this as an indirect collision effect as some objects
            # (like bricks) only trigger effects, e.g., yield rewards, after a
            # direct collision. This exception is in order to avoid multiple
            # brick destructions in one timestep, which would mess with
            # learning when the destroyed bricks yield opposite sign rewards.
            # -----------------------------------------------------------------
            self.hit_objects |= self.get_collision_elements(
                ball.position, is_indirect=True
            )

            vx, vy = -orig_vx, -orig_vy
            ball.position = bx + vx, by + vy
            ball.velocity_index = self.velocity_to_index[(vx, vy)]

    def end_game_manager(self):
        """
        Helper function which determines whether the game is done.

        Important: this should set BreakoutEngine.done directly and return
        nothing. This is so that 'done' is accessible by any function that has
        access to the instance of BreakoutEngine and may be modified
        conditioned on some event happening.

        Moreover, we should only attempt to set a default value for 'done' when
        it is None, to avoid overriding its value if it has been set by a third
        -party event.

        Mutates
        -------
        self.done : bool
            Is the game over?
        self.reward : int or float
            Reward at this time step.
        self.num_lives : int
            Number of remaining lives.
        """

        # Are there any bricks remaining?
        if self.all_good_bricks_destroyed():
            self.reward += self.reward_upon_no_bricks_left
            self.done = True
            print(green("Game over! You won."))

        # Catch lost balls
        for ball in self.balls:
            if ball.position[1] <= 0:
                lost_ball = self.balls.pop(self.balls.index(ball))
                self.lost_balls.append(lost_ball)
                self.reward += self.reward_upon_ball_loss

        # No balls left! Reset.
        if len(self.balls) == 0:
            self.num_lives -= 1
            self.balls = self.lost_balls
            self.lost_balls = []
            self.randomize_ball_position_and_velocity()

            if self.num_lives > 0:
                self.done = False
                print(red("[---] Lives remaining:"), self.num_lives)
            else:
                self.done = True
                print(red("[---] Game over! You lost."))
                print(red("*" * 80))

        if self.done is None:
            self.done = False

    def update_paddle_position(self, action):
        """
        Change the paddle position while checking for movement validity.

        Parameters
        ----------
        action : int
            Action taken by the agent.

        Mutates
        -------
        self.paddle.position : numpy.ndarray([int, int])
            Position of the paddle.
        """
        speeds = np.arange(-self.paddle_speed, self.paddle_speed + 1)
        dx = np.random.choice(speeds, p=self.paddle_speed_distribution)
        dx, dy = {
            self.LEFT: np.array([-dx, 0]),
            self.RIGHT: np.array([+dx, 0]),
            self.NOOP: np.array([0, 0]),
        }[action]

        # If there is not enough space for a full translation,
        # move the paddle anyway until it bumps against a wall.
        px = np.clip(
            self.paddle.position[0] + dx,
            self.accessible_domain[0],
            self.accessible_domain[1],
        )
        py = self.paddle.position[1] + dy

        self.paddle.position = np.array([px, py])

    def get_collision_elements(self, ball_position, is_indirect=False):
        """
        Retrieve bricks that were hit by the ball during the time step.

        Parameters
        ----------
        ball_position : (int, int)
            Ball position.
        is_indirect : bool
            Whether the collision was indirect (a higher-order collision that
            we detect in a roundabout way because it happens too faster than
            the temporal resolution of the game allowd).

        Returns
        -------
        set_hit_objects : {BreakoutObject}
            Objects that were hit by the ball given its position.
        """
        set_hit_objects = set()

        for brick in self.bricks:
            if brick.visible and brick.contains_position(ball_position):
                if not is_indirect or brick.indirect_collision_effects:
                    set_hit_objects |= {brick}

        for wall in self.walls:
            if wall.visible and wall.contains_position(ball_position):
                if not is_indirect or wall.indirect_collision_effects:
                    set_hit_objects |= {wall}

        for misc in self.miscellaneous:
            if misc.visible and misc.contains_position(ball_position):
                if not is_indirect or misc.indirect_collision_effects:
                    set_hit_objects |= {misc}

        return set_hit_objects

    def randomize_velocity(self, old_index):
        """
        Randomize a velocity in single-index form, and optionally making sure
        that the updated velocity vector stays in the same quadrant.

        Parameters
        ----------
        old_index : int
            Old velocity index. See also `index_to_velocity`.

        Returns
        -------
        new_index : int
            Updated velocity index. See also `index_to_velocity`.
        """
        if not bool(np.random.binomial(1, self.bounce_stochasticity)):
            return old_index
        else:
            while True:
                new_index = random.choice(list(self.index_to_velocity.keys()))

                # If True, make sure that new velocity stays in same quadrant
                if not self.allow_bounce_against_physics:
                    old_velocity = self.index_to_velocity[old_index]
                    new_velocity = self.index_to_velocity[new_index]

                    if (np.sign(old_velocity) == np.sign(new_velocity)).all():
                        break
                else:
                    break

            return new_index

    def get_paddle_response_function(self):
        """
        Get the function that describes the horizontal (x) component of the
        ball velocity after it bounces off the paddle, as a function of the
        abscissa of its impact on the paddle.

        Returns
        -------
        prf : numpy.ndarray([int])
            Horizontal components of the ball velocity, indexed by coordinate
            along paddle length.
        """
        size = self.paddle.shape[0]
        prf_center = [0] * (1 + (size + 1) % 2)
        half = (size - len(prf_center)) // 2

        if size < 2 * self.ball_movement_radius + 1:
            warnings.warn(red("Paddle too small for PRF to be correct!"))
            if size == 1:
                raise RuntimeError("Invalid paddle length")
            elif size == 2:
                return np.array([-self.ball_movement_radius, self.ball_movement_radius])

        prf_left = blockedrange(half, self.ball_movement_radius)
        prf_left = [[k + 1] * len(sub) for k, sub in enumerate(prf_left)]
        prf_left = [item for sub in prf_left for item in sub]
        prf_left = np.array(prf_left).ravel()
        prf = np.hstack((-prf_left[::-1], prf_center, prf_left))

        assert len(prf) == size
        return prf

    def get_ball_vx_after_paddle_bounce(self, x, y, u, v):
        """
        Given an integer coordinate (x, y), if the ball hit the paddle, return
        the ball x speed, otherwise return None. The bounce angle is a funtion
        of where the ball lands on the paddle relatively to its center:

            [-r ... -1, ... -1, 0, 1, ... 1 2 ... 2 ... r-1 ... r-1 r ... r]

        The paddle is equally divided into zones with bounce angles increasing
        away from the paddle center. If the ball lands at the center, then it
        bounces purely vertically.

        Note: we need to distinguish two cases if the ball bounces at time t+1:

            #1: the ball is already above the paddle before bouncing (i.e. ball
            x[t] is already within the range of paddle x[t]'s). This means that
            x[t], y[t] + v[t] will be considered the point of impact, and *NOT*
            x[t] + u[t], y[t] + v[t]. Thus u[t+1] is given by the paddle
            response function (PRF) at coordinate x[t], *NOT* x[t] + u[t].

            #2: the ball is not above the paddle before bouncing, but collides
            with the paddle at time t+1. Then, the closest paddle x[t] will be
            considered as the point of impact. Thus Thus u[t+1] is given by the
            paddle response function (PRF) at coordinate min(x[t]) or max(x[t])
            for paddle pixels {x[t]}, depending on which is closer to the ball
            x[t].

        Parameters
        ----------
        x : int
            Abscissa of ball before bounce
        y : int
            Ordinate of ball before bounce
        u : int
            Incoming (i.e., pre-bounce) ball velocity along x-axis
        v : int
            Incoming (i.e., pre-bounce) ball velocity along y-axis

        Returns
        -------
        int or None.
            Speed we expect the ball to bounce at off the paddle. If the ball
            is not touching a paddle, return None.
        """

        # Corner case, some test environments do not have a paddle
        if not self.paddle.visible:
            return None

        # The ball will not collide with the paddle
        if (x + u, y + v) not in self.paddle.offset_nzis:
            return None

        # Get paddle characteristics and determine where the ball will bounce
        prf = self.get_paddle_response_function()
        paddle_x_min = min(px for px, py in self.paddle.offset_nzis)
        paddle_x_max = max(px for px, py in self.paddle.offset_nzis)

        if paddle_x_min <= x <= paddle_x_max:  # Case #1 (see docstring)
            ball_impact_x = x
        elif x > paddle_x_max:  # Case #2A (see docstring)
            ball_impact_x = paddle_x_max
        elif x < paddle_x_min:  # Case #2B (see docstring)
            ball_impact_x = paddle_x_min

        # Get integral coordinate of ball impact relatively to paddle left tip
        ball_impact_x_relative = float(ball_impact_x - self.paddle.position[0])

        return prf[int(ball_impact_x_relative)]

    def randomize_paddle_position(self):
        """
        Useful scrambling function to be called at beginning of new episode.
        If values have been provided for the paddle's initial position, use
        them. Otherwise, randomly draw them from the valid domain.

        Note: None for the y-component of the paddle_starting_position defaults
        to 0 offset, not random (as in the x-component).

        Mutates
        -------
        self.paddle.position : (int, int)
            Updated paddle position.
        """
        if self.paddle_starting_position[0] is not None:
            if not (
                self.accessible_domain[0]
                <= self.paddle_starting_position[0]
                <= self.accessible_domain[1]
            ):
                raise ValueError("Bad x-position for the paddle.")
            px = self.paddle_starting_position[0]
        else:
            px = random.randrange(
                self.accessible_domain[0], self.accessible_domain[1] + 1
            )

        if self.paddle_starting_position[1] is not None:
            py = self.paddle_starting_position[1]
        elif self.bottom_wall_of_punishment:
            py = self.wall_thickness
        else:
            py = 0

        self.paddle.position = (px, py)

    def randomize_ball_position_and_velocity(self):
        """
        Useful scrambling function to be called at beginning of new episode.
        Randomize ball position and velocity, subject to the requirement that
        the ball must move downwards toward the paddle.

        Mutates
        -------
        Ball.position : [(int, int)]
            Updated positions for each ball.
        """
        downward_velocities = {
            k: v for k, v in self.index_to_velocity.items() if v[1] < 0
        }

        for ball in self.balls:
            ball.velocity_index = random.choice(list(downward_velocities.keys()))

        brick_ordinates = [self.height - 1 - self.wall_thickness]
        brick_ordinates += [
            brick.position[1] + brick.nzis_min[1] for brick in self.bricks
        ]
        maximum_ball_y = min(brick_ordinates)

        for ball in self.balls:
            while True:
                ###############################################################
                # Below is the correct way to randomize ball ordinate, to avoid
                # overfitting on the ball/paddle gap modulo _MAX_SPEED.
                ###############################################################
                ball_offsets = range(-self.num_balls - 1, self.num_balls + 2)
                ball.position = (
                    self.width // 2 + np.random.choice(ball_offsets),
                    maximum_ball_y // 2 - random.randrange(_MAX_SPEED),
                )
                ###############################################################
                occupied_positions = self.occupied_by(exclude={ball})

                if tuple(ball.position) not in occupied_positions:
                    if self.debugging:
                        print(
                            purple("Ball-paddle separation at collision:"),
                            yellow("%i pixels" % (ball.position[1] % 2)),
                            purple("vertically when |v[y]| = %i" % _MAX_SPEED),
                        )
                    break

    ###########################################################################
    # Helper methods
    ###########################################################################

    def debugprint_header(self):
        """
        Prints the header of debugging messages at each new step.
        """
        if self.debugging:
            string = cyan("[%03i] " % self.current_episode_frame)
            string += blue("Ball positions and ")
            string += cyan("velocities ")
            string += blue(": ")

            for ball in self.balls:
                velocity_ = self.index_to_velocity[ball.velocity_index]
                string += blue("({}, {}, ".format(*ball.position))
                string += cyan("{}, {}".format(*velocity_))
                string += blue("),")

            print(string)

    def debugprint_line(self, event_type, *args):
        """
        Prints an indented line that marks a particular event.
        """
        N_SPACES = 5

        if self.debugging:
            if event_type == "collision":
                print(" " * N_SPACES, yellow("<!>"))
            elif event_type == "destruction":
                print(" " * N_SPACES, red("<!>"))
            elif event_type == "conditional event":
                event_name = type(args[0]).__name__
                print(" " * N_SPACES, red("<E> : {}".format(event_name)))
            elif event_type == "reward":
                print(" " * N_SPACES, green("-> REWARD:"), purple(self.reward))
            elif event_type == "ball physics":
                print(
                    " " * N_SPACES,
                    green("-> {}".format(args[0])),
                    yellow("vx<PRF>: {}".format(args[1])),
                )
            elif event_type == "higher-order collision":
                print(" " * N_SPACES, cyan("<>"))
            elif event_type == "ball inside paddle":
                print(" " * N_SPACES, purple("<>"))

    def all_good_bricks_destroyed(self):
        """
        Helper function which returns True if all bricks yielding strictly
        positive reward have been destroyed, otherwise False.
        """
        for brick in self.bricks:
            if brick.reward > 0:
                return False
        return True

    def xy2rc(self, position):
        """
        Converts this environment's native right-handed coordinates (x, y) into
        entity states' (row, column) coordinates.

        Parameters:
        -----------
        position : (int, int)
            Position as (x, y) indices.

        Returns:
        --------
        position : (int, int)
            Position as (row, col) indices.
        """
        return (self.height - 1 - position[1], position[0])
