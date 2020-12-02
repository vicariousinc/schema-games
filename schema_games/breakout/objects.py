import copy
import numpy as np
import random

from schema_games.utils import (
    compute_edge_nzis,
    compute_shape_from_nzis,
    get_distinct_colors,
    shape_to_nzis,
    offset_nzis_from_position,
)
from schema_games.breakout.constants import (
    _MAX_SPEED,
    _BALL_SHAPE,
    CLASSIC_BALL_COLOR,
    DEFAULT_PADDLE_COLOR,
    CLASSIC_WALL_COLOR,
    MAX_NZIS_PER_ENTITY,
    CLASSIC_BRICK_COLORS,
)


###############################################################################
# API
###############################################################################


class BreakoutObject(object):
    """
    Base class for objects in BreakoutEngine.

    Parameters
    ----------
    position : [int, int] or (int, int) or numpy.ndarray
        Initial position coordinates of the object.
    nzis : [(int, int)] or None
        Nonzero indices for the parts/pixels of the object. (0, 0) corresponds
        to object position and is located at the top left corner of the object.
    shape : (int, int) or None
        Convenience constructor for rectangular objects. Generates the correct
        nzis for a rectangle that shape. When not None, nzis has to be None.
    hitpoints : int
        Number of hitpoints that object has. Note that in order for special
        behavior to happen when that number reaches a special value (e.g.,
        destroy object when hitpoints = 0), that behavior has to be encoded in
        a special _destruction_effect(self, env) method where env is an
        instance of BreakoutEngine.
    is_entity : bool
        If True, the engine reports the object position as an entity.
    color : (int, int, int)
        Color of the object, to be used when rendering the environment and to
        be returned as an observable visual attribute if applicable. Each value
        is an int between 0 and 255 (included).
    visible : bool
        If False, the object is not rendered in the visual output, and not
        considered for physical interaction (e.g., a ball will not bounce off
        of it). This is useful to temporarily remove an object that might
        reappear later.
    entity_id : None or int
        Unique identifier of this entity used when parsing the game image into
        primitive entities (based on shape and color) in the proxy vision
        system. This is useful if we don't want to bother trying to track each
        of the entities by solving a binding problem between frames.
    indirect_collision_effects : bool
        If True, then indirect collisions trigger effects for this object; i.e.
        the special methods presented below may be called during a indirect
        collision. May be set to False in subclasses that yield rewards in
        order to avoid reward entanglement during indirect collisions. Note
        that even if this option is set to False, indirect collisions are still
        taken into account to determine the ball's bounce trajectory.

    Special methods
    ---------------
    '_collision_effect' will be called when a ball collides with this object,
    whether it destroys it or not.

    '_destruction_effect' will be called when a ball collides with this object
    and destroys it.
    """

    unique_entity_id = 0
    unique_object_id = 0
    unique_color_id = 0

    # Map from RGB color to a unique id for that color.
    color_map = {}

    def __init__(
        self,
        position,
        nzis=None,
        shape=None,
        hitpoints=np.PINF,
        is_entity=True,
        color=(128, 128, 128),
        visible=True,
        indirect_collision_effects=True,
    ):

        self._position = np.array(position)
        self.hitpoints = hitpoints
        self.is_entity = is_entity
        self.color = color
        self.visible = visible
        self.is_rectangular = True
        self.indirect_collision_effects = indirect_collision_effects

        assert self.hitpoints > 0
        assert (nzis is None and shape is not None) or (
            nzis is not None and shape is None
        )

        # Set non-zero indices of the object mask's
        if nzis is None:
            self._nzis = shape_to_nzis(shape)
        else:
            self._nzis = np.array(nzis)

        if is_entity:
            self.entity_id = BreakoutObject.unique_entity_id
            BreakoutObject.unique_entity_id += MAX_NZIS_PER_ENTITY
            assert self._nzis.shape[0] <= MAX_NZIS_PER_ENTITY
        else:
            self.entity_id = None

        self.object_id = BreakoutObject.unique_object_id
        BreakoutObject.unique_object_id += 1
        BreakoutObject.register_color(self.color)

        # Sets up slots for memoization
        self.reset_cache()

    def reset_cache(self):
        """
        If the shape changes for any reason, we need to reset cached values.
        Caching these values is useful to reduce overhead as the game engine
        looks up these properties frequently.
        """
        self._cached_shape = None
        self._cached_offset_nzis = None
        self._cached_offset_edge_nzis = None
        self._cached_nzis_min = None
        self._cached_nzis_max = None

    ###########################################################################
    # Protected attributes: setting any of them triggers a cache refresh
    ###########################################################################

    @property
    def position(self):
        return self._position

    @position.setter
    def position(self, pos):
        if all(np.array(pos) == self._position):
            return
        try:
            self._position = np.array(pos)
        except:
            raise
        finally:
            self.reset_cache()

    @property
    def nzis(self):
        return self._nzis

    @nzis.setter
    def nzis(self, new_nzis):
        try:
            self._nzis = new_nzis
        except:
            raise
        finally:
            self.reset_cache()

    ###########################################################################
    # Read-only, cached attributes that derived from `nzis`
    ###########################################################################

    @property
    def offset_edge_nzis(self):
        """
        Return only the non-zero indices on the edge / boundary of the object,
        and offset them by the position of the object within the game frame.
        """
        if self._cached_offset_edge_nzis is None:
            self._cached_offset_edge_nzis = offset_nzis_from_position(
                compute_edge_nzis(self._nzis), self.position
            )
        return self._cached_offset_edge_nzis

    @offset_edge_nzis.setter
    def offset_edge_nzis(self, other):
        raise RuntimeError("Setting `offset_edge_nzis` is not supported.")

    @property
    def offset_nzis(self):
        """
        Return the non-zero indices of the object, offset by the position of
        the object within the game frame.
        """
        if self._cached_offset_nzis is None:
            self._cached_offset_nzis = offset_nzis_from_position(
                self._nzis, self._position
            )
        return self._cached_offset_nzis

    @offset_nzis.setter
    def offset_nzis(self, other):
        raise RuntimeError("Setting `offset_nzis` is not supported.")

    @property
    def shape(self):
        """
        Shape of the object's bounding box.
        """
        if self._cached_shape is None:
            self._cached_shape = compute_shape_from_nzis(self._nzis)
        return self._cached_shape

    @shape.setter
    def shape(self, other):
        raise RuntimeError("Setting `shape` is not supported.")

    @property
    def nzis_min(self):
        """
        Easy optimization if the object is known to be rectangular.
        """
        if self._cached_nzis_min is None:
            self._cached_nzis_min = self._nzis.min(axis=0)
        return self._cached_nzis_min

    @nzis_min.setter
    def nzis_min(self, other):
        raise RuntimeError("Setting `nzis_min` is not supported.")

    @property
    def nzis_max(self):
        """
        Easy optimization if the object is known to be rectangular.
        """
        if self._cached_nzis_max is None:
            self._cached_nzis_max = self._nzis.max(axis=0)
        return self._cached_nzis_max

    @nzis_max.setter
    def nzis_max(self, other):
        raise RuntimeError("Setting `nzis_max` is not supported.")

    ###########################################################################
    # Other properties
    ###########################################################################

    def contains_position(self, pos):
        """
        Does this object contain that position?

        Parameters
        ----------
        pos : (int, int)
            Some position in the game.

        Returns
        -------
        bool
        """
        if self.is_rectangular:
            return self.contains_position_within_bounding_box(pos)
        else:
            return pos in self.offset_nzis

    def contains_position_within_bounding_box(self, pos):
        """
        Check whether some position lies within the object's bounding box.
        We are using this for collision detection, so this is assuming the
        object is square.
        """
        if pos[0] < self.position[0] + self.nzis_min[0]:
            return False
        if pos[0] > self.position[0] + self.nzis_max[0]:
            return False
        if pos[1] < self.position[1] + self.nzis_min[1]:
            return False
        if pos[1] > self.position[1] + self.nzis_max[1]:
            return False
        return True

    def _collision_effect(self, environment):
        """
        Called by the environment when the ball collides with this object
        without destroying it.

        Parameters
        ----------
        environment : BreakoutEngine
            Instance of the game that called that method in the first place. It
            is passed as an argument to allow it to be changed when a collision
            with self occurs.
        """
        pass

    def _destruction_effect(self, environment):
        """
        Called by the environment when the ball collides with this object and
        destroys it.

        Parameters
        ----------
        environment : BreakoutEngine
            Instance of the game that called that method in the first place. It
            is passed as an argument to allow it to be changed when a collision
            with self occurs.
        """
        pass

    @classmethod
    def register_color(cls, color):
        """
        Registers unique colors within the class attribute `BreakoutObject.
        color_map`. This is a legacy attribute that we do not use anywhere, but
        we keep it as it might be useful for debugging.
        """
        if color not in BreakoutObject.color_map:
            BreakoutObject.color_map[color] = BreakoutObject.unique_color_id
            BreakoutObject.unique_color_id += 1


class MoveableObject(BreakoutObject):
    """
    Base class for all objects whose position may change.
    """

    pass


class MomentumObject(MoveableObject):
    """
    Base class for all objects with action-independent momentum.
    """

    pass


###############################################################################
# Different kinds of bricks
###############################################################################


class Brick(BreakoutObject):
    """
    Base class for Breakout bricks. It has a custom attribute 'reward' that is
    only accessed within this class.
    """

    def __init__(self, *args, **kwargs):
        """
        Parameters
        ----------
        reward : int
            Reward upon collision.
        """
        kwargs.setdefault("color", random.choice(get_distinct_colors(6)))
        kwargs.setdefault("hitpoints", 1)
        kwargs.setdefault("indirect_collision_effects", False)
        self.reward = kwargs.pop("reward", 1)

        super(Brick, self).__init__(*args, **kwargs)

    def _collision_effect(self, environment):
        self.hitpoints -= 1

    def _destruction_effect(self, environment):
        environment.reward += self.reward
        environment.brick_hit_counter += 1
        environment.bricks.remove(self)

    @staticmethod
    def brick_colors_classic(num_bricks):
        """
        Helper function.
        """
        if num_bricks < len(CLASSIC_BRICK_COLORS):
            return CLASSIC_BRICK_COLORS[:-2] + [CLASSIC_BRICK_COLORS[-1]]
        else:
            return CLASSIC_BRICK_COLORS


class StrongBrick(Brick):
    """
    Bricks that take multiple hits to be destroyed.
    """

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("hitpoints", 3)
        super(StrongBrick, self).__init__(*args, **kwargs)

        self.init_hitpoints = copy.copy(self.hitpoints)
        self.reward = copy.copy(self.hitpoints)
        self.init_color = copy.copy(self.color)

    def _collision_effect(self, environment):
        """
        Dim brick color
        """
        self.hitpoints -= 1

        # Change color
        coef = float(self.hitpoints) / self.init_hitpoints
        self.color = (
            int(self.init_color[0] * coef),
            int(self.init_color[1] * coef),
            int(self.init_color[2] * coef),
        )

    def _destruction_effect(self, environment):
        environment.reward += self.reward
        environment.brick_hit_counter += 1
        environment.bricks.remove(self)


class PaddleShrinkingBrick(Brick):
    """
    Bricks that shrink the paddle when hit.
    """

    def __init__(self, *args, **kwargs):
        """
        Parameters
        ----------
        shrinkage : int
            Amount by which to shrink the paddle.
        """
        kwargs.setdefault("color", (242, 79, 34))
        self.shrinkage = kwargs.pop("shrinkage", 4)
        super(PaddleShrinkingBrick, self).__init__(*args, **kwargs)

    def _collision_effect(self, environment):
        """
        Shrink paddle
        """
        self.hitpoints -= 1
        environment.paddle.shrink(self.shrinkage, min_length=1)

    def _destruction_effect(self, environment):
        environment.brick_hit_counter += 1
        environment.bricks.remove(self)


class PaddleGrowingBrick(Brick):
    """
    Bricks that grow the paddle when hit.
    """

    def __init__(self, *args, **kwargs):
        """
        Parameters
        ----------
        growth : int
            Amount by which to grow the paddle.
        """
        kwargs.setdefault("color", (34, 197, 242))
        self.growth = kwargs.pop("growth", 4)
        super(PaddleGrowingBrick, self).__init__(*args, **kwargs)

    def _collision_effect(self, environment):
        """
        Grow paddle
        """
        self.hitpoints -= 1
        max_length = environment.width - 2 * environment.wall_thickness
        environment.paddle.grow(self.growth, max_length=max_length)

    def _destruction_effect(self, environment):
        environment.brick_hit_counter += 1
        environment.bricks.remove(self)


class AcceleratorBrick(Brick):
    """
    Bricks that permanently accelerate the ball when hit.
    """

    trigger_counter = _MAX_SPEED - 1

    def __init__(self, *args, **kwargs):
        super(AcceleratorBrick, self).__init__(*args, **kwargs)

    def _collision_effect(self, environment):
        """
        Accelerate ball. To be triggered only a limited number of times.
        """
        self.hitpoints -= 1

        if AcceleratorBrick.trigger_counter > 0:
            new_bmr = environment.ball_movement_radius + 1
            environment.ball_movement_radius = min(new_bmr, _MAX_SPEED)
            AcceleratorBrick.trigger_counter -= 1

    def _destruction_effect(self, environment):
        environment.reward += self.reward
        environment.brick_hit_counter += 1
        environment.bricks.remove(self)


class ResetterBrick(Brick):
    """
    Upon collision, this brick yields some reward and then calls the
    environment's layout function again to reset the game.
    """

    def __init__(self, *args, **kwargs):
        super(ResetterBrick, self).__init__(*args, **kwargs)

    def _collision_effect(self, environment):
        environment.reward += self.reward
        environment.randomize_bricks_positions()

    def _destruction_effect(self, environment):
        pass


###############################################################################
# Other entities: paddle, ball, wall, etc.
###############################################################################


class Paddle(BreakoutObject):
    """
    Paddle. Note that ball-paddle collisions will *not* trigger a call of
    Paddle._collision_effect!
    """

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("color", DEFAULT_PADDLE_COLOR)
        super(Paddle, self).__init__(*args, **kwargs)

    def shrink(self, amount, min_length):
        dx, dy = self.shape
        new_shape = (max(self.shape[0] - amount, min_length), dy)
        self.nzis = shape_to_nzis(new_shape)

    def grow(self, amount, max_length):
        dx, dy = self.shape
        new_shape = (min(self.shape[0] + amount, max_length), dy)
        self.nzis = shape_to_nzis(new_shape)

    def _collision_effect(self, environment):
        raise RuntimeError("Paddle-bound collisions not to be handled here.")

    def _destruction_effect(self, environment):
        raise RuntimeError("Paddle-bound collisions not to be handled here.")


class Ball(MomentumObject):
    """
    Ball. Unlike MomentumObject, it has a special attribute, velocity_index,
    that determines its velocity.
    """

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("color", CLASSIC_BALL_COLOR)

        # Force ball shape since it's an immutable parameter
        assert not {"shape", "nzis"}.intersection(kwargs.keys())
        kwargs["shape"] = _BALL_SHAPE
        kwargs["nzis"] = None
        self.velocity_index = kwargs.pop("velocity_index", None)

        super(Ball, self).__init__(*args, **kwargs)


class Wall(BreakoutObject):
    """
    Wall. It has shape (1, 1) by default.
    """

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("color", CLASSIC_WALL_COLOR)

        if "nzis" not in kwargs and "shape" not in kwargs:
            kwargs["nzis"] = [(0, 0)]

        super(Wall, self).__init__(*args, **kwargs)

        # Things get complicated if walls can be invisible. Protect this.
        assert self.visible


class PaddleShrinkingWall(Wall):
    """
    Wall that reduces the size of the paddle when hit once. All instances share
    a class attribute 'trigger_count' to make sure that the effect be triggered
    only once.
    """

    trigger_count = False

    def __init__(self, *args, **kwargs):
        super(PaddleShrinkingWall, self).__init__(*args, **kwargs)

    def _collision_effect(self, environment):
        """
        Shrink paddle.
        """
        if PaddleShrinkingWall.trigger_count > 0:
            environment.paddle.shrink(environment.paddle.shape[0] // 2)
            PaddleShrinkingWall.trigger_count -= 1


class WallOfPunishment(Wall):
    """
    Virtual wall that is equivalent to losing a ball. Useful when training
    agents. Note that no collision effects are triggered and that rewards are
    handled by the engine normally.
    """

    def __init__(self, *args, **kwargs):
        """
        No need to do anything here, the reward is handled below.
        """
        super(WallOfPunishment, self).__init__(*args, **kwargs)

        # The whole point of this is to have an entity that makes negative
        # rewards given ball loss learnable by an entity-based agent ...
        self.is_entity = True
        self.visible = False


class HorizontallyMovingObstacle(MomentumObject):
    """
    Wall that bounces back and forth. Note that velocity is encoded differently
    than the ball, which is a special case.
    """

    def __init__(self, *args, **kwargs):
        """
        Parameters
        ----------
        velocity : (int, int)
            Initial velocity of the object.
        """
        self.velocity = kwargs.pop("velocity")
        assert self.velocity[1] == 0
        super(HorizontallyMovingObstacle, self).__init__(*args, **kwargs)
