"""
Conditional events. At every step, the game engine checks which of all
registered events happen and triggers the requisite consequences if applicable.
"""
import random
from abc import abstractmethod, ABCMeta

from schema_games.breakout.constants import _MAX_SPEED


###############################################################################
# API
###############################################################################

class ConditionalEvent(object):
    """
    Base class for conditional events.
    """
    __metaclass__ = ABCMeta

    def __init__(self):
        pass

    @abstractmethod
    def happens(self, environment):
        """
        Evaluates whether some event has happened in the environment. The event
        is defined by subclassing ConditionalEvent and overloading this method.

        Parameters
        ----------
        environment : BreakoutEngine
            Game instance to use to determine whether some even has happed.

        Returns
        -------
        happened : bool
            Whether the event has happened
        """
        pass

    @abstractmethod
    def trigger(self, environment):
        """
        Effect to be triggered if self.happens returns True. Note that it is
        the environment's job to call self.trigger, and *not* this instance of
        ConditionalEvent!

        Parameters
        ----------
        environment : BreakoutEngine
            Game instance to use to determine whether some even has happed.
        """
        pass


###############################################################################
# Custom events
###############################################################################

class BallAcceleratesEvent(ConditionalEvent):
    """
    Ball accelerates at specific ball counts.

    Parameters
    ----------
    brick_hits : int
        Number of bricks to destroy to trigger the event.
    """
    def __init__(self, brick_hits):
        self.brick_hits = brick_hits
        self.trigger_count = 1

    def happens(self, environment):
        return (environment.brick_hit_counter == self.brick_hits and
                self.trigger_count > 0)

    def trigger(self, environment):
        new_bmr = min(environment.ball_movement_radius + 1, _MAX_SPEED)
        environment.ball_movement_radius = new_bmr
        self.trigger_count -= 1


class PaddleShrinksEvent(ConditionalEvent):
    """
    Paddle stochastically shrinks by 'decrement' once every 'cycle_length',
    with some stochasticity added for good measure.

    Parameters
    ----------
    decrement : int
        Amount by which to decrease paddle length.
    cycle_length : int
        Paddle shrinks at least once every that many timesteps.
    min_paddle_length : int
        Paddle length at which the event will stop shrinking it further.
    """
    def __init__(self, decrement, cycle_length, min_paddle_length):
        self.decrement = decrement
        self.cycle_length = cycle_length
        self.min_paddle_length = min_paddle_length

        # Counter variables
        self.cycle_counter = 0
        self.index_in_cycle = random.randrange(self.cycle_length)

    def happens(self, environment):
        """
        Environment time frames are grouped into cycles of length cycle_length.
        Within each cycle, one frame is randomly chosen to trigger this event.
        """
        new_cycle = (environment.current_episode_frame //
                     self.cycle_length > self.cycle_counter)
        if new_cycle:
            self.index_in_cycle = random.randrange(self.cycle_length)
            self.cycle_counter = (environment.current_episode_frame //
                                  self.cycle_length)

        return (environment.current_episode_frame %
                self.cycle_length == self.index_in_cycle)

    def trigger(self, environment):
        environment.paddle.shrink(self.decrement,
                                  min_length=self.min_paddle_length)
