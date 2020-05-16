#!/usr/bin/env python

# Copyright (c) 2018 Intel Labs.
# authors: German Ros (german.ros@intel.com)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

""" This module implements an agent that roams around a track following random
waypoints and avoiding other vehicles.
The agent also responds to traffic lights. """

from enum import Enum

from agents.tools.misc import get_speed, scalar_proj, norm
from agents.discrete_nav.local_planner import RoadOption

import carla


Tds = 10 #Number of timesteps to check behavior planner
F = 50 # number of past timesteps to remember for lane changing
w = 0.4 # weight of Qv in lane change reward function
theta_left = 2.0 # threshold to switch left
theta_right = 2.0 # threshold to switch right
eps = 150 # meters # TODO: Google DSRC?
theta_a = 2.0 # meters/sec^2 # threshold above which acceleration is "uncomfortable"


class AgentState(Enum):
    """
    AGENT_STATE represents the possible states of a roaming agent
    """
    NAVIGATING = 1
    BLOCKED_BY_VEHICLE = 2


class Agent(object):
    """
    Base class to define agents in CARLA
    """

    def __init__(self, vehicle, dt):
        """
        :param vehicle: actor to apply to local planner logic onto
        """
        self.t = 0
        self.dt = dt
        self.vehicle = vehicle
        self.proximity_threshold = 10.0  # meters
        self.local_planner = None
        self.world = self.vehicle.get_world()
        self.map = self.vehicle.get_world().get_map()
        self.state = AgentState.NAVIGATING
        self.current_waypoint = None

        self.theta_a = 2.0  # m/s^2

    def discrete_state(self):
        return self.local_planner._target_road_option

    def get_measurements(self):
        velocity_fwd = scalar_proj(self.vehicle.get_velocity(),
                                   self.current_waypoint.transform.get_forward_vector())

        acceleration = norm(self.vehicle.get_acceleration())
        is_changing_lanes = (self.local_planner._target_road_option in {RoadOption.CHANGELANELEFT,
                                                                        RoadOption.CHANGELANERIGHT})

        if is_changing_lanes:
            comfort_cost = 3
        elif acceleration >= theta_a:
            comfort_cost = 2
        else:
            comfort_cost = 1

        return self.t, velocity_fwd, comfort_cost

    def run_step(self, debug=False):
        """
        Execute one step of navigation.
        :return: control
        """
        self.t += self.dt
        self.current_waypoint = self.map.get_waypoint(self.vehicle.get_location())

        control = carla.VehicleControl()

        if debug:
            control.steer = 0.0
            control.throttle = 0.0
            control.brake = 0.0
            control.hand_brake = False
            control.manual_gear_shift = False

        return control

    def emergency_stop(self):
        """
        Send an emergency stop command to the vehicle
        :return: control
        """
        control = carla.VehicleControl()
        control.steer = 0.0
        control.throttle = 0.0
        control.brake = 1.0
        control.hand_brake = False

        return control
