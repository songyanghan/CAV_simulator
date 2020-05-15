#!/usr/bin/env python

# Copyright (c) 2018 Intel Labs.
# authors: German Ros (german.ros@intel.com)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

""" This module implements an agent that roams around a track following random waypoints and avoiding other vehicles.
The agent also responds to traffic lights. """

from agents.discrete_nav.agent import Agent, AgentState
from agents.discrete_nav.local_planner import LocalPlanner
from agents.tools.misc import get_speed
from numpy import random


class RoamingAgent(Agent):
    """
    RoamingAgent implements a basic agent that navigates scenes making random
    choices when facing an intersection.

    This agent respects traffic lights and other vehicles.
    """

    def __init__(self, dt, target_speed, vehicle):
        """

        :param vehicle: actor to apply to local planner logic onto
        """
        super(RoamingAgent, self).__init__(vehicle)
        self._state = AgentState.NAVIGATING
        self._dt = dt
        self._target_speed = target_speed
        self._proximity_threshold = stopping_distance(self._target_speed)#10.0 # DEBUG:
        self._local_planner = LocalPlanner(self._vehicle, {'dt': dt,
                                                           'target_speed': target_speed})

    def run_step(self, debug=False):
        """
        Execute one step of navigation.
        :return: carla.VehicleControl
        """

        # is there an obstacle in front of us?
        hazard_detected = False
        self._proximity_threshold = stopping_distance(get_speed(self._vehicle))

        # retrieve relevant elements for safe navigation, i.e.: traffic lights
        # and other vehicles
        actor_list = self._world.get_actors()
        vehicle_list = actor_list.filter("*vehicle*")
        lights_list = actor_list.filter("*traffic_light*")

        # check possible obstacles
        vehicle_state, vehicle = self._is_vehicle_hazard(vehicle_list)
        if vehicle_state:
            if debug:
                print('!!! VEHICLE BLOCKING AHEAD [{}])'.format(vehicle.id), self._vehicle.get_transform().location.distance(vehicle.get_transform().location))

            self._state = AgentState.BLOCKED_BY_VEHICLE
            hazard_detected = True

        if hazard_detected:
            control = self.emergency_stop()
        else:
            self._state = AgentState.NAVIGATING
            control = self._local_planner.run_step(debug)

        return control

# DEBUG: stopping distance taken from https://korkortonline.se/en/theory/reaction-braking-stopping/
def stopping_distance(speed):
    return speed**2 / 200.0  # meters
