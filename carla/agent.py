#!/usr/bin/env python

# Companion code for the UConn undergraduate Honors Thesis "Evaluating Driving
# Performance of a Novel Behavior Planning Model on Connected Autonomous
# Vehicles" by Keyur Shah (UConn '20). Thesis was advised by Dr. Fei Miao;
# see http://feimiao.org/research.html.
#
# This code is meant for use with the autonomous vehicle simulator CARLA
# (https://carla.org/).
#
# Disclaimer: The CARLA project, which this project uses code from, follows the
# MIT license. The license is available at https://opensource.org/licenses/MIT.

from enum import Enum

from tools.misc import get_speed, scalar_proj, norm
from local_planner import RoadOption

import carla


class Agent(object):
    """
    Base class to define agents in CARLA
    """

    def __init__(self, vehicle, dt, param_dict):
        """
        :param vehicle: actor to apply to local planner logic onto
        """
        self.t = 0
        self.dt = dt
        self.vehicle = vehicle
        self.local_planner = None
        self.world = self.vehicle.get_world()
        self.map = self.vehicle.get_world().get_map()
        self.current_waypoint = None

        self.Tds = param_dict['Tds']
        self.theta_a = param_dict['theta_a']
        # Safety parameters
        self.theta_l = param_dict['theta_l']
        self.theta_c = param_dict['theta_c']
        self.theta_r = param_dict['theta_r']
        self.change_distance = param_dict['chg_distance']

    def discrete_state(self):
        # Local planner is not allowed to take left/right turn roadoptions,
        # so target road option will necessarily be one of LANEFOLLOW (LK),
        # CHANGELANELEFT (CL), or CHANGELANERIGHT (CR) and thus we use the
        # target road option variable as our state
        return self.local_planner._target_road_option

    def get_measurements(self):
        velocity_fwd = scalar_proj(self.vehicle.get_velocity(),
                                   self.current_waypoint.transform.get_forward_vector())

        acceleration = norm(self.vehicle.get_acceleration())
        is_changing_lanes = (self.local_planner._target_road_option in {RoadOption.CHANGELANELEFT,
                                                                        RoadOption.CHANGELANERIGHT})

        if is_changing_lanes:
            comfort_cost = 3
        elif acceleration >= self.theta_a:
            comfort_cost = 2
        else:
            comfort_cost = 1

        return velocity_fwd, comfort_cost

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
