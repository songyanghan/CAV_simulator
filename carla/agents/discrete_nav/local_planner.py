#!/usr/bin/env python

# Copyright (c) 2018 Intel Labs.
# authors: German Ros (german.ros@intel.com)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

""" This module contains a local planner to perform low-level waypoint following based on PID controllers. """

from enum import Enum
from queue import Queue
from collections import deque
import random

import carla
from agents.discrete_nav.controller import VehiclePIDController
from agents.tools.misc import distance_vehicle, draw_waypoints


Tds = 10 #Number of timesteps to check discrete state switching
T = 1 #Number of timesteps to share future data for - NOT USED YET
F = 50 # number of past timesteps to remember for lane changing
w = 0.4 # weight of Qv in lane change reward function
theta_left = 2.0 # threshold to switch left
theta_right = 2.0 # threshold to switch right
eps = 35 # meters

class RoadOption(Enum):
    """
    RoadOption represents the possible topological configurations when moving from a segment of lane to other.
    """
    VOID = -1
    LEFT = 1
    RIGHT = 2
    STRAIGHT = 3
    LANEFOLLOW = 4
    CHANGELANELEFT = 5
    CHANGELANERIGHT = 6


class LocalPlanner(object):
    """
    LocalPlanner implements the basic behavior of following a trajectory of waypoints that is generated on-the-fly.
    The low-level motion of the vehicle is computed by using two PID controllers, one is used for the lateral control
    and the other for the longitudinal control (cruise speed).

    When multiple paths are available (intersections) this local planner makes a random choice.
    """

    # minimum distance to target waypoint as a percentage (e.g. within 90% of
    # total distance)
    MIN_DISTANCE_PERCENTAGE = 0.9

    def __init__(self, vehicle, opt_dict=None):
        """
        :param vehicle: actor to apply to local planner logic onto
        :param opt_dict: dictionary of arguments with the following semantics:
            dt -- time difference between physics control in seconds. This is typically fixed from server side
                  using the arguments -benchmark -fps=F . In this case dt = 1/F

            target_speed -- desired cruise speed in Km/h

            sampling_radius -- search radius for next waypoints in seconds: e.g. 0.5 seconds ahead

            lateral_control_dict -- dictionary of arguments to setup the lateral PID controller
                                    {'K_P':, 'K_D':, 'K_I':, 'dt'}

            longitudinal_control_dict -- dictionary of arguments to setup the longitudinal PID controller
                                        {'K_P':, 'K_D':, 'K_I':, 'dt'}
        """
        self._vehicle = vehicle
        self._map = self._vehicle.get_world().get_map()

        self._discrete_state = None
        self._change_buffer = Queue(maxsize=F)
        self.Qf = 0

        self._dt = None
        self._target_speed = None
        self._sampling_radius = None
        self._min_distance = None
        self._current_waypoint = None
        self._target_road_option = None
        self._next_waypoints = None
        self.target_waypoint = None
        self._vehicle_controller = None
        # queue with tuples of (waypoint, RoadOption)
        self._waypoints_queue = deque(maxlen=20000)
        self._buffer_size = 5
        self._waypoint_buffer = deque(maxlen=self._buffer_size)

        # initializing controller
        self._init_controller(opt_dict)

    def __del__(self):
        if self._vehicle:
            self._vehicle.destroy()
        print("Destroying ego-vehicle!")

    def reset_vehicle(self):
        self._vehicle = None
        print("Resetting ego-vehicle!")

    def _init_controller(self, opt_dict):
        """
        Controller initialization.

        :param opt_dict: dictionary of arguments.
        :return:
        """
        # default params
        self._dt = 1.0 / 20.0
        self._switch_timestep = 0
        self._target_speed = 20.0  # Km/h
        self._sampling_radius = self._target_speed * 1 / 3.6  # 1 seconds horizon
        self._min_distance = self._sampling_radius * self.MIN_DISTANCE_PERCENTAGE
        args_lateral_dict = {
            'K_P': 0.4,#1.95,
            'K_D': 0.01,
            'K_I': 1.4,
            'dt': self._dt}
        args_longitudinal_dict = {
            'K_P': 1.0,
            'K_D': 0,
            'K_I': 1,
            'dt': self._dt}

        # parameters overload
        if opt_dict:
            if 'dt' in opt_dict:
                self._dt = opt_dict['dt']
            if 'target_speed' in opt_dict:
                self._target_speed = opt_dict['target_speed']
            if 'sampling_radius' in opt_dict:
                self._sampling_radius = self._target_speed * \
                                        opt_dict['sampling_radius'] / 3.6
            if 'lateral_control_dict' in opt_dict:
                args_lateral_dict = opt_dict['lateral_control_dict']
            if 'longitudinal_control_dict' in opt_dict:
                args_longitudinal_dict = opt_dict['longitudinal_control_dict']

        while not self._change_buffer.full():
            self._change_buffer.put(False)

        self._current_waypoint = self._map.get_waypoint(self._vehicle.get_location())
        self._vehicle_controller = VehiclePIDController(self._vehicle,
                                                        args_lateral=args_lateral_dict,
                                                        args_longitudinal=args_longitudinal_dict)

        # compute initial waypoints
        self._waypoints_queue.append((self._current_waypoint.next(self._sampling_radius)[0], RoadOption.LANEFOLLOW))

        self._target_road_option = RoadOption.LANEFOLLOW
        # fill waypoint trajectory queue
        self._compute_next_waypoints(k=200)

    def set_speed(self, speed):
        """
        Request new target speed.

        :param speed: new target speed in Km/h
        :return:
        """
        self._target_speed = speed

    def _compute_next_waypoints(self, k=1):
        """
        Add new waypoints to the trajectory queue.

        :param k: how many waypoints to compute
        :return:
        """
        # check we do not overflow the queue
        available_entries = self._waypoints_queue.maxlen - len(self._waypoints_queue)
        k = min(available_entries, k)

        for _ in range(k):
            last_waypoint = self._waypoints_queue[-1][0]
            next_waypoints = list(last_waypoint.next(self._sampling_radius))

            if len(next_waypoints) == 1:
                # only one option available ==> lanefollowing
                next_waypoint = next_waypoints[0]
                road_option = RoadOption.LANEFOLLOW
            else:
                road_options_list = _retrieve_options(
                    next_waypoints, last_waypoint)
                road_option = RoadOption.STRAIGHT if RoadOption.STRAIGHT in road_options_list else random.choice(road_options_list)
                next_waypoint = next_waypoints[road_options_list.index(
                    road_option)]

            self._waypoints_queue.append((next_waypoint, road_option))

    def run_step(self, debug=True):
        change_lane = False
        if self._switch_timestep == Tds - 1: #time cycle to evaluate switching options

            if self._target_road_option == RoadOption.LANEFOLLOW and self.target_waypoint:
                left_waypt = self.target_waypoint.get_left_lane()
                right_waypt = self.target_waypoint.get_right_lane()

                ego = self._vehicle
                current_waypoint = self._map.get_waypoint(ego.get_location())
                Qv_current = 0
                Qv_left = 0
                Qv_right = 0

                current_eps_neighbors = []
                left_eps_neighbors = []
                right_eps_neighbors = []

                #### GET VELOCITIES FROM EPS NEIGHBORS ####
                for other in ego.get_world().get_actors().filter('*vehicle*'):
                    if ego.id == other.id:
                        continue

                    other_waypoint = self._map.get_waypoint(other.get_location())
                    if other_waypoint.road_id != current_waypoint.road_id:
                        continue

                    if (other_waypoint.lane_id == current_waypoint.lane_id
                        and ego.get_location().distance(other.get_location()) < eps):
                            current_eps_neighbors.append(other)
                            # DEBUG: This is from the CARLA client not the controller
                            fwd_speed = scalar_proj(other.get_velocity(), other_waypoint.transform.get_forward_vector())
                            Qv_current = Qv_current + fwd_speed

                    elif (left_waypt
                        and other_waypoint.lane_id == left_waypt.lane_id
                        and ego.get_location().distance(other.get_location()) < eps):
                            left_eps_neighbors.append(other)
                            fwd_speed = scalar_proj(other.get_velocity(), other_waypoint.transform.get_forward_vector())
                            Qv_left = Qv_left + fwd_speed

                    elif (right_waypt
                        and other_waypoint.lane_id == right_waypt.lane_id
                        and ego.get_location().distance(other.get_location()) < eps):
                            right_eps_neighbors.append(other)
                            fwd_speed = scalar_proj(other.get_velocity(), other_waypoint.transform.get_forward_vector())
                            Qv_right = Qv_right + fwd_speed

                # TODO: Make this param
                Qv_current = Qv_current / len(current_eps_neighbors) if current_eps_neighbors else 0.9*self._target_speed
                Qv_left = Qv_left / len(left_eps_neighbors) if left_eps_neighbors else 0.9*self._target_speed
                Qv_right = Qv_right / len(right_eps_neighbors) if right_eps_neighbors else 0.9*self._target_speed
                # DEBUG: default values of Qv if no neighbors?
                #### GET VELOCITIES FROM EPS NEIGHBORS ####

                # DEBUG: simplify to one param. w*Qv - Qf >= 0
                rCL = w*(Qv_left - Qv_current) - self.Qf
                rCR = w*(Qv_right - Qv_current) - self.Qf

                # DEBUG: Default to left change over right change, right?
                # DEBUG: Why look at cars behind when calculating Qv?
                # DEBUG: Just use closest car in front, not all cars? (This is due to only using next velocity)
                # DEBUG: Certain drivers just want to go slow... they shouldn't switch if they won't speed up
                if (left_waypt
                    and str(current_waypoint.lane_change) in {'Left', 'Both'}
                    and rCL >= theta_left):
                        # TODO: Safety checking with cts controller
                        change_lane = True
                        self._waypoint_buffer.clear()
                        self._waypoints_queue.clear()
                        self._waypoints_queue.append((left_waypt.next(3.0)[0], RoadOption.CHANGELANELEFT))
                        print('Ego', ego.id, 'Qv_current', Qv_current, 'Qv_left', Qv_left, 'Qf', self.Qf)
                        print('Ego', ego.id, 'CHANGE' , current_waypoint.lane_id, 'into', left_waypt.lane_id)

                elif (right_waypt
                    and str(current_waypoint.lane_change) in {'Right', 'Both'}
                    and rCR >= theta_right):
                        change_lane = True
                        self._waypoint_buffer.clear()
                        self._waypoints_queue.clear()
                        self._waypoints_queue.append((right_waypt.next(3.0)[0], RoadOption.CHANGELANERIGHT))
                        print('Ego', ego.id, 'Qv_current', Qv_current, 'Qv_right', Qv_right, 'Qf', self.Qf)
                        print('Ego', ego.id, 'CHANGE' , current_waypoint.lane_id, 'into', right_waypt.lane_id)

            elif self._target_road_option == RoadOption.CHANGELANELEFT:
                pass

            elif self._target_road_option == RoadOption.CHANGELANERIGHT:
                pass

            else: #at intersection, roadoption is left, right or straight:
                pass

        else:
            pass

        control = self.run_planner_step(debug)

        self._switch_timestep = (self._switch_timestep + 1) % Tds

        #Drop oldest change_lane value and add newest. Update moving sum Qf
        self.Qf = self.Qf - self._change_buffer.get() + change_lane
        self._change_buffer.put(change_lane)

        return control

    def run_planner_step(self, debug=True):
        """
        Execute one step of local planning which involves running the longitudinal and lateral PID controllers to
        follow the waypoints trajectory.

        :param debug: boolean flag to activate waypoints debugging
        :return:
        """

        # not enough waypoints in the horizon? => add more!
        if len(self._waypoints_queue) < int(self._waypoints_queue.maxlen * 0.5):
            self._compute_next_waypoints(k=100)

        if len(self._waypoints_queue) == 0:
            control = carla.VehicleControl()
            control.steer = 0.0
            control.throttle = 0.0
            control.brake = 1.0
            control.hand_brake = False
            control.manual_gear_shift = False

            return control

        #   Buffering the waypoints
        if not self._waypoint_buffer:
            for i in range(self._buffer_size):
                if self._waypoints_queue:
                    self._waypoint_buffer.append(
                        self._waypoints_queue.popleft())
                else:
                    break

        # current vehicle waypoint
        vehicle_transform = self._vehicle.get_transform()
        self._current_waypoint = self._map.get_waypoint(vehicle_transform.location)
        # target waypoint
        self.target_waypoint, self._target_road_option = self._waypoint_buffer[0]
        # move using PID controllers
        control = self._vehicle_controller.run_step(self._target_speed, self.target_waypoint)

        # purge the queue of obsolete waypoints
        max_index = -1

        for i, (waypoint, _) in enumerate(self._waypoint_buffer):
            if waypoint.transform.location.distance(vehicle_transform.location) < self._min_distance:
                max_index = i
        if max_index >= 0:
            for i in range(max_index + 1):
                self._waypoint_buffer.popleft()

        if debug:
            draw_waypoints(self._vehicle.get_world(), [self.target_waypoint], self._vehicle.get_location().z + 1.0)

        return control


def _retrieve_options(list_waypoints, current_waypoint):
    """
    Compute the type of connection between the current active waypoint and the multiple waypoints present in
    list_waypoints. The result is encoded as a list of RoadOption enums.

    :param list_waypoints: list with the possible target waypoints in case of multiple options
    :param current_waypoint: current active waypoint
    :return: list of RoadOption enums representing the type of connection from the active waypoint to each
             candidate in list_waypoints
    """
    options = []
    for next_waypoint in list_waypoints:
        # this is needed because something we are linking to
        # the beggining of an intersection, therefore the
        # variation in angle is small
        next_next_waypoint = next_waypoint.next(3.0)[0]
        link = _compute_connection(current_waypoint, next_next_waypoint)
        options.append(link)

    return options

def _compute_connection(current_waypoint, next_waypoint):
    """
    Compute the type of topological connection between an active waypoint (current_waypoint) and a target waypoint
    (next_waypoint).

    :param current_waypoint: active waypoint
    :param next_waypoint: target waypoint
    :return: the type of topological connection encoded as a RoadOption enum:
             RoadOption.STRAIGHT
             RoadOption.LEFT
             RoadOption.RIGHT
    """
    n = next_waypoint.transform.rotation.yaw
    n = n % 360.0

    c = current_waypoint.transform.rotation.yaw
    c = c % 360.0

    diff_angle = (n - c) % 180.0
    if diff_angle < 1.0:
        return RoadOption.STRAIGHT
    elif diff_angle > 90.0:
        return RoadOption.LEFT
    else:
        return RoadOption.RIGHT


def norm(v):
    return (v.x**2 + v.y**2 + v.z**2)**0.5

def dot(u, v):
    return u.x*v.x + u.y*v.y + u.z*v.z

def scalar_proj(u, v):
    return dot(u, v) / norm(v)
