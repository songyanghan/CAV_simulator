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
from queue import Queue
from collections import deque
import random

import carla
from controller import VehiclePIDController
from tools.misc import draw_waypoints


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


class PathPlanner(object):
    """
    PathPlanner implements the basic behavior of following a trajectory of waypoints that is generated on-the-fly.
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

        self._dt = None
        self._target_speed = None
        self._sampling_radius = None
        self._min_distance = None
        self._current_waypoint = None
        self._target_road_option = None
        self.target_waypoint = None
        self._vehicle_controller = None

        # course charted by path planner
        # contains tuples of (waypoint, RoadOption)
        self._waypoints_queue = deque(maxlen=20000)
        self._buffer_size = 5
        # immediate next few waypoints in the planned path (helpful for
        # controllers like MPC which use the next SEVERAL reference positions)
        # In our case, PID only uses the next SINGLE waypoint so it's unused
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
        # default params
        self._dt = 1.0 / 20.0
        # self._switch_timestep = 0
        self._target_speed = 20.0  # Km/h
        self._sampling_radius = self._target_speed * 1 / 3.6  # 1 seconds horizon
        self._min_distance = self._sampling_radius * self.MIN_DISTANCE_PERCENTAGE
        args_lateral_dict = {
            'K_P': 0.5, # Keyur: 0.5  Local_planner: 1.95  Traffic_manager: 10
            'K_D': 0, # Keyur: 0.01  Local_planner: 0.2  Traffic_manager: 0
            'K_I': 0.1, # Keyur: 1.4  Local_planner: 0.07  Traffic_manager: 0.1
            'dt': self._dt}
        args_longitudinal_dict = {
            'K_P': 0.1, # Keyur: 1.0  Local_planner: 1.0  Traffic_manager: 5.0
            'K_D': 0.15, # Keyur: 0  Local_planner: 0  Traffic_manager: 0
            'K_I': 0.01, # Keyur: 1  Local_planner: 0.05  Traffic_manager: 0.1
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

        self._current_waypoint = self._map.get_waypoint(self._vehicle.get_location())
        self._vehicle_controller = VehiclePIDController(self._vehicle,
                                                        args_lateral=args_lateral_dict,
                                                        args_longitudinal=args_longitudinal_dict)

        # compute initial waypoints
        self._waypoints_queue.append((self._current_waypoint.next(self._sampling_radius)[0], RoadOption.LANEFOLLOW))

        self._target_road_option = RoadOption.LANEFOLLOW
        # fill waypoint trajectory queue
        self._compute_next_waypoints(k=200)

    def set_lane_left(self, distance_ahead, debug=True):
        current_waypoint = self._map.get_waypoint(self._vehicle.get_location())
        left_waypt = current_waypoint.get_left_lane()

        self._waypoint_buffer.clear()
        self._waypoints_queue.clear()
        self._waypoints_queue.append((left_waypt.next(distance_ahead)[0], RoadOption.CHANGELANELEFT))
        if debug:
            print('Ego', self._vehicle.id, 'CHANGE LEFT from lane', current_waypoint.lane_id,'into', left_waypt.lane_id)

    def set_lane_right(self, distance_ahead, debug=True):
        current_waypoint = self._map.get_waypoint(self._vehicle.get_location())
        right_waypt = current_waypoint.get_right_lane()

        self._waypoint_buffer.clear()
        self._waypoints_queue.clear()
        self._waypoints_queue.append((right_waypt.next(distance_ahead)[0], RoadOption.CHANGELANERIGHT))
        if debug:
            print('Ego', self._vehicle.id, 'CHANGE RIGHT from lane', current_waypoint.lane_id, 'into', right_waypt.lane_id)
            
    def set_lane_origin(self, distance_ahead, debug=True):
        current_waypoint = self._map.get_waypoint(self._vehicle.get_location())

        self._waypoint_buffer.clear()
        self._waypoints_queue.clear()
        self._waypoints_queue.append((current_waypoint.next(distance_ahead)[0], RoadOption.LANEFOLLOW))
        if debug:
            print('Ego', self._vehicle.id, 'Set back to the original lane to avoid collisions', current_waypoint.lane_id)

    def set_speed(self, speed):
        self._target_speed = speed

    def _compute_next_waypoints(self, k=1):  # Adds k new waypoints to queue
        # check we do not overflow the queue
        available_entries = self._waypoints_queue.maxlen - len(self._waypoints_queue)
        k = min(available_entries, k)

        for _ in range(k):
            last_waypoint = self._waypoints_queue[-1][0]
            next_waypoints = list(last_waypoint.next(self._sampling_radius))

            if len(next_waypoints) == 0: # fixes a bug # DEBUG:
                break
            elif len(next_waypoints) == 1:
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

    def run_step(self, debug=False):
        # not enough waypoints in the horizon? => add more!
        if len(self._waypoints_queue) < int(self._waypoints_queue.maxlen * 0.5):
            self._compute_next_waypoints(k=100)

        if len(self._waypoints_queue) == 0 and len(self._waypoint_buffer) == 0:
            control = carla.VehicleControl()
            control.steer = 0.0
            control.throttle = 0.0
            control.brake = 1.0
            control.hand_brake = False
            control.manual_gear_shift = False

            return control

        # Buffering the first few waypoints
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
