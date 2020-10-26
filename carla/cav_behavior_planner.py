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

from queue import Queue
from enum import Enum

from behavior_planner import BehaviorPlanner
from path_planner import RoadOption, PathPlanner
from tools.misc import scalar_proj, dot, norm


class Behavior(Enum):
    KEEP_LANE = 0
    CHANGE_LEFT = 1
    CHANGE_RIGHT = 2


class CAVBehaviorPlanner(BehaviorPlanner):
    """
    CAVBehaviorPlanner uses the connected behavior-planning algorithm to make
    lane change decisions
    """

    def __init__(self, dt, target_speed, vehicle, param_dict, CAV_agents_dict):
        super(CAVBehaviorPlanner, self).__init__(vehicle, dt, param_dict)
        self.dt = dt
        self.target_speed = target_speed
        self.path_planner = PathPlanner(self.vehicle, {'dt': dt,
                                                       'target_speed': target_speed})

        self.F = param_dict['F']
        self.w = param_dict['w']
        self.theta_CL = param_dict['theta_CL']
        self.theta_CR = param_dict['theta_CR']
        self.eps = param_dict['eps']

        self.chg_hazard_l = False
        self.hazard_c = False
        self.chg_hazard_r = False

        self.switcher_step = 0  # cycles through 0, 1, ..., (Tds - 1) each timestep
        self.Qv_l = 0.9 * self.target_speed
        self.Qv_c = 0.9 * self.target_speed
        self.Qv_r = 0.9 * self.target_speed
        self.change_buf = Queue(maxsize=self.F)
        while not self.change_buf.full():
            self.change_buf.put(False)
        self.Qf = 0
        self.rCL = 0
        self.rCR = 0
        
        self.neighbor_left = []
        self.neighbor_current = []
        self.neighbor_right = []
        
        self.behavior = Behavior.KEEP_LANE
        
        self.closeneighbor_left = []
        self.closeneighbor_current = []
        self.closeneighbor_right = []
        
        self.CAV_agents_dict = CAV_agents_dict

    def detect_nearby_vehicles(self):
        self.lanechanging_conflict = False
        
        self.close_eps = 30
        
        left_waypt = self.current_waypoint.get_left_lane()
        right_waypt = self.current_waypoint.get_right_lane()

        self.chg_hazard_l = False  # there is a hazard on the left
        self.hazard_c = False  # there is a hazard ahead
        self.chg_hazard_r = False  # there is a hazard on the right

        nbrs_l, nbrs_c, nbrs_r = [], [], []
        self.neighbor_left, self.neighbor_current, self.neighbor_right = [], [], []

        for other in self.world.get_actors().filter("*vehicle*"):
            # must be a different vehicle
            if self.vehicle.id == other.id:
                continue

            other_loc = other.get_location()
            other_waypoint = self.map.get_waypoint(other_loc)

            # must be on the same segment of road as ego vehicle
            if other_waypoint.road_id != self.current_waypoint.road_id:
                continue

            loc = self.vehicle.get_location()
            fwd = self.current_waypoint.transform.get_forward_vector()
            other_fwd_speed = scalar_proj(other.get_velocity(),
                                          other_waypoint.transform.get_forward_vector())

            # Other is on LEFT lane
            if left_waypt and other_waypoint.lane_id == left_waypt.lane_id:
                #Check if it's an eps-neighbor
                if loc.distance(other_loc) < self.eps and dot(loc - other_loc, fwd) <= 0:
                    nbrs_l.append(other_fwd_speed)
                    self.neighbor_left.append(other)
                    
                if loc.distance(other_loc) < self.close_eps:
                    self.closeneighbor_left.append(other)

                # Check if it's a hazard. Any one hazard should make the flag stay true
                self.chg_hazard_l = (self.chg_hazard_l or
                    abs(norm(other.get_location() - loc)) < self.theta_l)

            # Other is on CURRENT lane
            elif other_waypoint.lane_id == self.current_waypoint.lane_id:
                #Check if it's an eps-neighbor
                if loc.distance(other_loc) < self.eps and dot(loc - other_loc, fwd) <= 0:
                    nbrs_c.append(other_fwd_speed)
                    self.neighbor_current.append(other)

                # Check if it's a hazard. Any one hazard should make the flag stay true
                self.hazard_c = (self.hazard_c or
                    loc.distance(other_loc) < self.theta_c and dot(loc - other_loc, fwd) <= 0)

            # Other is on RIGHT lane
            elif right_waypt and other_waypoint.lane_id == right_waypt.lane_id:
                #Check if it's an eps-neighbor
                if loc.distance(other_loc) < self.eps and dot(loc - other_loc, fwd) <= 0:
                    nbrs_r.append(other_fwd_speed)
                    self.neighbor_right.append(other)
                    
                if loc.distance(other_loc) < self.close_eps:
                    self.closeneighbor_right.append(other)

                # Check if it's a hazard. Any one hazard should make the flag stay true
                self.chg_hazard_r = (self.chg_hazard_r or
                    abs(norm(other.get_location() - loc)) < self.theta_r)

        self.Qv_l = sum(nbrs_l)/len(nbrs_l) if nbrs_l else 0.9*self.target_speed
        self.Qv_c = sum(nbrs_c)/len(nbrs_c) if nbrs_c else 0.9*self.target_speed
        self.Qv_r = sum(nbrs_r)/len(nbrs_r) if nbrs_r else 0.9*self.target_speed

        self.rCL = self.w*(self.Qv_l - self.Qv_c) - self.Qf
        self.rCR = self.w*(self.Qv_r - self.Qv_c) - self.Qf
        
    def left_change_conflict_detection(self):
        """
        detect whether there is a conflict with other vehicles on the left lane
        before starting a lane-changing
        """
        for vehicle in self.closeneighbor_left:
            if self.CAV_agents_dict[vehicle.id].discrete_state() in {RoadOption.CHANGELANELEFT, RoadOption.CHANGELANERIGHT}:
                return True
            
        return False
    
    def right_change_conflict_detection(self):
        """
        detect whether there is a conflict with other vehicles on the right lane
        before starting a lane-changing
        """
        for vehicle in self.closeneighbor_right:
            if self.CAV_agents_dict[vehicle.id].discrete_state() in {RoadOption.CHANGELANELEFT, RoadOption.CHANGELANERIGHT}:
                return True
            
        return False

    def run_step(self, debug=False):
        self.current_waypoint = self.map.get_waypoint(self.vehicle.get_location())

        self.detect_nearby_vehicles()

        if self.hazard_c:
            self.behavior = Behavior.KEEP_LANE
            return self.emergency_stop()

        if self.discrete_state() == RoadOption.CHANGELANELEFT:
            if self.chg_hazard_l or self.left_change_conflict_detection():
                self.behavior = Behavior.CHANGE_RIGHT
                # Cancel the attempted lane change
                self.path_planner.set_lane_right(self.change_distance)

        elif self.discrete_state() == RoadOption.CHANGELANERIGHT:
            if self.chg_hazard_r or self.right_change_conflict_detection():
                self.behavior = Behavior.CHANGE_LEFT
                # Cancel the attempted lane change
                self.path_planner.set_lane_left(self.change_distance)

        elif (self.discrete_state() == RoadOption.LANEFOLLOW
            and self.path_planner.target_waypoint
            and self.switcher_step == self.Tds - 1):

                    change_lane = False
                    self.behavior = Behavior.KEEP_LANE

                    # Check if we can change left
                    if (self.rCL >= self.theta_CL
                        and str(self.current_waypoint.lane_change) in {'Left', 'Both'}
                        and not self.chg_hazard_l
                        and not self.left_change_conflict_detection()):

                            change_lane = True
                            self.behavior = Behavior.CHANGE_LEFT
                            self.path_planner.set_lane_left(self.change_distance)

                    # Check if we can change right
                    elif (self.rCR >= self.theta_CR
                          and str(self.current_waypoint.lane_change) in {'Right', 'Both'}
                          and not self.chg_hazard_r
                          and not self.right_change_conflict_detection()):

                            change_lane = True
                            self.behavior = Behavior.CHANGE_RIGHT
                            self.path_planner.set_lane_right(self.change_distance)

                    # Update Qf and save most recent change_lane value
                    self.Qf = self.Qf - self.change_buf.get() + change_lane
                    self.change_buf.put(change_lane)

        self.switcher_step = (self.switcher_step + 1) % self.Tds
        return self.path_planner.run_step()
