from queue import Queue

from agents.discrete_nav.agent import Agent, AgentState
from agents.discrete_nav.local_planner import RoadOption, LocalPlanner
from agents.tools.misc import is_within_distance_ahead, scalar_proj, dot, norm


Tds = 10 #Number of timesteps to check behavior planner
F = 50 # number of past timesteps to remember for lane changing
w = 0.05 # weight of Qv in lane change reward function
theta_left = 1.0 # threshold to switch left
theta_right = 1.0 # threshold to switch right
eps = 150 # meters # TODO: Google DSRC?
theta_a = 2.0 # meters/sec^2 # threshold above which acceleration is "uncomfortable"


class RoamingAgent(Agent):
    """
    RoamingAgent uses the connected behavior-planning algorithm to make lane
    change decisions
    """

    def __init__(self, dt, target_speed, vehicle):
        """

        :param vehicle: actor to apply to local planner logic onto
        """
        super(RoamingAgent, self).__init__(vehicle, dt)
        self.dt = dt
        self.target_speed = target_speed
        self.local_planner = LocalPlanner(self.vehicle, {'dt': dt,
                                                         'target_speed': target_speed})

        self.switcher_step = 0  # cycles through 0, 1, ..., (Tds - 1) each timestep
        self.proximity_threshold = 12.0  # meters
        self.chg_safety_distance = 4.0  # meters both ways
        self.chg_hazard_l = False
        self.hazard_c = False
        self.chg_hazard_r = False
        self.change_distance = 20.0  # meters

        self.Qv_l = 0.9 * self.target_speed
        self.Qv_c = 0.9 * self.target_speed
        self.Qv_r = 0.9 * self.target_speed
        self.change_buf = Queue(maxsize=F)
        while not self.change_buf.full():
            self.change_buf.put(False)
        self.Qf = 0
        self.rCL = 0
        self.rCR = 0

    def detect_nearby_vehicles(self):
        left_waypt = self.current_waypoint.get_left_lane()
        right_waypt = self.current_waypoint.get_right_lane()

        self.chg_hazard_l = False
        self.hazard_c = False
        self.chg_hazard_r = False

        nbrs_l, nbrs_c, nbrs_r = [], [], []

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
                if loc.distance(other_loc) < eps and dot(loc - other_loc, fwd) <= 0:
                    nbrs_l.append(other_fwd_speed)

                # Check if it's a hazard. Any one hazard should make the flag stay true
                self.chg_hazard_l = (self.chg_hazard_l or
                    abs(norm(other.get_location() - loc)) < self.chg_safety_distance)

            # Other is on CURRENT lane
            elif other_waypoint.lane_id == self.current_waypoint.lane_id:
                #Check if it's an eps-neighbor
                if loc.distance(other_loc) < eps and dot(loc - other_loc, fwd) <= 0:
                    nbrs_c.append(other_fwd_speed)

                # Check if it's a hazard. Any one hazard should make the flag stay true
                self.hazard_c = (self.hazard_c or
                    loc.distance(other_loc) < self.proximity_threshold and dot(loc - other_loc, fwd) <= 0)

            # Other is on RIGHT lane
            elif right_waypt and other_waypoint.lane_id == right_waypt.lane_id:
                #Check if it's an eps-neighbor
                if loc.distance(other_loc) < eps and dot(loc - other_loc, fwd) <= 0:
                    nbrs_r.append(other_fwd_speed)

                # Check if it's a hazard. Any one hazard should make the flag stay true
                self.chg_hazard_r = (self.chg_hazard_r or
                    abs(norm(other.get_location() - loc)) < self.chg_safety_distance)

        self.Qv_l = sum(nbrs_l)/len(nbrs_l) if nbrs_l else 0.9*self.target_speed
        self.Qv_c = sum(nbrs_c)/len(nbrs_c) if nbrs_c else 0.9*self.target_speed
        self.Qv_r = sum(nbrs_r)/len(nbrs_r) if nbrs_r else 0.9*self.target_speed

        self.rCL = w*(self.Qv_l - self.Qv_c) - self.Qf
        self.rCR = w*(self.Qv_r - self.Qv_c) - self.Qf

    def run_step(self, debug=False):
        """
        Execute one step of navigation.
        :return: carla.VehicleControl
        """
        self.t += self.dt # TODO: Call super run_step first maybe?
        self.current_waypoint = self.map.get_waypoint(self.vehicle.get_location())

        self.detect_nearby_vehicles()

        if self.hazard_c:
            self.state = AgentState.BLOCKED_BY_VEHICLE
            return self.emergency_stop()

        if self.discrete_state() == RoadOption.CHANGELANELEFT:
            if self.chg_hazard_l:
                # Cancel the attempted lane change
                self.local_planner.set_lane_right(self.change_distance)

        elif self.discrete_state() == RoadOption.CHANGELANERIGHT:
            if self.chg_hazard_r:
                # Cancel the attempted lane change
                self.local_planner.set_lane_left(self.change_distance)

        elif (self.discrete_state() == RoadOption.LANEFOLLOW
            and self.local_planner.target_waypoint
            and self.switcher_step == Tds - 1):

                    self.state = AgentState.NAVIGATING
                    change_lane = False

                    # Check if we can change left
                    if (self.rCL >= theta_left
                        and str(self.current_waypoint.lane_change) in {'Left', 'Both'}
                        and not self.chg_hazard_l):

                            change_lane = True
                            self.local_planner.set_lane_left(self.change_distance)

                    # Check if we can change right
                    elif (self.rCR >= theta_right
                          and str(self.current_waypoint.lane_change) in {'Right', 'Both'}
                          and not self.chg_hazard_r):

                            change_lane = True
                            self.local_planner.set_lane_right(self.change_distance)

                    # Update Qf and save most recent change_lane value
                    self.Qf = self.Qf - self.change_buf.get() + change_lane
                    self.change_buf.put(change_lane)

        self.switcher_step = (self.switcher_step + 1) % Tds
        return self.local_planner.run_step()
