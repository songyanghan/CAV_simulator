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

from __future__ import print_function

import argparse
import datetime
import glob
import logging
import os
import random
import re
import sys
from statistics import mean

try:
    import numpy as np
except ImportError:
    raise RuntimeError(
        'cannot import numpy, make sure numpy package is installed')

# ==============================================================================
# -- parameters for driving algorithms -----------------------------------------
# ==============================================================================
param_dict = {
    'Tds':          10,     # check for lane changes once every Tds timesteps
    'F':            50,     # past timesteps to remember for Qf quality factor
    'w':            0.4,    # weight of Qv in lane change reward function
    'theta_CL':     2.0,    # reward function threshold to switch left
    'theta_CR':     2.0,    # reward function threshold to switch right
    'eps':          150.0,  # [m] communication radius for connected vehicles
    'theta_a':      2.0,    # [m/s^2] "uncomfortable" acceleration threshold
    'p_l':          0.01,   # random change left probability
    'p_r':          0.01,   # random change right probability
    'theta_l':      4.0,    # [m] safety radius to prevent left changes
    'theta_c':      12.0,   # [m] safety cushion in front of vehicle
    'theta_r':      4.0,    # [m] safety radius to prevent right changes
    'chg_distance': 15.0}   # [m] longitudinal distance to travel while changing

# ==============================================================================
# -- find carla module ---------------------------------------------------------
# ==============================================================================
# To import Carla, we need to find the Python egg, and add it to the system path
# e.g. 'carla-0.9.9-py3.7-linux-x86_64.egg'.
# Different installations put the .egg file in different locations.

# Possible installation locations are in the list possible_paths:
possible_paths = ['../carla/dist/carla-*%d.%d-%s.egg',
    '/opt/carla/PythonAPI/carla/dist/carla-*%d.%d-%s.egg',
    '../carla-simulator/dist/carla-*%d.%d-%s.egg',
    '/opt/carla-simulator/PythonAPI/carla/dist/carla-*%d.%d-%s.egg']

# Look for the .egg in each path
for path in possible_paths:
    # glob.glob returns a list of all paths matching
    egg_path = glob.glob(path % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))

    # If the list is nonempty, then it means we found the .egg,
    # and we can add it to path.
    if len(egg_path) > 0:
        break

# Try to add egg_path[0] to the path. If we didn't find anything, egg_path
# is empty, and we get an index error.
try:
    sys.path.append(egg_path[0])
except IndexError:
    raise FileNotFoundError ("Can't find Carla .egg. This script may need updating, check your install.")

# ==============================================================================
# -- add PythonAPI for release mode --------------------------------------------
# ==============================================================================
try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/carla')
except IndexError:
    pass

import carla
from cav_behavior_planner import CAVBehaviorPlanner as CAV
from random_behavior_planner import RandomBehaviorPlanner as UCAV


# ==============================================================================
# -- Global functions ----------------------------------------------------------
# ==============================================================================

def find_weather_presets():
    rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
    name = lambda x: ' '.join(m.group(0) for m in rgx.finditer(x))
    presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]


def get_actor_display_name(actor, truncate=250):
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name


# ==============================================================================
# -- World ---------------------------------------------------------------
# ==============================================================================

class World(object):
    def __init__(self, client, carla_world, num_CAVs, num_UCAVs):
        self.client = client
        self.world = carla_world
        self.map = self.world.get_map()
        self.num_CAVs = num_CAVs
        self.num_UCAVs = num_UCAVs
        self.CAVs = [None] * self.num_CAVs
        self.UCAVs = [None] * self.num_UCAVs
        self._weather_presets = find_weather_presets()
        self._weather_index = 0
        self.restart()

    def restart(self):
        self.destroy()

        spawn_points = []
        if self.map.name == 'Town05':
            # These are three segments of straight road that I thought would be great spawn zones
            for x in np.linspace(-134, 100, 4):
                for y in (201, 205, 209):
                    transform = self.map.get_waypoint(carla.Location(x=x, y=y)).transform
                    transform.location.z = transform.location.z + 2.0
                    spawn_points.append(transform)

            for y in np.linspace(95, -87, 3):
                for x in (211, 208, 204):
                    transform = self.map.get_waypoint(carla.Location(x=x, y=y)).transform
                    transform.location.z = transform.location.z + 2.0
                    spawn_points.append(transform)

            for x in np.linspace(-134, 80, 4):
                for y in (-200, -204, -208):
                    transform = self.map.get_waypoint(carla.Location(x=x, y=y)).transform
                    transform.location.z = transform.location.z + 2.0
                    spawn_points.append(transform)
        else:
            spawn_points = self.map.get_spawn_points()
        spawn_points = random.sample(spawn_points, self.num_CAVs + self.num_UCAVs)

        CAVs_batch, UCAVs_batch = [], []
        blueprint = self.world.get_blueprint_library().filter('vehicle.audi.tt')[0]

        # Spawn all the CAVs
        blueprint.set_attribute('color', '255,0,0')
        for i in range(0, self.num_CAVs):
            CAVs_batch.append(carla.command.SpawnActor(blueprint, spawn_points[i]))
        self.CAVs = [self.world.get_actor(response.actor_id) for response in self.client.apply_batch_sync(CAVs_batch)]

        # Spawn all the UCAVs
        blueprint.set_attribute('color', '255,255,255')
        for i in range(0, self.num_UCAVs):
            UCAVs_batch.append(carla.command.SpawnActor(blueprint, spawn_points[self.num_CAVs + i]))
        self.UCAVs = [self.world.get_actor(response.actor_id) for response in self.client.apply_batch_sync(UCAVs_batch)]

    def next_weather(self, reverse=False):
        self._weather_index += -1 if reverse else 1
        self._weather_index %= len(self._weather_presets)
        preset = self._weather_presets[self._weather_index]
        self.world.set_weather(preset[0])

    def destroy(self):
        self.client.apply_batch([carla.command.DestroyActor(vehicle)
                                 for vehicle in self.CAVs + self.UCAVs
                                 if vehicle is not None])


# ==============================================================================
# -- game_loop() ---------------------------------------------------------
# ==============================================================================

def game_loop(args):
    world = None

    # We will record measurements at each timestep in a file
    dirname = os.path.join(os.path.dirname(__file__), '../run_logs')
    timestamp = datetime.datetime.now()
    filename = 'c%du%d_%s.txt' % (args.cavs, args.ucavs, timestamp)
    filepath = os.path.join(dirname, filename)

    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(4.0)

        world = World(client, client.load_world(args.map), args.cavs, args.ucavs)
        settings = world.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = args.timestep
        settings.no_rendering_mode = args.no_render
        world.world.apply_settings(settings)

        CAV_agents = []
        UCAV_agents = []
        # Generate varying vehicle target speeds within an allowed interval
        target_speeds = np.linspace(40, 70, args.cavs + args.ucavs).tolist()
        target_speeds = random.sample(target_speeds, k=len(target_speeds))

        for i, vehicle in enumerate(world.CAVs):
            CAV_agents.append(CAV(args.timestep, target_speeds[i], vehicle, param_dict))
        for i, vehicle in enumerate(world.UCAVs):
            UCAV_agents.append(UCAV(args.timestep, target_speeds[args.cavs + i], vehicle, param_dict))

        with open(filepath, 'a') as outfile:
            print('Simulation_Start:', timestamp, file=outfile)
            print('Arguments:', args, file=outfile)
            print('Num_Vehicles:', args.cavs + args.ucavs, file=outfile) # density x length
            print('Avg_Velocity', 'Avg_Comfort_Cost', file=outfile)

            print('Simulation started with step size', args.timestep, 'secs and', args.steps, 'steps!')
            step = 0

            while step < args.steps:
                if step % 500 == 0:
                    print('Reached step', step, 'of', args.steps)

                world.world.tick()
                step += 1

                # Run one step of behavior planning, path planning, and control
                batch = []
                for agent in CAV_agents:
                    batch.append(carla.command.ApplyVehicleControl(agent.vehicle, agent.run_step(debug=True)))
                for agent in UCAV_agents:
                    batch.append(carla.command.ApplyVehicleControl(agent.vehicle, agent.run_step(debug=True)))
                client.apply_batch(batch)

                # Save measurements to file
                v_fwds, comfort_costs = zip(*[agent.get_measurements() for agent in CAV_agents+UCAV_agents])
                if CAV_agents:
                    CAV_v_fwds, CAV_comfort_costs = zip(*[agent.get_measurements() for agent in CAV_agents])
                    print(mean(CAV_v_fwds), mean(CAV_comfort_costs), mean(v_fwds), mean(comfort_costs), file=outfile)
                else:
                    print(-1, -1, mean(v_fwds), mean(comfort_costs), file=outfile)

    except KeyboardInterrupt:
        print('Cancelled by user. Saving data and exiting!')

    finally:
        if world is not None:
            world.destroy()
            world.world.tick()

        print_report(filepath)


def print_report(filename):
    with open(filename, 'r') as infile:
        total_steps = 0
        mean_CAV_v_sum, mean_CAV_cc_sum = 0, 0
        mean_v_sum, mean_cc_sum = 0, 0

        infile.readline()
        infile.readline()
        num_vehicles = float(infile.readline().split()[1])
        infile.readline()
        line = infile.readline()
        while line:
            total_steps += 1
            line = line.split()
            mean_CAV_v_sum += float(line[0])
            mean_CAV_cc_sum += float(line[1])
            mean_v_sum += float(line[2])
            mean_cc_sum += float(line[3])
            line = infile.readline()

        print('Total Timesteps:', total_steps)
        if mean_CAV_cc_sum >= 0:
            print('Average CAV Velocity:', mean_CAV_v_sum*3.6 / total_steps)
            print('Average CAV Driving Comfort Cost:', mean_CAV_cc_sum / total_steps)
        else:
            print('Average CAV Velocity: N/A')
            print('Average CAV Driving Comfort Cost: N/A')
        print('Average Velocity:', mean_v_sum*3.6 / total_steps)
        print('Average Driving Comfort Cost:', mean_cc_sum / total_steps)

# ==============================================================================
# -- main() --------------------------------------------------------------
# ==============================================================================


def main():
    argparser = argparse.ArgumentParser(
        description='CARLA Manual Control Client')
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-t', '--timestep',
        metavar='DT',
        default=0.05,
        type=float,
        help='simulation timestep length in seconds (default: 0.05)')
    argparser.add_argument(
        '-m', '--map',
        default='Town05',
        type=str,
        help='map name (default: Town05)')
    argparser.add_argument(
        '-c', '--cavs',
        metavar='C',
        default=9,
        type=int,
        help='number of connected (behavior-planned) autonomous vehicles (default: 9)')
    argparser.add_argument(
        '-u', '--ucavs',
        metavar='U',
        default=5,
        type=int,
        help='number of unconnected autonomous vehicles (default: 5)')
    argparser.add_argument(
        '--no-render',
        default=False,
        action='store_true',
        help='prevent Unreal Engine from rendering the scene')
    argparser.add_argument(
        '-s', '--steps',
        type=int,
        default=3000,
        help='total number of simulation steps (default: 3000)'
        )
    args = argparser.parse_args()

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    game_loop(args)


if __name__ == '__main__':
    main()
