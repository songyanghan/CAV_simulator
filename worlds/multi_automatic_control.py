#!/usr/bin/env python

# Copyright (c) 2018 Intel Labs.
# authors: German Ros (german.ros@intel.com)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
    Example of automatic vehicle control from client side.
"""

from __future__ import print_function

import argparse
import collections
import datetime
import glob
import logging
import math
import os
import random
import re
import sys
import weakref
import time

try:
    import numpy as np
except ImportError:
    raise RuntimeError(
        'cannot import numpy, make sure numpy package is installed')


# ==============================================================================
# -- find carla module ---------------------------------------------------------
# ==============================================================================
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

# ==============================================================================
# -- add PythonAPI for release mode --------------------------------------------
# ==============================================================================
try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/carla')
except IndexError:
    pass

import carla
from agents.discrete_nav.roaming_agent import RoamingAgent
from agents.discrete_nav.random_agent import RandomAgent


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
    def __init__(self, client, carla_world, actor_filter, num_CAVs, num_UCAVs):
        self.client = client
        self.world = carla_world
        self.map = self.world.get_map()
        self.num_CAVs = num_CAVs
        self.num_UCAVs = num_UCAVs
        self.CAVs = [None] * self.num_CAVs
        self.UCAVs = [None] * self.num_UCAVs
        self._weather_presets = find_weather_presets()
        self._weather_index = 0
        self._actor_filter = actor_filter
        self.restart()

    def restart(self):
        self.destroy()

        spawn_points = []
        if self.map.name == 'Town05':
            # These are three segments of straight road that I thought would be great spawn zones
            for x in np.linspace(45, 95, 2):
                for y in (201, 205, 209):
                    transform = self.map.get_waypoint(carla.Location(x=x, y=y)).transform
                    transform.location.z = transform.location.z + 2.0
                    spawn_points.append(transform)

            for y in np.linspace(95, -87, 4):
                for x in (211, 208, 204):
                    transform = self.map.get_waypoint(carla.Location(x=x, y=y)).transform
                    transform.location.z = transform.location.z + 2.0
                    spawn_points.append(transform)

            for x in np.linspace(0, 100, 3):
                for y in (-200, -204, -208):
                    transform = self.map.get_waypoint(carla.Location(x=x, y=y)).transform
                    transform.location.z = transform.location.z + 2.0
                    spawn_points.append(transform)
        else:
            spawn_points = self.map.get_spawn_points()
        spawn_points = random.sample(spawn_points, self.num_CAVs + self.num_UCAVs)

        blueprints = self.world.get_blueprint_library().filter(self._actor_filter)

        batch = []
        for i in range(0, self.num_CAVs + self.num_UCAVs):
            # Get a random blueprint.
            blueprint = random.choice(blueprints)
            blueprint.set_attribute('role_name', 'ego')
            if blueprint.has_attribute('color'):
                color = random.choice(blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color', color)
            batch.append(carla.command.SpawnActor(blueprint, spawn_points[i]))
        CAVs_batch = batch[:self.num_CAVs]
        UCAVs_batch = batch[-self.num_UCAVs:]
        self.CAVs = [self.world.get_actor(response.actor_id) for response in self.client.apply_batch_sync(CAVs_batch)]
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

    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(4.0)

        world = World(client, client.load_world(args.map), args.filter, args.cavs, args.ucavs)
        settings = world.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = args.timestep
        world.world.apply_settings(settings)

        agents = []
        for vehicle in world.CAVs:
            # TODO: Make distribution more realistic. Do research
            target_speed = random.uniform(50, 100)
            agents.append(RoamingAgent(args.timestep, target_speed, vehicle))
        for vehicle in world.UCAVs:
            target_speed = random.uniform(50, 100)
            agents.append(RandomAgent(args.timestep, target_speed, vehicle))


        while True:
            world.world.tick()

            client.apply_batch(
                [carla.command.ApplyVehicleControl(agent.vehicle, agent.run_step(debug=True))
                for agent in agents])

    except IndexError:
        print('Died due to IndexError bug')

    finally:
        if world is not None:
            world.destroy()
            world.world.tick()


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
        '--filter',
        metavar='PATTERN',
        default='vehicle.audi.tt',
        help='actor filter (default: "vehicle.audi.tt")')
    args = argparser.parse_args()

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    print(__doc__)

    try:

        game_loop(args)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


if __name__ == '__main__':
    main()
