# CAV_simulator
Companion code for the UConn undergraduate Honors Thesis "Evaluating Driving Performance of a Novel Behavior Planning Model on Connected Autonomous Vehicles" by Keyur Shah (UConn '20). Thesis was advised by Dr. Fei Miao; see http://feimiao.org/research.html.

This code is meant for use with the autonomous vehicle simulator CARLA (https://carla.org/)

## How to Use
* Install CARLA version >= 0.9.7 (tested with 0.9.7) using the documentation at https://carla.readthedocs.io/en/latest/. You can either clone and build from source or download a pre-packaged version directly.
  * Note that building from source will require you to also build Unreal Engine and other dependencies. Make sure you download the right version of Unreal Engine based on the CARLA documentation for the target version.
* Launch the CARLA simulator by running the `./CarlaUE4.sh` script (Linux) or `CarlaUE4.exe` (Windows). Location within the directory may vary based on install method, so refer to CARLA documentation to find this.
* In a separate terminal, run `python3 worlds/multi_cavs_ucavs.py`
* You may need to install some dependencies first (pygame, numpy, etc) -- the script will exit and tell you if those aren't present

## CARLA
The CARLA simulator uses a client-server architecture, where the server process runs a simulator based on the Unreal Engine (v4.x). Client scripts interact with the simulator by calling a provided Python or C++ API that communicates with the simulator (server)'s TCP/IP port.

This repository only includes the CLIENT-side code -- the server-side process is the `CarlaUE4.sh` / `CarlaUE4.exe` script built using the instructions above.

## Code Repository
`worlds/multi_cavs_ucavs.py` is the client-side script which:
1. Uses the Python API to initiate communication with the server
2. Sets up a simulation world with the desired settings
3. Spawns the desired numbers of vehicles in random (but pre-defined) locations
4. (Inner loop) Runs the simulation world for one tick (i.e. timestep)
5. (Inner loop) Updates the state of all vehicles and computes new control inputs
6. (Inner loop) Saves relevant measurements to a file in the `run_logs/` directory
7. Aggregates the individual measurements and prints final values

The vehicles' driving behavior is controlled by three layers of control:
1. Behavior Planner: updates discrete variables and sets target behavior based on road state
2. Path Planner: based on the target behavior, charts a reference trajectory of evenly-spaced waypoints along the center line of the target lane
3. Controller: updates continuous state variables and calculates control inputs based on error compared to reference trajectory

## Running the Simulation
The `worlds/multi_cavs_ucavs.py` script takes several command-line arguments. The ones you are most likely to change are:
| Argument    | Description |
| ---         | ---         |
| -t [FLOAT]  | Float number of seconds for each timestep to last. DO NOT increase above 0.1 |
| -c [INT]    | Integer number of connected vehicles (not advised to exceed 25 vehicles total unless you're sure the hardware can handle it). |
| -u [INT]    | Integer number of unconnected (random) vehicles (not advised to exceed 25 vehicles total unless you're sure the hardware can handle it). |
| --no-render | Run in no-rendering mode. The physics engine will still run, but the graphics loop will not and nothing will show up on screen. Speeds up simulation. |
| -s [INT]    | Integer total number of timesteps to run the simulation for before saving and exiting. |

You can also run `python3 worlds/multi_cavs_ucavs.py --help` to see more about all the arguments.

NOTE: Specific parameter values for the driving algorithm are defined within the `worlds/multi_cavs_ucavs.py` file and can be changed there.
