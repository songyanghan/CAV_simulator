# CAV_simulator

This code is meant for use with the autonomous vehicle simulator CARLA (https://carla.org/)

To use:
- Install CARLA version >= 0.9.7 (tested with 0.9.7) using the documentation at https://carla.readthedocs.io/en/latest/. You can either clone and build from source or download a pre-packaged version directly.
- Launch the CARLA simulator by running the ./CarlaUE4.sh script (Linux) or CarlaUE4.exe (Windows). Location within the directory may vary based on install method, so refer to the documentation to find this
- In a separate terminal, run `python3 worlds/multi_automatic_control.py`
- You may need to install some dependencies first (pygame, numpy, etc) -- the script will exit and tell you if those aren't present
