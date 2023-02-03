# hyperplane_inference
Joystick setup for hyperplane inference problem

Dependencies:

Julia 1.8.5 (add POMDPs, Distributions, Random, Parameters, StaticArrays, BasicPOMCP, POMDPTools, D3Trees, POMDPModels, QMDP, QuickPOMDPs, StatsBase, DataStructures, Graphs, MetaGraphs, GraphPlot, POMDPGifs, Colors, Compose, Gadfly, LinearAlgebra, Blink, Infiltrator, RobotOS, PyCall)

Python 2.7

ROS Melodic http://wiki.ros.org/melodic/Installation/Ubuntu

In Julia, set ENV["PYTHON"]="path_to_ros_specific_python/name_of_executable" and rebuild PyCall

(for using joystick) pip install pyPS4Controller
(for windows install) pip install pygames

Setup

1. create a new package in catkin_ws (ex catkin_ws/src/joystick_pomcp)
2. this repo can go into the src folder of the new package.
3. cd catkin_ws
4. catkin_make

For launching the inference game:
- Plug in Joystick with USB.
- roscore

Launch rosnodes:
- angle_updater.py (or angle_updater_windows.py if using windows)
- game_manager.py
- hyperplane_inference_gridworld.jl
