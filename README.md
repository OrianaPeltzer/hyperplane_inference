# hyperplane_inference
Joystick setup for hyperplane inference problem

Dependencies:

Julia 1.3 (add POMDPs, Distributions, Random, Parameters, BeliefUpdaters, StaticArrays, BasicPOMCP, POMDPModelTools, D3Trees, POMDPModels, QMDP, QuickPOMDPs, StatsBase, DataStructures, LightGraphs, MetaGraphs, GraphPlot, POMDPSimulators, POMDPGifs, Colors, Compose, Gadfly, LinearAlgebra, Blink, Infiltrator, RobotOS)

Python 2.7

ROS Melodic http://wiki.ros.org/melodic/Installation/Ubuntu

(for using joystick) pip install pyPS4Controller

Setup

1. create a new package in catkin_ws (ex catkin_ws/src/joystick_pomcp)
2. the files in this repo go into the src folder of the new package.
3. cd catkin_ws
4. catkin_make
