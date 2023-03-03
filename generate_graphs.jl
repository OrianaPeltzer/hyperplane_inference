using POMDPs
using Distributions: Normal
using Random
using Parameters
using POMDPTools
using StaticArrays
using BasicPOMCP
using D3Trees
using POMDPModels
using QMDP
using QuickPOMDPs
using StatsBase
using DataStructures # For astar priority queue
using Graphs, MetaGraphs
using GraphPlot
using POMDPGifs
using Colors
using Compose
using Gadfly
using LinearAlgebra
using Blink

# For embed
using Infiltrator

include("hyperplane_inference_gridworld_pomdp/convex_sets.jl")

# Map of the environment
# map_name = "2_10x10_squareobstacle_map"
# map_name = "3_20x20_office_map"
map_name = "4_10x10_office_map"
map_dir = "maps/"*map_name*".jl"
include(map_dir)

# Grid world, map, pomdp and problem-specific functions
include("hyperplane_inference_gridworld_pomdp/world.jl")
include("hyperplane_inference_gridworld_pomdp/hyper_a_star.jl")
# Helper functions for problem specifics
include("hyperplane_inference_gridworld_pomdp/utilities.jl")
# POMCP-specific functions to solve the POMDP
include("hyperplane_inference_gridworld_pomdp/pomcp_functions.jl")
# To visualize
include("hyperplane_inference_gridworld_pomdp/visualization.jl")

OBSTACLE_MAP = map_occupancy_grid()
G = map_hyperplane_graph()

map_graph, K_map = initialize_grid_graph(OBSTACLE_MAP)

start_pos, true_goal_index, goal_options, M_pref = map_problem_setup()

mapworld_pomdp = MapWorld(obstacle_map = OBSTACLE_MAP,
                          grid_side = 10,
                          goal_options = goal_options,
                          map_graph = map_graph,
                          hyperplane_graph = G,
                          start_position = start_pos,
                          true_goal_index = true_goal_index,
                          true_preference = M_pref) # Other args are default

@show vertices(mapworld_pomdp.hyperplane_graph)

start_x, start_y = map_initial_pos()

robot_state = initial_state(mapworld_pomdp)

# Save the hyperplane graph
savegraph("maps/"*map_name*".mg", G)

# Render the map
c = POMDPTools.render(mapworld_pomdp, robot_state, G)
draw(SVGJS("maps/"*map_name*".svg", 6inch, 6inch), c)




# Ubuntu
# loadurl(my_window, "file:///home/peltzer/catkin_ws/src/joystick_pomcp/src/hyperplane_inference/foo.svg")
# Windows
# loadurl(my_window, "c:/opt/ros/melodic/x64/catkin_ws/src/hyperplane_inference/src/hyperplane_inference/foo.svg")
