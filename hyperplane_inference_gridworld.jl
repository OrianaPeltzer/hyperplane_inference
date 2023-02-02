# Gridworld Inference problem: Infer the goal and neighboring hyperplanes.
# Action is compliant with human instructions.

using POMDPs
using Distributions: Normal
using Random
using Parameters
using BeliefUpdaters
using StaticArrays
using BasicPOMCP
using POMDPModelTools
using D3Trees
using POMDPModels
using QMDP
using QuickPOMDPs

using StatsBase
using DataStructures # For astar priority queue

using LightGraphs, MetaGraphs
using GraphPlot

using POMDPSimulators
using POMDPGifs
using Colors
using Compose
using Gadfly

using LinearAlgebra

using Blink

# For embed
using Infiltrator

#!/usr/bin/env julia

using RobotOS
@rosimport std_msgs.msg.Header
@rosimport geometry_msgs.msg: Point
@rosimport std_msgs.msg: Float64, Bool
rostypegen()
using .geometry_msgs.msg
using .std_msgs.msg


# Grid world, map, pomdp and problem-specific functions
include("hyperplane_inference_gridworld/world.jl")

include("hyperplane_inference_gridworld/hyper_a_star.jl")

# Helper functions for problem specifics
include("hyperplane_inference_gridworld/utilities.jl")
# POMCP-specific functions to solve the POMDP
include("hyperplane_inference_gridworld/pomcp_functions.jl")
# To visualize
include("hyperplane_inference_gridworld/visualization.jl")

# All the global variables (sorry!)
global mapworld_pomdp
# global solver
global policy
global curr_belstate
global true_state
global my_window

# Create map and problem
function initialize_map()

    global mapworld_pomdp
    global curr_belstate
    global true_state
    global my_window

    RNG = MersenneTwister(1234)

    # Generate a true 10x10 false/true map and ensure start position is empty (false)
    grid_side = 10
    OBSTACLE_MAP = fill(false, grid_side, grid_side)
    OBSTACLE_MAP[5,5] = true
    OBSTACLE_MAP[5,6] = true
    OBSTACLE_MAP[6,5] = true
    OBSTACLE_MAP[6,6] = true
    println("OBSTACLE MAP: ", OBSTACLE_MAP)
    # OBSTACLE_MAP[1, 1] = false
    # OBSTACLE_MAP[1, 2] = false # Giving robot a way out hehe

    BII_gamma=0.9

    map_graph, K_map = initialize_grid_graph(OBSTACLE_MAP)

    draw(SVG("mapgraph.svg", 16cm, 16cm), gplot(map_graph))

    # println("Vertices in map_graph: ", vertices(map_graph))

    goal_options = [GridPosition(x,8) for x=1:grid_side if OBSTACLE_MAP[x,8] == false]

    dist_matrix, dist_matrix_DP = get_dist_matrix(map_graph)

    @show K_map
    @show goal_options

    @show dist_matrix[11,5]
    @show dist_matrix_DP[11,5]

    # define POMDP
    mapworld_pomdp = MapWorld(obstacle_map = OBSTACLE_MAP,
                              grid_side = grid_side,
                              goal_options = goal_options,
                              n_possible_goals = length(goal_options),
                              map_graph = map_graph,
                              K_map = K_map,
                              dist_matrix = dist_matrix,
                              dist_matrix_DP = dist_matrix_DP,
                              BII_gamma = BII_gamma) # Other args are default

    curr_belstate = initial_belief_state(mapworld_pomdp)
    true_state = initial_state(mapworld_pomdp)

    # SHORTEST PATH DEBUGGING
    # start_vertex = 5
    # end_vertex = 11
    #
    # preference_constraint = NeighborConstraint(curr_belstate.neighbor_A,curr_belstate.neighbor_b,1,true)
    # path = LightGraphs.a_star(mapworld_pomdp.map_graph, start_vertex, end_vertex, preference_constraint, mapworld_pomdp.dist_matrix_DP)
    #
    # @show path

    # For visualization
    c = POMDPModelTools.render(mapworld_pomdp, true_state, curr_belstate)
    draw(SVGJS("foo.svg", 6inch, 6inch), c)
    loadurl(my_window, "file:///home/peltzer/catkin_ws/src/joystick_pomcp/src/foo.svg")

    return
end

# Main loop - is called each time an observation is received
function iterate_pomcp(angle::Float64)

    RNG = MersenneTwister(1234)

    println("Started pomcp solving")

    global mapworld_pomdp
    global curr_belstate
    global true_state
    global my_window

    # are we at the right goal or not?
    curr_pos = true_state.position

    goal = mapworld_pomdp.goal_options[true_state.goal_index]

    # If we are at the goal, return
    if curr_pos == goal
        return
    end

    # Turn the input angle into an observation
    println("Angle detected: ", angle)
    heading = angle_to_heading(angle)
    println("Resulting heading: ", heading)
    o = HumanInputObservation(heading,true,true)

    # println("Created human observation")

    # Update the belief state. The action is irrelevant.
    curr_belstate = update_goal_belief(mapworld_pomdp, curr_belstate, o)

    # println("Belief after goal update")
    # @show curr_belstate

    println("Finished updating goal belief")


    # Take compliant action
    a = heading
    # a, info = action_info(policy, curr_belstate, tree_in_info=true);
    @show a

    # Generate new state (the observation is irrelevant)
    # true_state, r, o = gen(mapworld_pomdp, true_state, a, RNG)

    curr_belstate = update_pose_belief(mapworld_pomdp, curr_belstate, a)

    println("done updating pose belief")

    # Update true state
    true_state = GridState(curr_belstate.position, true_state.done, true_state.goal_index, true_state.neighbor_A, true_state.neighbor_b)
    # @show r

    @show curr_belstate

    display(curr_belstate.belief_intention)
    println("")

    println("Marginal preferences:")
    display(sum(curr_belstate.belief_intention, dims=1))
    println("")

    if true_state.position == goal
        println("--------------------------")
        println("   Made it to the goal!!  ")
        println("--------------------------")
    end

    c = POMDPModelTools.render(mapworld_pomdp, true_state, curr_belstate)

    draw(SVGJS("foo.svg", 6inch, 6inch), c)
    loadurl(my_window, "file:///home/peltzer/catkin_ws/src/joystick_pomcp/src/foo.svg")

    return

end


function callback(msg::Float64Msg, pub_obj::Publisher{BoolMsg})
    angle = msg.data
    println("angle received: ", angle)
    iterate_pomcp(angle)
    # pt_msg = Point(msg.x, msg.y, 0.0)
    publish(pub_obj, BoolMsg(true))
    println("Finished solving.")
    return
end



function main()

    global my_window
    my_window = Window()

    init_node("pomcp_node")
    pub = Publisher{BoolMsg}("solution", queue_size=10)
    sub = Subscriber{Float64Msg}("input_angle", callback, (pub,), queue_size=10)
    initialize_map()
    println("Map initialized, waiting for input!")

    # Send bool to game manager so that it gets ready to receive input
    # publish(pub, BoolMsg(true))
    # publish(pub, BoolMsg(true))

    spin()
end


if ! isinteractive()
    main()
end
