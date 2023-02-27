#!/usr/bin/env julia
# Gridworld Inference problem: Infer the goal and neighboring hyperplanes.
# Action is solution to a POMDP problem where incorrect set transitions are penalized

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
using MCTS

# For embed
using Infiltrator



# using RobotOS
# @rosimport std_msgs.msg.Header
# @rosimport geometry_msgs.msg: Point
# @rosimport std_msgs.msg: Float64, Bool
# rostypegen()
# using .geometry_msgs.msg
# using .std_msgs.msg


# ---------------- Experiment Parameters -----------------------
# Map of the environment
map_name = "2_10x10_squareobstacle_map"
# Save directory
save_dir = "/home/orianapeltzer/SA_data/"*map_name*"/"






include("hyperplane_inference_gridworld_mcts/convex_sets.jl")
map_dir = "maps/"*map_name*".jl"
include(map_dir)

# Grid world, map, pomdp and problem-specific functions
include("hyperplane_inference_gridworld_mcts/world.jl")
include("hyperplane_inference_gridworld_mcts/hyper_a_star.jl")
# Helper functions for problem specifics
include("hyperplane_inference_gridworld_mcts/utilities.jl")
# POMCP-specific functions to solve the POMDP
include("hyperplane_inference_gridworld_mcts/pomcp_functions.jl")
# To visualize
include("hyperplane_inference_gridworld_mcts/visualization.jl")

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

    # Retrieve the occupancy map from the .jl file
    grid_side = 10
    OBSTACLE_MAP = map_occupancy_grid()

    BII_gamma=4.0

    map_graph, K_map = initialize_grid_graph(OBSTACLE_MAP)

    # Load the hyperplane graph for the environment
    G = loadgraph("maps/"*map_name*".mg", MGFormat())

    println("Loaded hyperplane graph!")

    start_pos, true_goal_index, goal_options, M_pref = map_problem_setup()
    dist_matrix, dist_matrix_DP = get_dist_matrix(map_graph)

    # define POMDP
    mapworld_pomdp_setup = MapWorldSetup(obstacle_map = OBSTACLE_MAP,
                              grid_side = grid_side,
                              goal_options = goal_options,
                              n_possible_goals = length(goal_options),
                              map_graph = map_graph,
                              K_map = K_map,
                              dist_matrix = dist_matrix,
                              dist_matrix_DP = dist_matrix_DP,
                              BII_gamma = BII_gamma,
                              hyperplane_graph = G,
                              start_position = start_pos,
                              true_goal_index = true_goal_index,
                              true_preference = M_pref) # Other args are default

    curr_belstate = initial_belief_state(mapworld_pomdp_setup)


    # Sample 1000 possible states from the belief and put them in the pomdp
    start_states = [random_state(RNG, curr_belstate) for iter in 1:500]

    mapworld_pomdp = MapWorld(start_states = start_states,
                              start_state_index = 1,
                              obstacle_map = OBSTACLE_MAP,
                              grid_side = grid_side,
                              goal_options = goal_options,
                              n_possible_goals = length(goal_options),
                              map_graph = map_graph,
                              K_map = K_map,
                              dist_matrix = dist_matrix,
                              dist_matrix_DP = dist_matrix_DP,
                              BII_gamma = BII_gamma,
                              hyperplane_graph = G,
                              start_position = start_pos,
                              true_goal_index = true_goal_index,
                              true_preference = M_pref) # Other args are default


    true_state = true_initial_state(mapworld_pomdp_setup)

    # SHORTEST PATH DEBUGGING
    # start_vertex = 5
    # end_vertex = 11
    #
    # preference_constraint = NeighborConstraint(curr_belstate.neighbor_A,curr_belstate.neighbor_b,1,true)
    # path = Graphs.a_star(mapworld_pomdp.map_graph, start_vertex, end_vertex, preference_constraint, mapworld_pomdp.dist_matrix_DP)
    #
    # @show path

    # For visualization
    c = POMDPTools.render(mapworld_pomdp, true_state, curr_belstate)
    draw(SVGJS(save_dir*"foo.svg", 6inch, 6inch), c)
    # Ubuntu
    # loadurl(my_window, "file:///home/orianapeltzer/SA_data/"*map_name*"/foo.svg")
    # loadurl(my_window, "file:///home/orianapeltzer/catkin_ws/src/joystick_pomcp/src/foo.svg")
    # Windows
    # loadurl(my_window, "c:/opt/ros/melodic/x64/catkin_ws/src/hyperplane_inference/src/hyperplane_inference/foo.svg")

    return
end

# Main loop - is called each time an observation is received
function iterate_pomcp(angle::Float64, solver, RNG::AbstractRNG; display_window=true, iteration=0)


    global mapworld_pomdp
    global curr_belstate
    global true_state
    global my_window

    # are we at the right goal or not?
    curr_pos = true_state.position

    goal = mapworld_pomdp.goal_options[true_state.goal_index]

    # If we are at the goal, return
    if curr_pos == goal
        println("Already made it to goal!!")
        return
    end

    # Turn the input angle into an observation
    println("Angle observed: ", angle)
    heading = angle_to_heading(angle)
    println("Resulting heading: ", heading)
    o = HumanInputObservation(heading,true,true)

    # println("Created human observation")

    # Update the belief state. The action is irrelevant.
    curr_belstate = update_goal_belief(mapworld_pomdp, curr_belstate, o)

    println("Belief after goal update")
    @show curr_belstate

    println("Finished updating goal belief")


    # Take compliant action
    # a = heading

    action_list = actions(mapworld_pomdp)
    ActionValues = fill(0.0,size(action_list))

    policy = solve(solver, mapworld_pomdp)
    println("MCTS Iteration solved!")

    # Look through all possible start states
    for (s_idx,sampled_state) in enumerate(mapworld_pomdp.start_states)
        # mapworld_pomdp_new = MapWorld(start_states = mapworld_pomdp.start_states,
        #                           start_state_index = s_idx,
        #                           obstacle_map = mapworld_pomdp.obstacle_map,
        #                           grid_side = mapworld_pomdp.grid_side,
        #                           goal_options = mapworld_pomdp.goal_options,
        #                           n_possible_goals = mapworld_pomdp.n_possible_goals,
        #                           map_graph = mapworld_pomdp.map_graph,
        #                           K_map = mapworld_pomdp.K_map,
        #                           dist_matrix = mapworld_pomdp.dist_matrix,
        #                           dist_matrix_DP = mapworld_pomdp.dist_matrix_DP,
        #                           BII_gamma = mapworld_pomdp.BII_gamma,
        #                           hyperplane_graph = mapworld_pomdp.hyperplane_graph,
        #                           start_position = mapworld_pomdp.start_position,
        #                           true_goal_index = mapworld_pomdp.true_goal_index,
        #                           true_preference = mapworld_pomdp.true_preference) # Other args are default
        #
        # # Solve POMDP!
        # policy = solve(solver, mapworld_pomdp_new)
        # println("MCTS Iteration solved!")

        # @infiltrate

        action,info = action_info(policy,sampled_state)

        # println("Got action info!")

        for (a_idx,a) in enumerate(action_list)
            v = POMDPs.value(policy,sampled_state,a)
            ActionValues[a_idx] += v
        end
        # println("Found values for each action!")
    end


    # # We don't know what the true state is: do one step of bellman equation!
    #
    # for a in action_list
    #     v = 0
    #     for sampled_state in mapworld_pomdp.start_states
    #         # sampled_state = random_state(RNG,curr_belstate)
    #         v +=  POMDPs.value(policy, sampled_state,a)
    #     end
    #     ActionValues = vcat(ActionValues, v)
    # end
    a = action_list[findmax(ActionValues)[2]]

    # a = action(policy, initialstate(mapworld_pomdp));#, tree_in_info=true);
    println("Found optimal action for current belief!")
    @show a

    # Generate new state (the observation is irrelevant)
    true_state, r = gen(mapworld_pomdp, true_state, a, RNG)

    curr_belstate = update_pose_belief(mapworld_pomdp, curr_belstate, a)

    println("done updating pose belief")

    # Update true state
    # true_state = GridState(curr_belstate.position, true_state.done,
    #                        true_state.goal_index, curr_belstate.neighbor_A,
    #                        curr_belstate.neighbor_b, true_state.intention_index)
    println("Reward:")
    @show r
    println("New true state:")
    @show true_state

    println("New belief state:")
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
        # true_state.done = true
    end

    c = POMDPTools.render(mapworld_pomdp, true_state, curr_belstate)

    draw(SVGJS(save_dir*string(iteration)*".svg", 6inch, 6inch), c)
    # Ubuntu
    if display_window
        loadurl(my_window, "file:///home/peltzer/SA_data/"*map_name*"/foo.svg")
    end
    # loadurl(my_window, "file:///home/peltzer/catkin_ws/src/joystick_pomcp/src/foo.svg")
    # Windows
    # loadurl(my_window, "c:/opt/ros/melodic/x64/catkin_ws/src/hyperplane_inference/src/hyperplane_inference/foo.svg")

    return true_state.position == goal

end



function simulate_mcts()
    """Simulate an instance of the pomdp problem"""

    global mapworld_pomdp
    global curr_belstate
    global true_state
    global my_window

    # my_window = Window()

    initialize_map()
    println("Map initialized!")

    RNG = MersenneTwister(1234)
    # solver = POMCPSolver(rng=RNG, max_depth=30, tree_queries = 100000)

    iteration = 1
    max_iterations = 31
    problem_terminated = false

    while (~problem_terminated && iteration < max_iterations)

        println("")
        println("")
        println("Starting iteration "*string(iteration))


        solver = MCTSSolver(rng=RNG, depth=31-iteration, exploration_constant=5.0)#+100*iteration)

        @show true_state.position
        @show curr_belstate.position

        o = sample_human_action(mapworld_pomdp, true_state, RNG)

        println("Sampled new observation: ")
        @show o

        problem_terminated = iterate_pomcp(o, solver, RNG, display_window=false,
                                           iteration=iteration)

        println("Finished iteration "*string(iteration))
        iteration += 1

    end

    if problem_terminated
        println("Success!")
    end

end


# # ROS functions
# function callback(msg::Float64Msg, pub_obj::Publisher{BoolMsg})
#     angle = msg.data
#     println("angle received: ", angle)
#     iterate_pomcp(angle)
#     # pt_msg = Point(msg.x, msg.y, 0.0)
#     publish(pub_obj, BoolMsg(true))
#     println("Finished solving.")
#     return
# end
#
#
# function main_ros()
#
#     global my_window
#     my_window = Window()
#
#     init_node("pomcp_node")
#     pub = Publisher{BoolMsg}("solution", queue_size=10)
#     sub = Subscriber{Float64Msg}("input_angle", callback, (pub,), queue_size=10)
#     initialize_map()
#     println("Map initialized, waiting for input!")
#
#     # Send bool to game manager so that it gets ready to receive input
#     # publish(pub, BoolMsg(true))
#     # publish(pub, BoolMsg(true))
#
#     spin()
# end

simulate_mcts()


# if ! isinteractive()
#     simulate_mcts()
# end
