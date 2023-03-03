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
using JLD2
using LinearAlgebra
using CSV
using DataFrames

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
# map_name = "2_10x10_squareobstacle_map"
map_name = "4_10x10_office_map"
# Save directory
save_dir = "/home/orianapeltzer/SA_data/"*map_name*"/"






include("hyperplane_inference_gridworld_pomdp/convex_sets.jl")
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
# Performance metrics
include("hyperplane_inference_gridworld_pomdp/performance_metrics.jl")

# All the global variables (sorry!)
global mapworld_pomdp
global mapworld_pomdp_goal
# global solver
global policy
global curr_belstate
global curr_belstate_goal
global true_state
global my_window

# Create map and problem
function initialize_map(trial_number)

    global mapworld_pomdp
    global mapworld_pomdp_goal
    global curr_belstate
    global curr_belstate_goal
    global true_state
    global my_window

    RNG = MersenneTwister(1234)


    # This can all be replaced by mapworld_pomdp = load(pomdp problem jld) -- #
    # Retrieve the occupancy map from the .jl file
    grid_side = 10
    OBSTACLE_MAP = map_occupancy_grid()

    BII_gamma=1.5

    map_graph, K_map = initialize_grid_graph(OBSTACLE_MAP)

    # Load the hyperplane graph for the environment
    G = loadgraph("maps/"*map_name*".mg", MGFormat())

    println("Loaded hyperplane graph!")

    # This generates start, goal, preferences
    # start_pos, true_goal_index, goal_options, M_pref = sample_problem_setup()
    # if trial_number==1
    #     start_pos, true_goal_index, goal_options, M_pref = map_problem_setup()
    # elseif trial_number==2
    #     start_pos, true_goal_index, goal_options, M_pref = map_problem_setup()
    #     start_pos = GridPosition(6,1)
    # else
    #     start_pos, true_goal_index, goal_options, M_pref = sample_problem_setup()
    # end
    start_pos, true_goal_index, goal_options, M_pref = sample_problem_setup()

    dist_matrix, dist_matrix_DP = get_dist_matrix(map_graph)

    # define POMDP
    mapworld_pomdp = MapWorld(obstacle_map = OBSTACLE_MAP,
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

    mapworld_pomdp_goal = MapWorld(obstacle_map = OBSTACLE_MAP,
                            grid_side = grid_side,
                            discount_factor = 0.96,
                            incorrect_transition_penalty = 0.0, #-10
                            correct_transition_reward= 0.0, #10.5
                            reward = 1000.0,
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
    # ----------------------------------------------------------------------- #

    # Save sampled problem in jld format
    trial_dir = save_dir*"trial_"*string(trial_number)*"/"
    # @save trial_dir*"pomdp.jld2" mapworld_pomdp

    # Write out problem specifics in info.txt
    # io = open(trial_dir*"info.txt", "w")
    # write(io, "Trial "*string(trial_number)*"\n\n")
    # write(io, "goal options:\n")
    # write(io, string(mapworld_pomdp.goal_options)*"\n\n")
    # write(io, "correct goal index:"*"\n")
    # write(io, string(mapworld_pomdp.true_goal_index)*"\n\n")
    # write(io, "start position:\n")
    # write(io, string(start_pos)*"\n")
    # close(io)


    curr_belstate = initial_belief_state(mapworld_pomdp)
    curr_belstate_goal = initial_belief_state_goal(mapworld_pomdp)
    true_state = initial_state(mapworld_pomdp)

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
    # draw(SVGJS(save_dir*"setup_"*string(trial_number)*".svg", 6inch, 6inch), c)
    # Ubuntu
    # loadurl(my_window, "file:///home/orianapeltzer/SA_data/"*map_name*"/foo.svg")
    # loadurl(my_window, "file:///home/orianapeltzer/catkin_ws/src/joystick_pomcp/src/foo.svg")
    # Windows
    # loadurl(my_window, "c:/opt/ros/melodic/x64/catkin_ws/src/hyperplane_inference/src/hyperplane_inference/foo.svg")

    return
end

# Main loop - is called each time an observation is received
function iterate_pomcp(angle::Float64, solver, RNG::AbstractRNG; display_window=true, iteration=0,trial_number=1,
                       allowed_observation=true,method="path_pref",new_observation_blended=true)


    global mapworld_pomdp
    global mapworld_pomdp_goal
    global curr_belstate
    global curr_belstate_goal
    global true_state
    global my_window

    trial_dir = save_dir*"trial_"*string(trial_number)*"/"

    # are we at the right goal or not?
    curr_pos = true_state.position

    goal = mapworld_pomdp.goal_options[true_state.goal_index]

    # If we are at the goal, return
    if curr_pos == goal
        println("Already made it to goal!!")
        return
    end


    # 1. Update goal belief ------------------------------------------------

    if method != "compliant"

        if allowed_observation
            println("Angle observed: ", angle)
            heading = angle_to_heading(angle)
            println("Resulting heading: ", heading)
            o = HumanInputObservation(heading,true,true)
        else
            o = HumanInputObservation('E',false,true)
        end

        if method=="path_pref" || method=="blended"
            curr_belstate, update_time = @timed update_goal_belief(mapworld_pomdp, curr_belstate, o)
        elseif method=="goal_only"
            curr_belstate_goal, update_time = @timed update_goal_belief(mapworld_pomdp, curr_belstate_goal, o)
        end


    elseif allowed_observation # compliant + new observation

    # if allowed_observation
        # Turn the input angle into an observation
        println("Angle observed: ", angle)
        heading = angle_to_heading(angle)
        println("Resulting heading: ", heading)
        o = HumanInputObservation(heading,true,true)

        update_time = 0.0

    else
        o = HumanInputObservation('E',false,true)
        update_time = 0.0
        println("Observation not allowed: o should not be used.")
    end

    println("Finished updating goal belief") # ------------------------------- #




    #2. Solve problem and pick action --------------------------------------- #

    # (a, info), solve_time_2 = @timed action_info(policy, curr_belstate);#, tree_in_info=true);

    if method=="path_pref" || method=="blended"
        # Solve POMDP!
        policy, solve_time_1 = @timed solve(solver, mapworld_pomdp)
        (a, info), solve_time_2 = @timed action_info(policy, curr_belstate);#, tree_in_info=true);
        # entropy_goal = compute_entropy_goal_distribution(mapworld_pomdp, curr_belstate)
        entropy_distribution = compute_entropy_goal_currentpref(mapworld_pomdp, curr_belstate)
        # Arbitration for policy blending
        if method=="blended" && new_observation_blended && entropy_distribution > 1.6
            println("blended policy is using compliant action!")
            a = heading
        end
    elseif method=="goal_only"
        # Solve POMDP!
        policy, solve_time_1 = @timed solve(solver, mapworld_pomdp_goal)
        (a, info), solve_time_2 = @timed action_info(policy, curr_belstate_goal);#, tree_in_info=true);
    else # compliant
        println("Picking action for compliant policy")
        if allowed_observation
            println("allowed observation: action = "*string(heading))
            a = heading
        else
            println("not allowed observation: action = null")
            a = 'n'
        end
        solve_time_1 = 0.0
        solve_time_2 = 0.0
    end

    println("POMCP Iteration solved!")
    pomdp_solve_time = solve_time_1 + solve_time_2
    @show a

    # ----------------------------------------------------------------------- #

    # 3. Real world sim ----------------------------------------------------- #
    # Generate new state (the observation is irrelevant)
    true_state, r, _ = gen(mapworld_pomdp, true_state, a, RNG)
    # ----------------------------------------------------------------------- #


    # 4. Update pose belief ------------------------------------------------- #

    if method=="path_pref" || method=="blended"
        curr_belstate = update_pose_belief(mapworld_pomdp, curr_belstate, a)
    elseif method=="goal_only"
        curr_belstate_goal = update_pose_belief(mapworld_pomdp, curr_belstate_goal, a)
    end

    println("done updating pose belief")

    # Update true state
    # true_state = GridState(curr_belstate.position, true_state.done,
    #                        true_state.goal_index, curr_belstate.neighbor_A,
    #                        curr_belstate.neighbor_b, true_state.intention_index)
    println("Reward:")
    @show r
    println("New true state:")
    @show true_state

    # println("New belief state:")
    # @show curr_belstate

    # display(curr_belstate.belief_intention)
    # println("")
    #
    # println("Marginal preferences:")
    # display(sum(curr_belstate.belief_intention, dims=1))
    # println("")

    if true_state.position == goal
        println("--------------------------")
        println("   Made it to the goal!!  ")
        println("--------------------------")
        # true_state.done = true
    end

    global run_number
    # @save trial_dir*"/beliefs/run_"*string(run_number)*"_iter_"*string(iteration)*"_belief.jld2" curr_belstate
    #
    # if method=="path_pref" || method=="blended"
    #     c = POMDPTools.render(mapworld_pomdp, true_state, curr_belstate)
    #     draw(SVGJS(trial_dir*"visualizations/run_"*string(run_number)*"_method_"*method*"_iter_"*string(iteration)*".svg", 6inch, 6inch), c)
    # elseif method=="goal_only"
    #     c = POMDPTools.render(mapworld_pomdp, true_state, curr_belstate_goal)
    #     draw(SVGJS(trial_dir*"visualizations/run_"*string(run_number)*"_method_"*method*"_iter_"*string(iteration)*".svg", 6inch, 6inch), c)
    # else # compliant
    #     c = POMDPTools.render(mapworld_pomdp, true_state)
    #     draw(SVGJS(trial_dir*"visualizations/run_"*string(run_number)*"_method_"*method*"_iter_"*string(iteration)*".svg", 6inch, 6inch), c)
    # end

    # Ubuntu
    if display_window
        loadurl(my_window, "file:///home/peltzer/SA_data/"*map_name*"/foo.svg")
    end
    # loadurl(my_window, "file:///home/peltzer/catkin_ws/src/joystick_pomcp/src/foo.svg")
    # Windows
    # loadurl(my_window, "c:/opt/ros/melodic/x64/catkin_ws/src/hyperplane_inference/src/hyperplane_inference/foo.svg")

    return true_state.position == goal, a, pomdp_solve_time, update_time

end



function simulate_pomcp(trial_number::Int64,run_number::Int64; time_between_inputs=1,method="path_pref",max_depth=30,max_iterations=31)
    """Simulate an instance of the pomdp problem"""

    global mapworld_pomdp
    global curr_belstate
    global curr_belstate_goal
    global true_state
    global my_window

    trial_dir = save_dir*"trial_"*string(trial_number)*"/"

    # my_window = Window()

    RNG = MersenneTwister(1234)
    # solver = POMCPSolver(rng=RNG, max_depth=30, tree_queries = 100000)

    iteration = 1
    problem_terminated = false

    # Initialize Performance Metrics
    time_steps = [0]
    if method=="path_pref" || method=="blended"
        p_correct_goal = [compute_probability_correct_goal(mapworld_pomdp, curr_belstate)]
        goal_dist_entropy = [compute_entropy_goal_distribution(mapworld_pomdp, curr_belstate)]
        intention_dist_entropy = [compute_entropy_goal_currentpref(mapworld_pomdp, curr_belstate)]
    elseif method=="goal_only"
        p_correct_goal = [compute_probability_correct_goal(mapworld_pomdp, curr_belstate_goal)]
        goal_dist_entropy = [compute_entropy_goal_distribution(mapworld_pomdp, curr_belstate_goal)]
        intention_dist_entropy = [0.0]
    else
        p_correct_goal = [0.0]
        goal_dist_entropy = [0.0]
        intention_dist_entropy = [0.0]
    end

    actions = ['X'] # invalid action
    x_positions_on_map = [true_state.position[1]]
    y_positions_on_map = [true_state.position[2]]

    # Success indicators
    goal_success = false
    no_preference_violations = true
    num_violations = 0
    pomdp_solve_time = -1.0
    inference_time = -1.0

    # For evaluating path cost
    current_position = deepcopy(true_state.position)
    path_cost = 0

    previous_obs = 0.0

    while (~problem_terminated && iteration < max_iterations)

        println("")
        println("")
        println("Starting iteration "*string(iteration))


        solver = POMCPSolver(rng=RNG, max_depth=max_depth-iteration+1, tree_queries = 5000)#+100*iteration)

        @show true_state.position
        @show curr_belstate.position

        new_observation_blended = false

        new_obs_time = (((iteration-1) % time_between_inputs)==0)

        if method != "compliant"
            if new_obs_time
                # Sample observation
                o = sample_human_action(mapworld_pomdp, true_state, RNG)
                println("Sampled new observation: ")
                @show o
                allowed_observation=true
                new_observation_blended=(((iteration-1) % time_between_inputs)==0)
            else
                println("No observation allowed!")
                o = previous_obs
                allowed_observation=false
            end
        else
            if new_obs_time
                # Sample observation
                o = sample_human_action(mapworld_pomdp, true_state, RNG)
                println("Sampled new observation: ")
                @show o
                # @infiltrate
                allowed_observation=true
                previous_obs=o
            elseif ((iteration-1) % time_between_inputs) <= 2
                println("Using previous observation:")
                o = previous_obs
                @show o
                allowed_observation=true
            else
                println("No observation allowed!")
                o = previous_obs
                allowed_observation=false
            end
        end



        problem_terminated, a, solve_time, update_time = iterate_pomcp(o, solver, RNG, display_window=false,
                                           iteration=iteration,trial_number=trial_number,
                                           allowed_observation=allowed_observation,method=method,
                                           new_observation_blended=new_observation_blended)

        @show curr_belstate
        println("Entropy of belief: "*string(compute_entropy_goal_currentpref(mapworld_pomdp, curr_belstate)))


        # First iteration only metrics
        if iteration==1
            pomdp_solve_time = solve_time
            inference_time = update_time
        end


        # Iteration-specific performance metrics ---
        # Path cost
        if current_position==true_state.position #&& action != 'n'
            path_cost += 1 # Banged into a wall
        else
            path_cost += norm(true_state.position - current_position)
        end

        # Metrics that vary in time
        time_steps = vcat(time_steps, iteration)
        if method=="path_pref" || method=="blended"
            p_correct_goal = vcat(p_correct_goal, compute_probability_correct_goal(mapworld_pomdp, curr_belstate))
            goal_dist_entropy = vcat(goal_dist_entropy, compute_entropy_goal_distribution(mapworld_pomdp, curr_belstate))
            intention_dist_entropy = vcat(intention_dist_entropy, compute_entropy_goal_currentpref(mapworld_pomdp, curr_belstate))
            println("Entropy of goal distribution")
            @show goal_dist_entropy[length(goal_dist_entropy)]
            println("Probability of correct goal")
            @show p_correct_goal[length(p_correct_goal)]

        elseif method=="goal_only"
            p_correct_goal = vcat(p_correct_goal, compute_probability_correct_goal(mapworld_pomdp, curr_belstate_goal))
            goal_dist_entropy = vcat(goal_dist_entropy, compute_entropy_goal_distribution(mapworld_pomdp, curr_belstate_goal))
            intention_dist_entropy = vcat(intention_dist_entropy, 0.0)

            println("Entropy of goal distribution")
            @show goal_dist_entropy[length(goal_dist_entropy)]
            println("Probability of correct goal")
            @show p_correct_goal[length(p_correct_goal)]
        else
            p_correct_goal = vcat(p_correct_goal, 0.0)
            goal_dist_entropy = vcat(goal_dist_entropy, 0.0)
            intention_dist_entropy = vcat(intention_dist_entropy, 0.0)
        end
        actions = vcat(actions, a)
        x_positions_on_map = vcat(x_positions_on_map, true_state.position[1])
        y_positions_on_map = vcat(y_positions_on_map, true_state.position[2])

        # @show current_position
        # @show a
        # @infiltrate
        violation = violated_preferences(mapworld_pomdp, current_position, a)
        # @show violation
        # Success indicators
        if violated_preferences(mapworld_pomdp, current_position, a)
            num_violations += 1
            if num_violations >= 1
                no_preference_violations = false
            end
        end

        println("Finished iteration "*string(iteration))
        current_position = deepcopy(true_state.position)
        iteration += 1

    end

    # End-of-run metrics
    if problem_terminated
        println("Success!")
        goal_success = true
    end

    # Create DataFrame
    headers = ["trial_number","run_id",
               "method",
               "time_between_inputs",
               "goal_success","pref_success",
               "path_cost",
               "time_steps","p_correct_goal","goal_dist_entropy",
               "intention_dist_entropy",
               "actions","pos_x","pos_y",
               "pomdp_solve_time", "inference_time"]
    trial_data = (trial_number,run_number,
                  time_between_inputs,
                  method,
                  goal_success, no_preference_violations,
                  path_cost,
                  time_steps, p_correct_goal, goal_dist_entropy,
                  intention_dist_entropy,
                  actions, x_positions_on_map, y_positions_on_map,
                  pomdp_solve_time, inference_time)

# DataFrame(A=Int[],B=Array{Int64}[],C=String[])
    df = DataFrame(trial_number=Int64[],
                   run_id = Int64[],
                   time_between_inputs = Int64[],
                   method=typeof(method)[],
                   goal_success = Bool[],
                   pref_success=Bool[],
                   path_cost=Float64[],
                   time_steps=typeof(time_steps)[],
                   p_correct_goal=typeof(p_correct_goal)[],
                   goal_dist_entropy=typeof(goal_dist_entropy)[],
                   intention_dist_entropy=typeof(intention_dist_entropy)[],
                   actions = typeof(actions)[],
                   pos_x = typeof(x_positions_on_map)[],
                   pos_y = typeof(y_positions_on_map)[],
                   pomdp_solve_time = Float64[],
                   inference_time = Float64[])

    push!(df,trial_data)

    @show df

    # Create csv with header and first row
    if run_number==1
        CSV.write(save_dir*"officemap_data_computation.csv",df)
    else
        CSV.write(save_dir*"officemap_data_computation.csv",df,append=true)
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


if ! isinteractive()
# if true

    global mapworld_pomdp
    global curr_belstate
    global curr_belstate_goal
    global true_state
    global run_number

    num_random_trials = 100
    # method_list=["path_pref","goal_only","compliant","blended"]
    method_list = ["path_pref","goal_only"]
    # times_between_inputs = [1,5,10,20,30]
    times_between_inputs = [1]
    sims_per_setup = 1


    max_depth=30 # lookahead for the robot
    max_iterations=2 # for stopping sim early



    run_number = 1

    for trial_number=1:num_random_trials

        # Generate problem instance (trial)
        initialize_map(trial_number)

        println("Map initialized!")

        # Loop through time between inputs (runs with different deltaT parameter)
        for method in method_list
            for deltat in times_between_inputs
                for sim_num in 1:sims_per_setup
                    global mapworld_pomdp
                    global curr_belstate
                    global curr_belstate_goal
                    global true_state
                    global run_number
                    curr_belstate = initial_belief_state(mapworld_pomdp)
                    curr_belstate_goal = initial_belief_state_goal(mapworld_pomdp)
                    true_state = initial_state(mapworld_pomdp)
                    simulate_pomcp(trial_number,run_number;time_between_inputs=deltat,method=method,max_depth=max_depth,max_iterations=max_iterations)

                    run_number += 1
                end
            end
        end
    end
end
