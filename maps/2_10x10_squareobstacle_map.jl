"""Returns an occupancy grid of the environment"""
function map_occupancy_grid()
    # Generate a true 10x10 false/true map and ensure start position is empty (false)
    grid_side = 10
    OBSTACLE_MAP = fill(false, grid_side, grid_side)
    OBSTACLE_MAP[5,5] = true
    OBSTACLE_MAP[5,6] = true
    OBSTACLE_MAP[6,5] = true
    OBSTACLE_MAP[6,6] = true
    # println("OBSTACLE MAP: ", OBSTACLE_MAP)
    # OBSTACLE_MAP[1, 1] = false
    # OBSTACLE_MAP[1, 2] = false # Giving robot a way out hehe
    return OBSTACLE_MAP
end

function map_hyperplane_graph()
    # Obstacle: square containing [5,5], [5,6], [6,5] and [6,6]
    obstacle_A = [1 0
                  -1 0
                  0 1
                  0 -1]
    obstacle_b = [6.0, -4.0, 6.0, -4.0]
    obstacle_point = [6.0,6.0] # point in the middle of the obstacle
    obs = obstacle(obstacle_A, obstacle_b, obstacle_point)

    # Contour: x and y bounded by [0,10]
    contour_A = [1 0
                 -1 0
                 0 1
                 0 -1]
    contour_b = [10.0, 0.0, 10.0, 0.0]
    cnt = contour(contour_A, contour_b)

    # point where the robot currently is
    robot_location = [1.0,1.0]

    G = create_hyperplane_arrangement(robot_location, [obs], cnt)

    return G
end

function map_initial_pos()
    return 1,1
end

function sample_problem_setup()
    """M_pref: Array size n where n is the number of vertices in the graph
    (number of obstacle-free polytopes). M_pref[i] is the index of the preferred
    next vertex at node i, with L(i)+1 meaning that there is no preference
    (occurs when goal is inside the region). Note that the index of the
    preferred next vertex should be mapped to the neighbor in the graph."""

    OBSTACLE_MAP = map_occupancy_grid()

    # Sample random start and goal

    all_start_options = [GridPosition(x, y) for x=1:10 for y in 1:10 if (OBSTACLE_MAP[x,y] == false)]

    # start_pos = GridPosition(1,1)
    start_pos = sample(all_start_options)


    all_goal_options = [GridPosition(x, y) for x=1:10 for y in 1:10 if ((OBSTACLE_MAP[x,y] == false) && (GridPosition(x,y)!=start_pos))]
    # goal_options = [GridPosition(x, 8) for x=1:10 if OBSTACLE_MAP[x,8] == false]

    goal_options = sample(all_goal_options,10,replace=false)


    # true_goal = GridPosition(1, 8)
    true_goal_index = 1

    M_pref = fill(0,8,1)
    M_pref[1] = 2 # at node #1 the preferred next node is #2
    M_pref[2] = 4
    M_pref[3] = 1
    M_pref[4] = 6
    M_pref[5] = 3 # -1 means the goal is in the region so there's no pref
    M_pref[6] = 8
    M_pref[7] = 5
    M_pref[8] = 7
    # M_pref[1,2] = 1 # at node #1 the preferred next node is #2
    # M_pref[2,4] = 1
    # M_pref[3,1] = 1
    # M_pref[4,6] = 1
    # M_pref[5,9] = 1 # Goal region
    # M_pref[6,8] = 1
    # M_pref[7,5] = 1
    # M_pref[8,7] = 1
    return start_pos, true_goal_index, goal_options, M_pref
end

function map_problem_setup()
    """M_pref: Array size n where n is the number of vertices in the graph
    (number of obstacle-free polytopes). M_pref[i] is the index of the preferred
    next vertex at node i, with L(i)+1 meaning that there is no preference
    (occurs when goal is inside the region). Note that the index of the
    preferred next vertex should be mapped to the neighbor in the graph."""

    OBSTACLE_MAP = map_occupancy_grid()

    # Sample random start and goal

    # all_start_options = [GridPosition(x, y) for x=1:10 for y in 1:10 if (OBSTACLE_MAP[x,8] == false)]

    start_pos = GridPosition(1,1)
    # start_pos = sample(all_start_options)


    # all_goal_options = [GridPosition(x, y) for x=1:10 for y in 1:10 if ((OBSTACLE_MAP[x,8] == false) && (GridPosition(x,y)!=start_pos))]
    goal_options = [GridPosition(x, 8) for x=1:10 if OBSTACLE_MAP[x,8] == false]

    # goal_options = sample(all_goal_options,10,replace=false)


    # true_goal = GridPosition(1, 8)
    true_goal_index = 1

    M_pref = fill(0,8,1)
    M_pref[1] = 2 # at node #1 the preferred next node is #2
    M_pref[2] = 4
    M_pref[3] = 1
    M_pref[4] = 6
    M_pref[5] = -1 # -1 means the goal is in the region so there's no pref
    M_pref[6] = 8
    M_pref[7] = 5
    M_pref[8] = 7
    # M_pref[1,2] = 1 # at node #1 the preferred next node is #2
    # M_pref[2,4] = 1
    # M_pref[3,1] = 1
    # M_pref[4,6] = 1
    # M_pref[5,9] = 1 # Goal region
    # M_pref[6,8] = 1
    # M_pref[7,5] = 1
    # M_pref[8,7] = 1
    return start_pos, true_goal_index, goal_options, M_pref
end

# function write_hyperplane_graph_to_file(G::MetaGraph, graph_name::String)
#     savegraph(graph_name, G)
# end
