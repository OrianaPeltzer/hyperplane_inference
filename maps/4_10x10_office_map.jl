"""Returns an occupancy grid of the environment"""
function map_occupancy_grid()
    # Generate a true 10x10 false/true map and ensure start position is empty (false)
    grid_side = 10
    OBSTACLE_MAP = fill(false, grid_side, grid_side)
    desk_lowerleft_corners = [[1,3],
                              [1,6],
                              [1,10],
                              [4,3],
                              [4,5],
                              [4,10],
                              [5,7],
                              [6,2],
                              [8,2],
                              [8,7],
                              [10,5],
                              [7,4]
                              ]
    desk_shapes = [(1,1),
                   (1,1),
                   (1,1),
                   (1,1),
                   (2,2),
                   (6,1),
                   (2,1),
                   (1,1),
                   (1,2),
                   (1,1),
                   (1,3),
                   (2,1)
                  ]

    for (i,(x0,y0)) in enumerate(desk_lowerleft_corners)
        desk_shape = desk_shapes[i]
        for x_offset=0:desk_shape[1]-1
            for y_offset=0:desk_shape[2]-1
                OBSTACLE_MAP[x0+x_offset,y0+y_offset] = true
            end
        end
    end
    display(OBSTACLE_MAP)
    # println("OBSTACLE MAP: ", OBSTACLE_MAP)
    # OBSTACLE_MAP[1, 1] = false
    # OBSTACLE_MAP[1, 2] = false # Giving robot a way out hehe
    return OBSTACLE_MAP
end


function desk_to_obstacle(lowerleftcorner,size)
    desk_A = [1 0
              -1 0
              0 1
              0 -1]
    desk_b = [lowerleftcorner[1]+size[1]-1,
              -(lowerleftcorner[1]-1),
              lowerleftcorner[2]+size[2]-1,
              -(lowerleftcorner[2]-1)]
    return obstacle(desk_A,desk_b,lowerleftcorner)
end

function map_hyperplane_graph()

    obs = Array{obstacle}[]
    desk_lowerleft_corners = [[1,3],
                              [1,6],
                              [1,10],
                              [4,3],
                              [4,5],
                              [4,10],
                              [5,7],
                              [6,2],
                              [8,2],
                              [8,7],
                              [10,5],
                              [7,4]
                              ]
    desk_shapes = [(1,1),
                   (1,1),
                   (1,1),
                   (1,1),
                   (2,2),
                   (6,1),
                   (2,1),
                   (1,1),
                   (1,2),
                   (1,1),
                   (1,3),
                   (2,1)
                  ]
    for (i,desk) in enumerate(desk_lowerleft_corners)
        desk_shape = desk_shapes[i]
        obs = vcat(obs,desk_to_obstacle(desk,desk_shape))
    end

    # Contour: x and y bounded by [0,10]
    contour_A = [1 0
                 -1 0
                 0 1
                 0 -1]
    contour_b = [10.0, 0.0, 10.0, 0.0]
    cnt = contour(contour_A, contour_b)

    # point where the robot currently is
    robot_location = [1.0,1.0]

    # @show obs
    # @show obs[1].b

    G = create_hyperplane_arrangement(robot_location, obs, cnt, map_occupancy_grid())

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

    # Goal region specific to this map
    all_goal_options = [GridPosition(x, y) for x=2:10 for y in 8:9 if ((OBSTACLE_MAP[x,y] == false) && (GridPosition(x,y)!=start_pos))]
    # goal_options = [GridPosition(x, 8) for x=1:10 if OBSTACLE_MAP[x,8] == false]

    goal_options = sample(all_goal_options,10,replace=false)


    # true_goal = GridPosition(1, 8)
    true_goal_index = 1

    M_pref = fill(0,56,1)
    M_pref[1] = 2 # at node #1 the preferred next node is #2
    M_pref[2] = 4
    M_pref[3] = 6
    M_pref[4] = 7
    M_pref[5] = 2
    M_pref[6] = 2
    M_pref[7] = 12
    M_pref[8] = 5
    M_pref[9] = 6
    M_pref[10] = 7
    M_pref[11] = 7 # at node #1 the preferred next node is #2
    M_pref[12] = 17
    M_pref[13] = 8
    M_pref[14] = 9
    M_pref[15] = 12
    M_pref[16] = 13
    M_pref[17] = 21
    M_pref[18] = 13
    M_pref[19] = 14
    M_pref[20] = 18
    M_pref[21] = 27 # at node #1 the preferred next node is #2
    M_pref[22] = 28
    M_pref[23] = 19
    M_pref[24] = 31
    M_pref[25] = 21
    M_pref[26] = 33
    M_pref[27] = 33
    M_pref[28] = 23
    M_pref[29] = 23
    M_pref[30] = 24
    M_pref[31] = 37 # at node #1 the preferred next node is #2
    M_pref[32] = 27
    M_pref[33] = 38
    M_pref[34] = 27
    M_pref[35] = 29
    M_pref[36] = 31
    M_pref[37] = 43
    M_pref[38] = 44
    M_pref[39] = 35
    M_pref[40] = 35
    M_pref[41] = 36 # at node #1 the preferred next node is #2
    M_pref[42] = 37
    M_pref[43] = 49
    M_pref[44] = 47
    M_pref[45] = 39
    M_pref[46] = 40
    M_pref[47] = 51
    M_pref[48] = 43
    M_pref[49] = 46
    M_pref[50] = 45
    M_pref[51] = 54 # at node #1 the preferred next node is #2
    M_pref[52] = 48
    M_pref[53] = 50
    M_pref[54] = 52
    M_pref[55] = 54
    M_pref[56] = 55
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

    start_pos = GridPosition(6,3)
    # start_pos = sample(all_start_options)


    all_goal_options = [GridPosition(x, y) for x=1:10 for y in 1:10 if ((OBSTACLE_MAP[x,8] == false) && (GridPosition(x,y)!=start_pos))]
    # goal_options = [GridPosition(x, 8) for x=1:10 if OBSTACLE_MAP[x,8] == false]

    # goal_options = sample(all_goal_options,10,replace=false)


    # true_goal = GridPosition(1, 8)
    true_goal_index = 8

    M_pref = fill(0,56,1)
    M_pref[1] = 2 # at node #1 the preferred next node is #2
    M_pref[2] = 4
    M_pref[3] = 6
    M_pref[4] = 7
    M_pref[5] = 2
    M_pref[6] = 2
    M_pref[7] = 12
    M_pref[8] = 5
    M_pref[9] = 6
    M_pref[10] = 7
    M_pref[11] = 7 # at node #1 the preferred next node is #2
    M_pref[12] = 17
    M_pref[13] = 8
    M_pref[14] = 9
    M_pref[15] = 12
    M_pref[16] = 13
    M_pref[17] = 21
    M_pref[18] = 13
    M_pref[19] = 14
    M_pref[20] = 18
    M_pref[21] = 27 # at node #1 the preferred next node is #2
    M_pref[22] = 28
    M_pref[23] = 19
    M_pref[24] = 31
    M_pref[25] = 21
    M_pref[26] = 33
    M_pref[27] = 33
    M_pref[28] = 23
    M_pref[29] = 23
    M_pref[30] = 24
    M_pref[31] = 37 # at node #1 the preferred next node is #2
    M_pref[32] = 27
    M_pref[33] = 38
    M_pref[34] = 27
    M_pref[35] = 29
    M_pref[36] = 31
    M_pref[37] = 43
    M_pref[38] = 44
    M_pref[39] = 35
    M_pref[40] = 35
    M_pref[41] = 36 # at node #1 the preferred next node is #2
    M_pref[42] = 37
    M_pref[43] = 49
    M_pref[44] = 47
    M_pref[45] = 39
    M_pref[46] = 40
    M_pref[47] = 51
    M_pref[48] = 43
    M_pref[49] = 46
    M_pref[50] = 45
    M_pref[51] = 54 # at node #1 the preferred next node is #2
    M_pref[52] = 48
    M_pref[53] = 50
    M_pref[54] = 52
    M_pref[55] = 54
    M_pref[56] = 55

    return start_pos, true_goal_index, goal_options, M_pref
end

# function write_hyperplane_graph_to_file(G::MetaGraph, graph_name::String)
#     savegraph(graph_name, G)
# end
