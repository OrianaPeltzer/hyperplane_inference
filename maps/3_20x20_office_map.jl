"""Returns an occupancy grid of the environment"""
function map_occupancy_grid()
    # Generate a true 10x10 false/true map and ensure start position is empty (false)
    grid_side = 20
    OBSTACLE_MAP = fill(false, grid_side, grid_side)

    # Desks
    # desk_lowerleft_corners = [[3,3],
    #                           [3,6],
    #                           [3,9],
    #                           [3,12],
    #                           [3,15],
    #                           [9,3],
    #                           [9,6],
    #                           [9,9],
    #                           [9,12],
    #                           [9,15],
    #                           [15,3],
    #                           [15,6],
    #                           [15,9],
    #                           [15,12]]
    # desk_shape = (4,2)

    desk_lowerleft_corners = [[3,4],
                              [17,6],
                              [6,8],
                              [12,8],
                              [2,9],
                              [8,12],
                              [17,12],
                              [3,16]
                              ]
    desk_shapes = [(16,2),
                   (2,4),
                   (6,2),
                   (2,6),
                   (3,4),
                   (4,2),
                   (2,4),
                   (16,2)
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
    # OBSTACLE_MAP[5,5] = true
    # OBSTACLE_MAP[5,6] = true
    # OBSTACLE_MAP[6,5] = true
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
    return obstacle(desk_A,desk_b,lowerleftcorner+[1,1])
end

function map_hyperplane_graph()
    # Obstacle: square containing [5,5], [5,6], [6,5] and [6,6]

    obs = Array{obstacle}[]
    desk_lowerleft_corners = [[3,4],
                              [17,6],
                              [6,8],
                              [12,8],
                              [2,9],
                              [8,12],
                              [17,12],
                              [3,16]
                              ]
    desk_shapes = [(16,2),
                   (2,4),
                   (6,2),
                   (2,6),
                   (3,4),
                   (4,2),
                   (2,4),
                   (16,2)
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
    contour_b = [20.0, 0.0, 20.0, 0.0]
    cnt = contour(contour_A, contour_b)

    # point where the robot currently is
    robot_location = [1.0,1.0]

    G = create_hyperplane_arrangement(robot_location, obs, cnt, map_occupancy_grid())

    return G
end

function map_initial_pos()
    return 6,2
end


function sample_problem_setup()
    """M_pref: Array size n where n is the number of vertices in the graph
    (number of obstacle-free polytopes). M_pref[i] is the index of the preferred
    next vertex at node i, with L(i)+1 meaning that there is no preference
    (occurs when goal is inside the region). Note that the index of the
    preferred next vertex should be mapped to the neighbor in the graph."""

    OBSTACLE_MAP = map_occupancy_grid()

    # Sample random start and goal

    all_start_options = [GridPosition(x, y) for x=1:20 for y in 1:20 if (OBSTACLE_MAP[x,y] == false)]

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

    start_pos = GridPosition(6,2)
    # start_pos = sample(all_start_options)


    # all_goal_options = [GridPosition(x, y) for x=1:20 for y in 2:10 if ((OBSTACLE_MAP[x,8] == false) && (GridPosition(x,y)!=start_pos))]
    goal_options = [GridPosition(x, y) for x=6:15 for y=14:15]

    # goal_options = sample(all_goal_options,10,replace=false)


    # true_goal = GridPosition(1, 8)
    true_goal_index = 12

    # Need to replace!
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
















# point_in_neighbor = get_point_belonging_to_set(20, 20,neighbor_state,neighbor_H_A,neighbor_H_b)
#
# # verify that the neighbor is not an obstacle
# neighbor_in_obstacle=false
# # for os in obstacle_states
# #     if is_in_obstacle_state(neighbor_state,os,obstacle_indices_in_A)
# #         neighbor_in_obstacle=true
# #     end
# # end
#
# if ~isnothing(point_in_neighbor)
#     for os in obstacle
#         if is_in_region(os.A, os.b, point_in_neighbor)
#             neighbor_in_obstacle = true
#         end
#     end
# else
#     neighbor_in_obstacle = true
# end
#
# if ~neighbor_in_obstacle
# # if neighbor_state ∉ obstacle_states
#     # Add the neighbor to the queue
#     neighbor_polytope = waitlisted_polytope(neighbor_H_A, neighbor_H_b, neighbor_state, vtx_id, j, i)
#     enqueue!(openset, neighbor_polytope)
# end




function is_in_obstacle_state(global_state, obstacle_state)
    important_indices = obstacle_state[2]
    obstacle_vec = obstacle_state[1]

    sequence = global_state[important_indices]

    if sequence==obstacle_vec
        return true
    end
    return false
end

# is_in_obstacle_state(point,obstacle_map)

# function get_point_belonging_to_set(bdry_x, bdry_y,global_sequence,global_A,global_b)
#     """set verifies (global_A*x-global_b)"""
#     for x=1:bdry_x
#         for y=1:bdry_y
#             point_sequence =[elt <=0 for elt in global_A*[x,y]-global_b]
#             if point_sequence==global_sequence
#                 return [x,y]
#             end
#         end
#     end
#     return nothing
# end
