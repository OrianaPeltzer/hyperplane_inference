my_belief_colormap = colormap("blues", 20)
my_green_belief_colormap = colormap("greens", 20)

function POMDPTools.render(pomdp::MapWorld, true_state::GridState, belief_state::GridBeliefState)

    nx, ny = pomdp.grid_side, pomdp.grid_side
    cells = []
    for x in 1:nx, y in 1:ny

        # Initialize a cell and set default color to white
        ctx = cell_ctx((x,y), (nx,ny))
        clr = "white"

        # If there is an obstacle, set it to black
        if pomdp.obstacle_map[x,y]
            clr = "black"
        end

        # If it is in one of the belief states, set its color to the belief value
        goal_index = findfirst(isequal(GridPosition(x,y)), pomdp.goal_options)
        if goal_index != nothing
            belief_value = sum(belief_state.belief_intention[goal_index,:])
            clr = my_belief_colormap[convert(Int64, round(10*belief_value, digits=0))+1]
        end

        cell = compose(ctx, rectangle(), fill(clr))
        push!(cells, cell)
    end

    grid = compose(context(), linewidth(0.5mm), stroke("white"), cells...)

    # myline = compose(context(), line([(1/nx, (ny-1)/ny), (2/nx, (ny-3)/ny)]), stroke("black"), linewidth(0.5mm))

    # grid = compose(context(), linewidth(0.5mm), stroke("gray"), cells...)
    outline = compose(context(), linewidth(1mm), rectangle())

    # Plot the agent
    agent_ctx = cell_ctx((true_state.position[1], true_state.position[2]), (nx,ny))
    agent = render_agent(agent_ctx)

    # Plot belief for each neighbor hyperplane
    G = pomdp.hyperplane_graph
    hyperplane_A, hyperplane_b = pos_to_neighbor_matrices(belief_state.position, G)
    current_vtx_id = pos_to_region_index(belief_state.position, G)
    preference_neighbors = get_prop(G,current_vtx_id, :pref_neighbors)

    preference_lines = []
    for i=1:length(preference_neighbors)
        pref_row_in_A = preference_neighbors[i].idx_in_A
        marginal_preference = sum(belief_state.belief_intention[:,i])
        clr = my_green_belief_colormap[convert(Int64, round(10*marginal_preference, digits=0))+1]

        # Vector pointing from robot to direction orthogonal to + towards Ax=b
        p1 = belief_state.position - [0.2,0.2]
        p2 = p1 + (0.5 + marginal_preference) .* (hyperplane_A[pref_row_in_A,:] ./ abs(sum(hyperplane_A[pref_row_in_A,:])))

        myline = compose(context(), line([(p1[1]/nx, (ny-p1[2])/ny), (p2[1]/nx, (ny-p2[2])/ny)]), stroke(clr), linewidth(0.5mm))

        push!(preference_lines, myline)
    end
    combined_preference_lines = compose(context(), preference_lines...)


    # Plot each hyperplane
    hyperplane_points = pos_to_lines(belief_state.position)
    hyper_lines = []
    for i=1:length(hyperplane_points)
        # marginal_preference = sum(belief_state.belief_intention[:,i])
        clr = "blue"

        p1 = hyperplane_points[i][1]
        p2 = hyperplane_points[i][2]
        myline = compose(context(), line([(p1[1]/nx, (ny-p1[2])/ny), (p2[1]/nx, (ny-p2[2])/ny)]), stroke(clr), linewidth(0.5mm))

        push!(hyper_lines, myline)
    end
    combined_lines = compose(context(), hyper_lines...)

    # Plot the goal
    goal_pos = pomdp.goal_options[true_state.goal_index]
    goal_ctx = cell_ctx((goal_pos[1], goal_pos[2]), (nx,ny))
    goal = render_goal(goal_ctx)

    sz = min(w,h)
    return compose(context(), goal, combined_preference_lines, agent, combined_lines, grid, outline)
end

function POMDPTools.render(pomdp::MapWorld, true_state::GridState, belief_state::GridBeliefStateGoal)

    nx, ny = pomdp.grid_side, pomdp.grid_side
    cells = []
    for x in 1:nx, y in 1:ny

        # Initialize a cell and set default color to white
        ctx = cell_ctx((x,y), (nx,ny))
        clr = "white"

        # If there is an obstacle, set it to black
        if pomdp.obstacle_map[x,y]
            clr = "black"
        end

        # If it is in one of the belief states, set its color to the belief value
        goal_index = findfirst(isequal(GridPosition(x,y)), pomdp.goal_options)
        if goal_index != nothing
            belief_value = sum(belief_state.belief_intention[goal_index])
            clr = my_belief_colormap[convert(Int64, round(10*belief_value, digits=0))+1]
        end

        cell = compose(ctx, rectangle(), fill(clr))
        push!(cells, cell)
    end

    grid = compose(context(), linewidth(0.5mm), stroke("white"), cells...)

    # myline = compose(context(), line([(1/nx, (ny-1)/ny), (2/nx, (ny-3)/ny)]), stroke("black"), linewidth(0.5mm))

    # grid = compose(context(), linewidth(0.5mm), stroke("gray"), cells...)
    outline = compose(context(), linewidth(1mm), rectangle())

    # Plot the agent
    agent_ctx = cell_ctx((true_state.position[1], true_state.position[2]), (nx,ny))
    agent = render_agent(agent_ctx)

    # Plot belief for each neighbor hyperplane
    G = pomdp.hyperplane_graph
    hyperplane_A, hyperplane_b = pos_to_neighbor_matrices(belief_state.position, G)
    current_vtx_id = pos_to_region_index(belief_state.position, G)
    preference_neighbors = get_prop(G,current_vtx_id, :pref_neighbors)

    # preference_lines = []
    # for i=1:length(preference_neighbors)
    #     pref_row_in_A = preference_neighbors[i].idx_in_A
    #     marginal_preference = sum(belief_state.belief_intention[:,i])
    #     clr = my_green_belief_colormap[convert(Int64, round(10*marginal_preference, digits=0))+1]
    #
    #     # Vector pointing from robot to direction orthogonal to + towards Ax=b
    #     p1 = belief_state.position - [0.2,0.2]
    #     p2 = p1 + (0.5 + marginal_preference) .* (hyperplane_A[pref_row_in_A,:] ./ abs(sum(hyperplane_A[pref_row_in_A,:])))
    #
    #     myline = compose(context(), line([(p1[1]/nx, (ny-p1[2])/ny), (p2[1]/nx, (ny-p2[2])/ny)]), stroke(clr), linewidth(0.5mm))
    #
    #     push!(preference_lines, myline)
    # end
    # combined_preference_lines = compose(context(), preference_lines...)


    # Plot each hyperplane
    hyperplane_points = pos_to_lines(belief_state.position)
    hyper_lines = []
    for i=1:length(hyperplane_points)
        # marginal_preference = sum(belief_state.belief_intention[:,i])
        clr = "blue"

        p1 = hyperplane_points[i][1]
        p2 = hyperplane_points[i][2]
        myline = compose(context(), line([(p1[1]/nx, (ny-p1[2])/ny), (p2[1]/nx, (ny-p2[2])/ny)]), stroke(clr), linewidth(0.5mm))

        push!(hyper_lines, myline)
    end
    combined_lines = compose(context(), hyper_lines...)

    # Plot the goal
    goal_pos = pomdp.goal_options[true_state.goal_index]
    goal_ctx = cell_ctx((goal_pos[1], goal_pos[2]), (nx,ny))
    goal = render_goal(goal_ctx)

    sz = min(w,h)
    return compose(context(), goal, agent, combined_lines, grid, outline)
end

function POMDPTools.render(pomdp::MapWorld, true_state::GridState)

    nx, ny = pomdp.grid_side, pomdp.grid_side
    cells = []
    for x in 1:nx, y in 1:ny

        # Initialize a cell and set default color to white
        ctx = cell_ctx((x,y), (nx,ny))
        clr = "white"

        # If there is an obstacle, set it to black
        if pomdp.obstacle_map[x,y]
            clr = "black"
        end

        # If it is in one of the belief states, set its color to the belief value
        goal_index = findfirst(isequal(GridPosition(x,y)), pomdp.goal_options)
        if goal_index != nothing
            belief_value = 0.1
            clr = my_belief_colormap[convert(Int64, round(10*belief_value, digits=0))+1]
        end

        cell = compose(ctx, rectangle(), fill(clr))
        push!(cells, cell)
    end

    grid = compose(context(), linewidth(0.5mm), stroke("white"), cells...)

    # myline = compose(context(), line([(1/nx, (ny-1)/ny), (2/nx, (ny-3)/ny)]), stroke("black"), linewidth(0.5mm))

    # grid = compose(context(), linewidth(0.5mm), stroke("gray"), cells...)
    outline = compose(context(), linewidth(1mm), rectangle())

    # Plot the agent
    agent_ctx = cell_ctx((true_state.position[1], true_state.position[2]), (nx,ny))
    agent = render_agent(agent_ctx)

    # Plot belief for each neighbor hyperplane
    G = pomdp.hyperplane_graph
    hyperplane_A, hyperplane_b = pos_to_neighbor_matrices(true_state.position, G)
    current_vtx_id = pos_to_region_index(true_state.position, G)
    preference_neighbors = get_prop(G,current_vtx_id, :pref_neighbors)

    # preference_lines = []
    # for i=1:length(preference_neighbors)
    #     pref_row_in_A = preference_neighbors[i].idx_in_A
    #     marginal_preference = sum(belief_state.belief_intention[:,i])
    #     clr = my_green_belief_colormap[convert(Int64, round(10*marginal_preference, digits=0))+1]
    #
    #     # Vector pointing from robot to direction orthogonal to + towards Ax=b
    #     p1 = belief_state.position - [0.2,0.2]
    #     p2 = p1 + (0.5 + marginal_preference) .* (hyperplane_A[pref_row_in_A,:] ./ abs(sum(hyperplane_A[pref_row_in_A,:])))
    #
    #     myline = compose(context(), line([(p1[1]/nx, (ny-p1[2])/ny), (p2[1]/nx, (ny-p2[2])/ny)]), stroke(clr), linewidth(0.5mm))
    #
    #     push!(preference_lines, myline)
    # end
    # combined_preference_lines = compose(context(), preference_lines...)


    # Plot each hyperplane
    hyperplane_points = pos_to_lines(true_state.position)
    hyper_lines = []
    for i=1:length(hyperplane_points)
        clr = "blue"

        p1 = hyperplane_points[i][1]
        p2 = hyperplane_points[i][2]
        myline = compose(context(), line([(p1[1]/nx, (ny-p1[2])/ny), (p2[1]/nx, (ny-p2[2])/ny)]), stroke(clr), linewidth(0.5mm))

        push!(hyper_lines, myline)
    end
    combined_lines = compose(context(), hyper_lines...)

    # Plot the goal
    goal_pos = pomdp.goal_options[true_state.goal_index]
    goal_ctx = cell_ctx((goal_pos[1], goal_pos[2]), (nx,ny))
    goal = render_goal(goal_ctx)

    sz = min(w,h)
    return compose(context(), goal, agent, combined_lines, grid, outline)
end

function POMDPTools.render(pomdp::MapWorld, states_pref, obs_pref,states_goal,obs_goal)

    nx, ny = pomdp.grid_side, pomdp.grid_side
    cells = []
    for x in 1:nx, y in 1:ny

        # Initialize a cell and set default color to white
        ctx = cell_ctx((x,y), (nx,ny))
        clr = "white"

        # If there is an obstacle, set it to black
        if pomdp.obstacle_map[x,y]
            clr = "black"
        end

        # If it is in one of the belief states, set its color to the belief value
        goal_index = findfirst(isequal(GridPosition(x,y)), pomdp.goal_options)
        if goal_index != nothing
            belief_value = 1.0/length(pomdp.goal_options)
            clr = my_belief_colormap[convert(Int64, round(10*belief_value, digits=0))+1]
        end

        cell = compose(ctx, rectangle(), fill(clr))
        push!(cells, cell)
    end

    grid = compose(context(), cells...)

    # myline = compose(context(), line([(1/nx, (ny-1)/ny), (2/nx, (ny-3)/ny)]), stroke("black"), linewidth(0.5mm))

    # grid = compose(context(), linewidth(0.5mm), stroke("gray"), cells...)
    outline = compose(context(), linewidth(1mm), rectangle(), stroke("black"))

    # Start agent
    start_agent_ctx = cell_ctx((pref_states[1].position[1], pref_states[1].position[2]), (nx,ny))
    start_agent = render_agent(start_agent_ctx)

    # Advancing agents
    pref_agents = []
    for (i,true_state) in enumerate(states_pref)
        agent_ctx = cell_ctx((true_state.position[1], true_state.position[2]), (nx,ny))
        agent = render_prev_agent(agent_ctx, "blue",convert(Int64, round(19*i/length(states_pref), digits=0))+1)
        push!(pref_agents,agent)
    end
    pref_agents_composed = compose(context(), pref_agents...)

    goal_agents = []
    for (i,true_state) in enumerate(states_goal)
        agent_ctx = cell_ctx((true_state.position[1], true_state.position[2]), (nx,ny))
        agent = render_prev_agent(agent_ctx, "green",convert(Int64, round(19*i/length(states_goal), digits=0))+1)
        push!(goal_agents,agent)
    end
    goal_agents_composed = compose(context(), goal_agents...)


    # Plot belief for each neighbor hyperplane
    G = pomdp.hyperplane_graph

    # PREFERENCE OBSERVATION ARROWS
    pref_observation_arrows = []
    for i=1:length(obs_pref)
        # pref_row_in_A = preference_neighbors[i].idx_in_A
        # marginal_preference = sum(belief_state.belief_intention[:,i])
        clr = my_belief_colormap[convert(Int64, round(19*i/length(obs_pref), digits=0))+1]
        # clr="blue"
        true_state = states_pref[i]

        # Vector pointing from robot to direction orthogonal to + towards Ax=b
        p1 = true_state.position - [0.5,0.5]
        p2 = p1 + (0.6 .*heading_to_vector(obs_pref[i]))

        myline = compose(context(),arrow(), stroke("black"), fill("black"),
                         (context(), line([(p1[1]/nx, (ny-p1[2])/ny), (p2[1]/nx, (ny-p2[2])/ny)]), stroke("black"), linewidth(0.5mm)))

        push!(pref_observation_arrows, myline)
    end
    combined_obs_pref_lines = compose(context(), pref_observation_arrows...)

    # GOAL OBSERVATION ARROWS
    goal_observation_arrows = []
    for i=1:length(obs_goal)
        # pref_row_in_A = preference_neighbors[i].idx_in_A
        # marginal_preference = sum(belief_state.belief_intention[:,i])
        clr = my_green_belief_colormap[convert(Int64, round(19*i/length(obs_goal), digits=0))+1]
        # clr="blue"
        true_state = states_goal[i]

        # Vector pointing from robot to direction orthogonal to + towards Ax=b
        p1 = true_state.position - [0.5,0.5]
        p2 = p1 + (0.6 .* heading_to_vector(obs_goal[i]))

        myline = compose(context(),arrow(), stroke("black"), fill("black"),
                         (context(), line([(p1[1]/nx, (ny-p1[2])/ny), (p2[1]/nx, (ny-p2[2])/ny)]), stroke("black"), linewidth(0.5mm)))

        push!(goal_observation_arrows, myline)
    end
    combined_goal_pref_lines = compose(context(), goal_observation_arrows...)


    # Plot each hyperplane
    all_hyperplane_points = pos_to_lines_allatonce()
    hyper_lines = []
    for hyperplane_points in all_hyperplane_points
        for i=1:length(hyperplane_points)
            # marginal_preference = sum(belief_state.belief_intention[:,i])
            clr = "blue"

            p1 = hyperplane_points[i][1]
            p2 = hyperplane_points[i][2]
            myline = compose(context(), line([(p1[1]/nx, (ny-p1[2])/ny), (p2[1]/nx, (ny-p2[2])/ny)]), stroke(clr), linewidth(0.5mm))

            push!(hyper_lines, myline)
        end
    end
    combined_lines = compose(context(), hyper_lines...) # REMOVED FROM COMPOSE

    # Plot the goal
    goal_pos = pomdp.goal_options[pomdp.true_goal_index]
    goal_ctx = cell_ctx((goal_pos[1], goal_pos[2]), (nx,ny))
    goal = render_goal(goal_ctx)

    sz = min(w,h)
    return compose(context(), goal,  pref_observation_arrows, goal_observation_arrows, start_agent,  goal_agents_composed, pref_agents_composed, grid, outline)
end


function POMDPTools.render(pomdp::MapWorld)

    nx, ny = pomdp.grid_side, pomdp.grid_side
    cells = []
    for x in 1:nx, y in 1:ny

        # Initialize a cell and set default color to white
        ctx = cell_ctx((x,y), (nx,ny))
        clr = "white"

        # If there is an obstacle, set it to black
        if pomdp.obstacle_map[x,y]
            clr = "black"
        end

        # # If it is in one of the belief states, set its color to the belief value
        # goal_index = findfirst(isequal(GridPosition(x,y)), pomdp.goal_options)
        # if goal_index != nothing
        #     belief_value = 0.1
        #     clr = my_belief_colormap[convert(Int64, round(10*belief_value, digits=0))+1]
        # end

        cell = compose(ctx, rectangle(), fill(clr))
        push!(cells, cell)
    end

    grid = compose(context(), linewidth(0.5mm), cells...)

    # myline = compose(context(), line([(1/nx, (ny-1)/ny), (2/nx, (ny-3)/ny)]), stroke("black"), linewidth(0.5mm))

    # grid = compose(context(), linewidth(0.5mm), stroke("gray"), cells...)
    outline = compose(context(), linewidth(1mm), rectangle())

    # Plot the agent
    # agent_ctx = cell_ctx((true_state.position[1], true_state.position[2]), (nx,ny))
    # agent = render_agent(agent_ctx)

    # Plot belief for each neighbor hyperplane
    # G = pomdp.hyperplane_graph
    # hyperplane_A, hyperplane_b = pos_to_neighbor_matrices(true_state.position, G)
    # current_vtx_id = pos_to_region_index(true_state.position, G)
    # preference_neighbors = get_prop(G,current_vtx_id, :pref_neighbors)

    # preference_lines = []
    # for i=1:length(preference_neighbors)
    #     pref_row_in_A = preference_neighbors[i].idx_in_A
    #     marginal_preference = sum(belief_state.belief_intention[:,i])
    #     clr = my_green_belief_colormap[convert(Int64, round(10*marginal_preference, digits=0))+1]
    #
    #     # Vector pointing from robot to direction orthogonal to + towards Ax=b
    #     p1 = belief_state.position - [0.2,0.2]
    #     p2 = p1 + (0.5 + marginal_preference) .* (hyperplane_A[pref_row_in_A,:] ./ abs(sum(hyperplane_A[pref_row_in_A,:])))
    #
    #     myline = compose(context(), line([(p1[1]/nx, (ny-p1[2])/ny), (p2[1]/nx, (ny-p2[2])/ny)]), stroke(clr), linewidth(0.5mm))
    #
    #     push!(preference_lines, myline)
    # end
    # combined_preference_lines = compose(context(), preference_lines...)


    # # Plot each hyperplane
    # hyperplane_points = pos_to_lines(true_state.position)
    # hyper_lines = []
    # for i=1:length(hyperplane_points)
    #     clr = "blue"
    #
    #     p1 = hyperplane_points[i][1]
    #     p2 = hyperplane_points[i][2]
    #     myline = compose(context(), line([(p1[1]/nx, (ny-p1[2])/ny), (p2[1]/nx, (ny-p2[2])/ny)]), stroke(clr), linewidth(0.5mm))
    #
    #     push!(hyper_lines, myline)
    # end
    # combined_lines = compose(context(), hyper_lines...)
    #
    # # Plot the goal
    # goal_pos = pomdp.goal_options[true_state.goal_index]
    # goal_ctx = cell_ctx((goal_pos[1], goal_pos[2]), (nx,ny))
    # goal = render_goal(goal_ctx)

    sz = min(w,h)
    return compose(context(), grid, outline)
end





function cell_ctx(xy, size)
    nx, ny = size
    x, y = xy
    return context((x - 1)/nx, (ny-y)/ny, 1/nx, 1/ny)
end


# function v_line_ctx(xy, size)
#     nx, ny = size
#     x, y = xy
#     return context((x - 1)/nx, (ny-y)/ny, 1/nx, 3/ny)
# end
#
# function h_line_ctx(xy, size)
#     nx, ny = size
#     x, y = xy
#     return context((x - 1)/nx, (ny-y)/ny, 3/nx, 1/ny)
# end


function render_agent(ctx)
    center = compose(context(), ellipse(0.5, 0.5, 0.20, 0.3), fill("orange"), stroke("black"))
    ld_rot = compose(context(), Compose.circle(0.2,0.8,0.17), fill("gray"), stroke("black"))
    rd_rot = compose(context(), Compose.circle(0.8,0.8,0.17), fill("gray"), stroke("black"))
    lu_rot = compose(context(), Compose.circle(0.2,0.2,0.17), fill("gray"), stroke("black"))
    ru_rot = compose(context(), Compose.circle(0.8,0.2,0.17), fill("gray"), stroke("black"))
    return compose(ctx, ld_rot, rd_rot, lu_rot, ru_rot, center)
end

function render_prev_agent(ctx,color,intensity)
    if color=="blue"
        clr = my_belief_colormap[intensity]
    else
        clr = my_green_belief_colormap[intensity]
    end
    center = compose(context(), ellipse(0.5, 0.5, 0.20, 0.3), fillopacity(float(intensity)/20.0), fill(color), stroke("black"))
    ld_rot = compose(context(), Compose.circle(0.2,0.8,0.17), fillopacity(float(intensity)/20.0), fill(color), stroke("black"))
    rd_rot = compose(context(), Compose.circle(0.8,0.8,0.17), fillopacity(float(intensity)/20.0), fill(color), stroke("black"))
    lu_rot = compose(context(), Compose.circle(0.2,0.2,0.17), fillopacity(float(intensity)/20.0), fill(color), stroke("black"))
    ru_rot = compose(context(), Compose.circle(0.8,0.2,0.17), fillopacity(float(intensity)/20.0), fill(color), stroke("black"))
    return compose(ctx, ld_rot, rd_rot, lu_rot, ru_rot, center)
end

# function render_agent(ctx,color,intensity)
#     if color=="blue"
#         clr = my_belief_colormap[intensity]
#     else
#         clr = my_green_belief_colormap[intensity]
#     end
#     center = compose(context(), ellipse(0.5, 0.5, 0.20, 0.3), fill(clr), stroke("black"))
#     ld_rot = compose(context(), Compose.circle(0.2,0.8,0.17), fill(clr), stroke("black"))
#     rd_rot = compose(context(), Compose.circle(0.8,0.8,0.17), fill(clr), stroke("black"))
#     lu_rot = compose(context(), Compose.circle(0.2,0.2,0.17), fill(clr), stroke("black"))
#     ru_rot = compose(context(), Compose.circle(0.8,0.2,0.17), fill(clr), stroke("black"))
#     return compose(ctx, ld_rot, rd_rot, lu_rot, ru_rot, center)
# end

function render_goal(ctx)
    goal = compose(context(), Compose.circle(0.5, 0.5, 0.15), fill("pink"), stroke("black"))
    return compose(ctx, goal)
end



# render with a path

# function POMDPTools.render(pomdp::MapWorld, true_state::GridState, belief_state::GridBeliefState, path)
#
#     path_v = [e.src for e in path]
#     dest = path[length(path)].dst
#
#     nx, ny = pomdp.grid_side, pomdp.grid_side
#     cells = []
#     for x in 1:nx, y in 1:ny
#
#         vertex = pomdp.K_map[x,y]
#
#         if pomdp.obstacle_map[x,y]
#             if vertex ∈ path_v
#                 clr="red"
#             else
#                 clr = "black"
#             end
#         elseif vertex ∈ path_v
#             clr="green"
#         elseif vertex == dest
#             clr="blue"
#         else
#             clr="white"
#         end
#
#         # Initialize a cell and set default color to white
#         ctx = cell_ctx((x,y), (nx,ny))
#         # clr = "white"
#
#         # If there is an obstacle, set it to black
#         # if pomdp.obstacle_map[x,y]
#         #     clr = "black"
#         # end
#
#         # If it is in one of the belief states, set its color to the belief value
#         # goal_index = findfirst(isequal(GridPosition(x,y)), pomdp.goal_options)
#         # if goal_index != nothing
#         #     belief_value = sum(belief_state.belief_intention[goal_index,:])
#         #     clr = my_belief_colormap[convert(Int64, round(10*belief_value, digits=0))+1]
#         # end
#
#         cell = compose(ctx, rectangle(), fill(clr))
#         push!(cells, cell)
#     end
#
#     grid = compose(context(), linewidth(0.5mm), stroke("white"), cells...)
#
#     # grid = compose(context(), linewidth(0.5mm), stroke("gray"), cells...)
#     outline = compose(context(), linewidth(1mm), rectangle())
#
#     # Plot the agent
#     agent_ctx = cell_ctx((true_state.position[1], true_state.position[2]), (nx,ny))
#     agent = render_agent(agent_ctx)
#
#     # Plot the goal
#     goal_pos = pomdp.goal_options[true_state.goal_index]
#     goal_ctx = cell_ctx((goal_pos[1], goal_pos[2]), (nx,ny))
#     goal = render_goal(goal_ctx)
#
#     sz = min(w,h)
#     return compose(context(), goal, agent, grid, outline)
# end

# Render the graph of convex sets on the obstacle map
function POMDPTools.render(pomdp::MapWorld, true_state::GridState, G::MetaGraph)

    color_list = ["darkorchid",
                  "green1",
                  "pink1",
                  "blue1",
                  "chocolate",
                  "wheat1",
                  "grey69",
                  "deeppink2",
                  "chocolate2",
                  "grey97",
                  "olivedrab3",
                  "mediumpurple2",
                  "ivory1",
                  "gray95",
                  "lemonchiffon2",
                  "palegreen4",
                  "cornsilk4",
                  "red1",
                  "darkgoldenrod4",
                  "lightpink4",
                  "wheat2",
                  "purple2",
                  "darkorchid",
                    "green1",
                    "pink1",
                    "blue1",
                    "chocolate",
                    "wheat1",
                    "grey69",
                    "deeppink2",
                    "chocolate2",
                    "grey97",
                    "olivedrab3",
                    "mediumpurple2",
                    "ivory1",
                    "gray95",
                    "lemonchiffon2",
                    "palegreen4",
                    "cornsilk4",
                    "red1",
                    "darkgoldenrod4",
                    "lightpink4",
                    "wheat2",
                    "purple2",
                    "darkorchid",
                  "green1",
                  "pink1",
                  "blue1",
                  "chocolate",
                  "wheat1",
                  "grey69",
                  "deeppink2",
                  "chocolate2",
                  "grey97",
                  "olivedrab3",
                  "mediumpurple2",
                  "ivory1",
                  "gray95",
                  "lemonchiffon2",
                  "palegreen4",
                  "cornsilk4",
                  "red1",
                  "darkgoldenrod4",
                  "lightpink4",
                  "wheat2",
                  "purple2",
                  "darkorchid",
                    "green1",
                    "pink1",
                    "blue1",
                    "chocolate",
                    "wheat1",
                    "grey69",
                    "deeppink2",
                    "chocolate2",
                    "grey97",
                    "olivedrab3",
                    "mediumpurple2",
                    "ivory1",
                    "gray95",
                    "lemonchiffon2",
                    "palegreen4",
                    "cornsilk4",
                    "red1",
                    "darkgoldenrod4",
                    "lightpink4",
                    "wheat2",
                    "purple2",
                    "darkorchid",
                                  "green1",
                                  "pink1",
                                  "blue1",
                                  "chocolate",
                                  "wheat1",
                                  "grey69",
                                  "deeppink2",
                                  "chocolate2",
                                  "grey97",
                                  "olivedrab3",
                                  "mediumpurple2",
                                  "ivory1",
                                  "gray95",
                                  "lemonchiffon2",
                                  "palegreen4",
                                  "cornsilk4",
                                  "red1",
                                  "darkgoldenrod4",
                                  "lightpink4",
                                  "wheat2",
                                  "purple2",
                                  "darkorchid",
                                    "green1",
                                    "pink1",
                                    "blue1",
                                    "chocolate",
                                    "wheat1",
                                    "grey69",
                                    "deeppink2",
                                    "chocolate2",
                                    "grey97",
                                    "olivedrab3",
                                    "mediumpurple2",
                                    "ivory1",
                                    "gray95",
                                    "lemonchiffon2",
                                    "palegreen4",
                                    "cornsilk4",
                                    "red1",
                                    "darkgoldenrod4",
                                    "lightpink4",
                                    "wheat2",
                                    "purple2",
                                    "darkorchid",
                                  "green1",
                                  "pink1",
                                  "blue1",
                                  "chocolate",
                                  "wheat1",
                                  "grey69",
                                  "deeppink2",
                                  "chocolate2",
                                  "grey97",
                                  "olivedrab3",
                                  "mediumpurple2",
                                  "ivory1",
                                  "gray95",
                                  "lemonchiffon2",
                                  "palegreen4",
                                  "cornsilk4",
                                  "red1",
                                  "darkgoldenrod4",
                                  "lightpink4",
                                  "wheat2",
                                  "purple2",
                                  "darkorchid",
                                    "green1",
                                    "pink1",
                                    "blue1",
                                    "chocolate",
                                    "wheat1",
                                    "grey69",
                                    "deeppink2",
                                    "chocolate2",
                                    "grey97",
                                    "olivedrab3",
                                    "mediumpurple2",
                                    "ivory1",
                                    "gray95",
                                    "lemonchiffon2",
                                    "palegreen4",
                                    "cornsilk4",
                                    "red1",
                                    "darkgoldenrod4",
                                    "lightpink4",
                                    "wheat2",
                                    "purple2"]

    nx, ny = pomdp.grid_side, pomdp.grid_side
    cells = []
    white_cells = []
    texts=[]
    for x in 1:nx, y in 1:ny

        # Initialize a cell
        ctx = cell_ctx((x,y), (nx,ny))
        # Find index of vertex for set
        set_idx = pos_to_region_index(GridPosition(x,y), G, adjustment=0.1)
        f_o = 1.0
        if set_idx > 0
            clr = color_list[set_idx]
            # f_o = 1.0
        else
            clr = "black"
            # f_o = 1.0
        end


        # If there is an obstacle, set it to black
        # if pomdp.obstacle_map[x,y]
        #     clr = "black"
        # end

        cell = compose(ctx, rectangle(), fill(clr))
        white_cell = compose(ctx, rectangle(), fill("white"), fillopacity(1.0))
        push!(cells, cell)
        push!(white_cells, white_cell)
        # txt = compose(ctx, text(string(set_idx)), fill("black"))
        # push!(texts,txt)
    end

    # grid = compose(context(), linewidth(0.1mm), stroke("white"), cells...)
    grid = compose(context(), cells...)
    wc = compose(context(), white_cells...)
    # textarray= compose(context(), linewidth(0.1mm), stroke("white"), texts...)

    # myline = compose(context(), line([(1/nx, (ny-1)/ny), (2/nx, (ny-3)/ny)]), stroke("black"), linewidth(0.5mm))

    # grid = compose(context(), linewidth(0.5mm), stroke("gray"), cells...)
    outline = compose(context(), linewidth(1mm), rectangle())

    # Plot the agent
    agent_ctx = cell_ctx((true_state.position[1], true_state.position[2]), (nx,ny))
    agent = render_agent(agent_ctx)

    # # Plot each hyperplane
    # hyperplane_points = pos_to_lines(belief_state.position)
    # hyper_lines = []
    # for i=1:length(hyperplane_points)
    #     marginal_preference = sum(belief_state.belief_intention[:,i])
    #     clr = my_belief_colormap[convert(Int64, round(10*marginal_preference, digits=0))+1]
    #
    #     p1 = hyperplane_points[i][1]
    #     p2 = hyperplane_points[i][2]
    #     myline = compose(context(), line([(p1[1]/nx, (ny-p1[2])/ny), (p2[1]/nx, (ny-p2[2])/ny)]), stroke(clr), linewidth(0.5mm))
    #
    #     push!(hyper_lines, myline)
    # end
    # combined_lines = compose(context(), hyper_lines...)

    # Plot the goal
    goal_pos = pomdp.goal_options[true_state.goal_index]
    goal_ctx = cell_ctx((goal_pos[1], goal_pos[2]), (nx,ny))
    goal = render_goal(goal_ctx)

    sz = min(w,h)
    # return compose(context(), goal, agent, combined_lines, grid, outline)
    return compose(context(), goal, agent, grid, outline)
end
