"""
    Problem Specific Structures and Functions for the POMDP.
    Can be used with any solver.
"""

# Alias for 2D position
const GridPosition = SVector{2,Int64}

struct GridState
    position::GridPosition
    done::Bool # are we in a terminal state?
    goal_index::Int64 # This is the goal
    neighbor_A::Matrix # A*current_state - b <= 0
    neighbor_b::Array
end

# We'll need a separate struct to explicitly represent the belief state
# (rather than implicitly as a distribution over states, as that would be intractable)
# We'll need to define the belief updater functions
@with_kw struct GridBeliefState
    position::GridPosition
    done::Bool # are we in a terminal state?
    neighbor_A::Matrix # A*current_state - b <= 0
    neighbor_b::Array
    belief_intention::Matrix{Float64} # size #goals*(1+#neighbors)
end

@with_kw struct HumanAngleObservation
    direction::Float64 # angle
    received_observation::Bool # Received human input
    visited_wrong_goal::Bool # If we visited a hypothetical goal that wasn't one
end

@with_kw struct HumanInputObservation
    heading::Char # ['N', 'S', 'E', 'W', '0', '1', '2', '3']
    received_observation::Bool # Received human input
    visited_wrong_goal::Bool # If we visited a hypothetical goal that wasn't one
end

function angle_to_heading(angle::Float64)

    if -pi/8 < angle && angle <= pi/8
        return 'E'
    elseif pi/8 < angle && angle <= 3*pi/8
        return '3' # NE
    elseif 3*pi/8 < angle && angle <= 5*pi/8
        return 'N'
    elseif 5*pi/8 < angle && angle <= 7*pi/8
        return '2' # NW
    elseif -7*pi/8 < angle && angle <= -5*pi/8
        return '0' # SW
    elseif -5*pi/8 < angle && angle <= -3*pi/8
        return 'S'
    elseif -3*pi/8 < angle && angle <= -pi/8
        return '1' # SE
    else
        return 'W'
    end
end

"""Struct for true map of the world"""
# The POMDP should contain the true map - remember it is your nature hat
@with_kw struct MapWorld <: POMDPs.POMDP{GridState,Char,HumanInputObservation}
    obstacle_map::Matrix{Bool}        = fill(false, 10, 10)
    grid_side::Int64                  = 10
    discount_factor::Float64          = 0.9
    penalty::Float64                  = -1.0
    diag_penalty::Float64             = -1.414 # sqrt(2)
    reward::Float64                   = 10.0
    new_obs_weight::Float64           = 0.75
    goal_options::Array{GridPosition} = [GridPosition(7, 7),
                                         GridPosition(1, 8)]
    # goal::GridPosition                = GridPosition(3, 3)
    n_possible_goals::Int64           = 2
    human_input_std::Float64          = 0.5
    map_graph::MetaGraph              = MetaGraph()
    K_map::Matrix{Int}                = zeros(1)
    dist_matrix::Matrix{Float64}      = zeros(1)
    dist_matrix_DP::Matrix{Float64}   = zeros(1)
    BII_gamma::Float64                = 0.8
end

"""Constraint encoding what the forbidden transitions are"""
struct NeighborConstraint
    """If Ax - b <= 0 and we transition to x' where Ax' - b has at least one
    positive element at one of the non-permitted indexes, the constraint is violated."""
    A::Matrix{Float64}
    b::Vector
    permitted_idx::Int64
    enforced::Bool
end

get_final_node(path::Vector{E} where {E <: Edge}) = get(path,length(path),Edge(-1,-1)).dst
traversal_time(path::Vector{E} where {E <: Edge},g::MetaGraph) = sum([get_prop(g,e,:weight) for e in path])

function violates_constraint(g::MetaGraph, constraint::NeighborConstraint, v, path)

    if length(path) == 0
        return false
    end

    if !constraint.enforced
        return false
    end
    x0 = get_prop(g,get_final_node(path),:x)
    y0 = get_prop(g,get_final_node(path),:y)
    x1 = get_prop(g,v,:x)
    y1 = get_prop(g,v,:y)

    first_string = constraint.A * [x0,y0] - constraint.b
    if all(<=(0),first_string)
        second_string = constraint.A * [x1,y1] - constraint.b
        for index=1:length(second_string)
            if (second_string[index] > 0) & (index != constraint.permitted_idx)
                return true
            end
        end
    end
    return false
end

"""(Problem Specific) Function to get initial belief state"""
function initial_belief_state(pomdp::MapWorld)

    # Starting state for the robot
    start_state = GridPosition(1, 1)

    start_A, start_b = pos_to_neighbor_matrices(start_state)

    belief_intention = initialize_prior_belief(pomdp, start_state, start_A, start_b)

    return GridBeliefState(start_state, false, start_A, start_b, belief_intention)
end


"""(Problem Specific) Function to get initial state"""
function initial_state(pomdp::MapWorld)

    # Starting state for the robot
    start_state = GridPosition(1, 1)

    start_A, start_b = pos_to_neighbor_matrices(start_state)

    return GridState(start_state, false, 1, start_A, start_b)
end

function initialize_prior_belief(p::MapWorld,pos::GridPosition,A::Matrix,b::Vector)
    # Uniform prior on which goal is the intended one
    belief_intention = zeros(p.n_possible_goals, 1+length(b))

    # Construct prior belief for each goal
    for (g_idx, goal) in enumerate(p.goal_options)
        if is_in_region(A, b, goal)
            belief_intention[g_idx, length(b)+1] = 1.0/p.n_possible_goals
        else
            shortest_path_distance = get_shortest_path_length(p, pos, goal)
            for pref=1:length(b)
                pos_index = p.K_map[pos[1], pos[2]]
                goal_index = p.K_map[goal[1], goal[2]]

                preference_constraint = NeighborConstraint(A,b,pref,true)

                shortest_path_constrained = Graphs.a_star(p.map_graph, pos_index, goal_index, preference_constraint, p.dist_matrix_DP)

                constrained_dist = traversal_time(shortest_path_constrained, p.map_graph)
                diff = constrained_dist - shortest_path_distance
                belief_intention[g_idx,pref] = 1.0#exp(-0.1*diff)
            end
            # normalize prior and divide by n_possible_goals
            belief_intention[g_idx,:] = belief_intention[g_idx,:] / (sum(belief_intention[g_idx,:]) * p.n_possible_goals)
        end
    end
    return belief_intention
end

"""Re-construct prior belief after transition in between regions/sets"""
function reshape_prior_belief(p::MapWorld,pos::GridPosition,new_pos::GridPosition, A::Matrix,b::Vector, bel::GridBeliefState)

    # Collect marginals
    marginals = sum(bel.belief_intention, dims=2)

    # Uniform prior on which goal is the intended one
    belief_intention = zeros(p.n_possible_goals, 1+length(b))

    # Construct prior belief for each goal
    for (g_idx, goal) in enumerate(p.goal_options)
        if is_in_region(A, b, goal)
            belief_intention[g_idx, length(b)+1] = marginals[g_idx]#1.0/p.n_possible_goals
        else
            shortest_path_distance = get_shortest_path_length(p, new_pos, goal)
            for pref=1:length(b)
                pos_index = p.K_map[new_pos[1], new_pos[2]]
                goal_index = p.K_map[goal[1], goal[2]]

                preference_constraint = NeighborConstraint(A,b,pref,true)
                shortest_path_constrained = Graphs.a_star(p.map_graph, pos_index, goal_index, preference_constraint, p.dist_matrix_DP)
                constrained_dist = traversal_time(shortest_path_constrained, p.map_graph)

                diff = constrained_dist - shortest_path_distance
                belief_intention[g_idx,pref] = 1.0#exp(-0.1*diff)

                # If the preference corresponds to where we come from, only assign small belief
                sequence = A*pos-b
                if sequence[pref] > 0
                    belief_intention[g_idx,pref] = 0.4
                end

                if is_obstacle_index(new_pos,pref)
                    belief_intention[g_idx,pref] = 0.0
                end

                if length(shortest_path_constrained) == 0
                    belief_intention[g_idx,pref] = 0.0
                end
            end
            # normalize prior and multiply by marginal goal probabilities
            belief_intention[g_idx,:] = marginals[g_idx] * belief_intention[g_idx,:] / sum(belief_intention[g_idx,:])
        end
    end
    return belief_intention
end

"""Sample a preferred neighbor from a weighted prior distribution"""
function sample_preference(p::MapWorld, pos::GridPosition,goal::GridPosition,A::Matrix,b::Vector)
    intention_distribution = zeros(length(b))

    shortest_path_distance = get_shortest_path_length(p, pos, goal)
    for pref=1:length(b)
        pos_index = p.K_map[pos[1], pos[2]]
        goal_index = p.K_map[goal[1], goal[2]]

        preference_constraint = NeighborConstraint(A,b,pref,true)
        shortest_path_constrained = Graphs.a_star(p.map_graph, pos_index, goal_index, preference_constraint, p.dist_matrix_DP)
        constrained_dist = traversal_time(shortest_path_constrained, p.map_graph)

        diff = constrained_dist - shortest_path_distance
        intention_distribution[pref] = exp(-p.BII_gamma*diff)
    end
    # normalize prior
    intention_distribution = intention_distribution / sum(intention_distribution)

    # Sample from weighted prior
    return sample(1:length(b), Weights(intention_distribution))
end

"""(Problem Specific) Sample a human input from the robot position and true intended goal"""
function sample_direction(p::GridPosition, g::GridPosition, h_std::Float64)
    # Compute direction
    exact_direction = atan(g[2]-p[2], g[1]-p[1])
    # Sample from N(mu,sigma)
    return rand(Normal(exact_direction,h_std))
end

# TODO
"""(Problem Specific) Sample a human input from the robot position and true intended goal"""
function sample_char(map_graph::MetaGraph, p::GridPosition, g::GridPosition, gamma::Float64)
    # directions ['N', 'S', 'E', 'W', '0', '1', '2', '3']
    direction_probabilities = zeros(8)
    # Compute direction
    exact_direction = atan(g[2]-p[2], g[1]-p[1])
    # Sample from N(mu,sigma)
    return rand(Normal(exact_direction,h_std))
end



function compute_measurement_likelihood(p::MapWorld, pos::GridPosition, goal::GridPosition, heading::Char)

    distance_to_goal = get_shortest_path_length(p,pos,goal)

    human_intended_position = update_position(p,pos,heading)

    distance_to_goal_complying = get_shortest_path_length(p, pos, human_intended_position) + get_shortest_path_length(p, human_intended_position, goal)

    diff = distance_to_goal_complying - distance_to_goal

    return exp(-p.BII_gamma * diff)
end

function compute_measurement_likelihood(p::MapWorld, pos::GridPosition, goal::GridPosition, heading::Char, neighbor_A::Matrix, neighbor_b::Vector, preference_index::Int64)
    """Returns the probability that the human created observation "heading" given
    the tuple (goal, neighbor_preference)."""

    # println("Entered compute measurement likelihood function")
    # println("")
    # @show heading
    # @show pos
    # @show goal
    # @show preference_index

    human_intended_position = update_position(p,pos,heading)
    # println("Found human intended position: ", human_intended_position)


    preference_constraint = NeighborConstraint(neighbor_A,neighbor_b,preference_index,true)
    pos_index = p.K_map[pos[1], pos[2]]
    goal_index = p.K_map[goal[1], goal[2]]

    path_to_goal = Graphs.a_star(p.map_graph, pos_index, goal_index, preference_constraint, p.dist_matrix_DP)
    # @show path_to_goal
    # @show [[get_prop(p.map_graph, e.src, :x), get_prop(p.map_graph, e.src, :y)] for e in path_to_goal]
    distance_to_goal = traversal_time(path_to_goal, p.map_graph)
    # @show distance_to_goal

    human_intended_index = p.K_map[human_intended_position[1], human_intended_position[2]]

    # @show human_intended_index

    path_to_goal_from_intended_index = Graphs.a_star(p.map_graph, human_intended_index, goal_index, preference_constraint, p.dist_matrix_DP)

    # @show path_to_goal_from_intended_index
    # @show [[get_prop(p.map_graph, e.src, :x), get_prop(p.map_graph, e.src, :y)] for e in path_to_goal_from_intended_index]

    distance_to_goal_from_intended_index = traversal_time(path_to_goal_from_intended_index, p.map_graph)
    # @show distance_to_goal_from_intended_index

    distance_to_goal_complying = get_shortest_path_length(p, pos, human_intended_position) + distance_to_goal_from_intended_index

    # @show distance_to_goal_complying

    diff = distance_to_goal_complying - distance_to_goal

    return exp(-p.BII_gamma * diff)
end


function update_position(p::MapWorld, pos::GridPosition, a::Char)

    curr_pos = pos

    if a == 'W'
        new_pos = GridPosition(max(1, curr_pos[1]-1), curr_pos[2])
    elseif a == 'E'
        new_pos = GridPosition(min(p.grid_side, curr_pos[1]+1), curr_pos[2])
    elseif a == 'S'
        new_pos = GridPosition(curr_pos[1], max(1, curr_pos[2]-1))
    elseif a == 'N'
        new_pos = GridPosition(curr_pos[1], min(p.grid_side, curr_pos[2]+1))
    elseif a == '0' # SW
        new_pos = GridPosition(max(1, curr_pos[1]-1), max(1, curr_pos[2]-1))
    elseif a == '2' # NW
        new_pos = GridPosition(max(1, curr_pos[1]-1), min(p.grid_side, curr_pos[2]+1))
    elseif a == '1' # SE
        new_pos = GridPosition(min(p.grid_side, curr_pos[1]+1), max(1, curr_pos[2]-1))
    elseif a == '3' # NE
        new_pos = GridPosition(min(p.grid_side, curr_pos[1]+1), min(p.grid_side, curr_pos[2]+1))
    else
        new_pos = curr_pos
    end

    # If we hit an obstacle, we don't actually get to move
    if p.obstacle_map[new_pos...] == true
        new_pos = curr_pos
    end

    return new_pos
end

function get_shortest_path_length(p::MapWorld, start::GridPosition,finish::GridPosition)
    """Returns the length of the shortest path from start to finish"""
    start_index = p.K_map[start[1], start[2]]
    end_index = p.K_map[finish[1], finish[2]]

    # println("start index: ", start_index)
    # println("end index: ", end_index)
    # @show start
    # @show finish

    return p.dist_matrix_DP[start_index,end_index]
end


function initialize_grid_graph(obstacle_map::Matrix{Bool})
    """Returns a grid graph that represents a 2D environment defined by the
        obstacle map matrix.
        Considers that the nominal time to move from one edge to another is 1 if
        the robot moves right, left, up or down, and 1.41 for movements along
        the diagonals.
        The output is a MetaGraph"""
    Ap = obstacle_map
    K = zeros(Int,size(Ap))
    G = MetaGraph()

    k = 0
    for i in 1:size(Ap,1)
        for j in 1:size(Ap,2)
            if Ap[i,j] == false
                k += 1
                add_vertex!(G,
                    Dict(:x=>(i-1),
                    :y=>(j-1))
                    )
                add_edge!(G,nv(G),nv(G))
                set_prop!(G, Edge(nv(G),nv(G)), :weight, 1.0)
                K[i,j] = k
            end
        end
    end
    for i in 1:size(Ap,1)
        for j in 1:size(Ap,2)
            if Ap[i,j] == false
                if j < size(Ap,2)
                    add_edge!(G,K[i,j],K[i,j+1])
                    set_prop!(G, Edge(K[i,j],K[i,j+1]), :weight, 1.0)
                end
                if j > 1
                    add_edge!(G,K[i,j],K[i,j-1])
                    set_prop!(G, Edge(K[i,j],K[i,j-1]), :weight, 1.0)
                end
                if i < size(Ap,1)
                    add_edge!(G,K[i,j],K[i+1,j])
                    set_prop!(G, Edge(K[i,j],K[i+1,j]), :weight, 1.0)
                end
                if i > 1
                    add_edge!(G,K[i,j],K[i-1,j])
                    set_prop!(G, Edge(K[i,j],K[i-1,j]), :weight, 1.0)
                end
                if j < size(Ap,2) && i < size(Ap,1)
                    if (Ap[i,j+1]==false) & (Ap[i+1,j]==false)
                        add_edge!(G,K[i,j],K[i+1,j+1])
                        set_prop!(G, Edge(K[i,j],K[i+1,j+1]), :weight, 1.41)
                    end
                end
                if j > 1 && i > 1
                    if (Ap[i,j-1]==false) & (Ap[i-1,j]==false)
                        add_edge!(G,K[i,j],K[i-1,j-1])
                        set_prop!(G, Edge(K[i,j],K[i-1,j-1]), :weight, 1.41)
                    end
                end
                if j < size(Ap,2) && i > 1
                    if (Ap[i,j+1]==false) & (Ap[i-1,j]==false)
                        add_edge!(G,K[i,j],K[i-1,j+1])
                        set_prop!(G, Edge(K[i,j],K[i-1,j+1]), :weight, 1.41)
                    end
                end
                if j > 1 && i < size(Ap,1)
                    if (Ap[i,j-1]==false) & (Ap[i+1,j]==false)
                        add_edge!(G,K[i,j],K[i+1,j-1])
                        set_prop!(G, Edge(K[i,j],K[i+1,j-1]), :weight, 1.41)
                    end
                end
            end
        end
    end
    return G,K
end


function compute_distance_matrix(graph::G where G)
   D = zeros(Float64,nv(graph),nv(graph))
   for v1 in vertices(graph)
       ds = dijkstra_shortest_paths(graph,v1,Graphs.weights(graph))
       D[v1,:] = ds.dists
       # @infiltrate
   end
   D
end

"""
    Get the distance matrix corresponding to the edge weights of a graph
"""
function get_dist_matrix(G::AbstractGraph)
    distmx = 1000000 .* ones(length(vertices(G)),length(vertices(G)))
    for e in edges(G)
        t_edge = get_prop(G,e, :weight)
        distmx[e.src,e.dst] = t_edge
    end
    for v in vertices(G)
        distmx[v,v] = 0 # We use this heuristic to get shortest path lengths, so not 1
    end

    #Now that the weight matrix is computed, let's find the distmx for the heuristic
    distmx_DP = compute_distance_matrix(G)

    return distmx, distmx_DP
end
