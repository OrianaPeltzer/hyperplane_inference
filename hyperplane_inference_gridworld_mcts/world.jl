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
    # intention_index::Int64 # This is the index of the preferred neighbor in the current region
    intentions::Array{Int64} # List of preference for all regions
end

# We'll need a separate struct to explicitly represent the belief state
# (rather than implicitly as a distribution over states, as that would be intractable)
# We'll need to define the belief updater functions
@with_kw struct GridBeliefState
    position::GridPosition
    done::Bool # are we in a terminal state?
    neighbor_A::Matrix # A*current_state - b <= 0
    neighbor_b::Array
    goal_options::Array{GridPosition}
    hyperplane_graph::MetaGraph
    belief_intention::Matrix{Float64} # size #goals*(1+#neighbors)
    preference_marginals::Dict # mapping vertex id to belief over preferences
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

function heading_to_angle(a::Char)

    float_pi = 3.141592653589793

    if a == 'W'
        return float_pi
    elseif a == 'E'
        return 0.0
    elseif a == 'S'
        return -float_pi/2
    elseif a == 'N'
        return float_pi/2
    elseif a == '0' # SW
        return -3*float_pi/4
    elseif a == '2' # NW
        return 3*float_pi/4
    elseif a == '1' # SE
        return -float_pi/4
    elseif a == '3' # NE
        return float_pi/4
    else
        println("Invalid heading!!!!!!!!")
        return None
    end

end



"""Struct for true map of the world"""
# The POMDP should contain the true map - remember it is your nature hat
@with_kw struct MapWorld <: POMDPs.MDP{GridState,Char}
    start_states::Array{GridState}    = []
    start_state_index::Int64          = 1
    obstacle_map::Matrix{Bool}        = fill(false, 10, 10)
    grid_side::Int64                  = 10
    discount_factor::Float64          = 0.94
    discount::Float64                 = 0.94
    penalty::Float64                  = -1.0
    diag_penalty::Float64             = -1.414 # sqrt(2)
    incorrect_transition_penalty::Float64 = -5.0
    correct_transition_reward::Float64 = 5.5
    reward::Float64                   = 100.0
    new_obs_weight::Float64           = 0.75
    goal_options::Array{GridPosition} = [GridPosition(7, 7),
                                         GridPosition(1, 8)]
    # goal::GridPosition                = GridPosition(3, 3)
    n_possible_goals::Int64           = 2
    human_input_std::Float64          = 0.5
    map_graph::MetaGraph              = MetaGraph()
    K_map::Matrix{Int}                = fill(0, 1, 1)
    dist_matrix::Matrix{Float64}      = fill(0, 1, 1)
    dist_matrix_DP::Matrix{Float64}   = fill(0, 1, 1)
    BII_gamma::Float64                = 0.8

    hyperplane_graph::MetaGraph       = MetaGraph()
    # True start, goal and preferences are kept here
    start_position::GridPosition      = GridPosition(1,1)
    true_goal_index::Int64            = 1
    true_preference::Matrix{Int}      = fill(0, 1, 1)
end


"""Struct for true map of the world without belief"""
# The POMDP should contain the true map - remember it is your nature hat
@with_kw struct MapWorldSetup
    obstacle_map::Matrix{Bool}        = fill(false, 10, 10)
    grid_side::Int64                  = 10
    discount_factor::Float64          = 0.94
    discount::Float64                 = 0.94
    penalty::Float64                  = -1.0
    diag_penalty::Float64             = -1.414 # sqrt(2)
    incorrect_transition_penalty::Float64 = -5.0
    correct_transition_reward::Float64 = 5.5
    reward::Float64                   = 100.0
    new_obs_weight::Float64           = 0.75
    goal_options::Array{GridPosition} = [GridPosition(7, 7),
                                         GridPosition(1, 8)]
    # goal::GridPosition                = GridPosition(3, 3)
    n_possible_goals::Int64           = 2
    human_input_std::Float64          = 0.5
    map_graph::MetaGraph              = MetaGraph()
    K_map::Matrix{Int}                = fill(0, 1, 1)
    dist_matrix::Matrix{Float64}      = fill(0, 1, 1)
    dist_matrix_DP::Matrix{Float64}   = fill(0, 1, 1)
    BII_gamma::Float64                = 0.8

    hyperplane_graph::MetaGraph       = MetaGraph()
    # True start, goal and preferences are kept here
    start_position::GridPosition      = GridPosition(1,1)
    true_goal_index::Int64            = 1
    true_preference::Matrix{Int}      = fill(0, 1, 1)
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

function violates_constraint(g::MetaGraph, constraint::NeighborConstraint, v::Int64, u::Int64) # replaced path with u

    # if length(path) == 0
    #     return false
    # end

    if !constraint.enforced
        return false
    end
    # x0 = get_prop(g,get_final_node(path),:x)
    # y0 = get_prop(g,get_final_node(path),:y)
    x0 = get_prop(g,u,:x)
    y0 = get_prop(g,u,:y)
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
function initial_belief_state(pomdp::MapWorldSetup)

    # Starting state for the robot
    start_state = pomdp.start_position

    start_A, start_b = pos_to_neighbor_matrices(start_state, pomdp.hyperplane_graph)

    belief_intention = initialize_prior_belief(pomdp, start_state, start_A, start_b)

    # Dict mapping each vertex id to a vector of preference marginals
    G = pomdp.hyperplane_graph
    preference_marginals = Dict()

    for v in vertices(G)
        num_prefs = length(get_prop(G,v,:pref_neighbors))
        preference_marginals[v] = fill(1.0/num_prefs, num_prefs) # Uniform prior
    end

    return GridBeliefState(start_state, false, start_A, start_b, pomdp.goal_options, pomdp.hyperplane_graph, belief_intention, preference_marginals)
end


"""(Problem Specific) Function to get initial state"""
function true_initial_state(pomdp::MapWorldSetup)

    # Starting state for the robot
    start_state = pomdp.start_position

    start_A, start_b = pos_to_neighbor_matrices(start_state, pomdp.hyperplane_graph)

    # i = pos_to_region_index(start_state, pomdp.hyperplane_graph)

    G = pomdp.hyperplane_graph
    global_prefs = pomdp.true_preference
    preference_indices = []

    for (i,gp) in enumerate(global_prefs)
        if gp == -1 # means the goal is in the region
            num_admissible_neighbors = length(get_prop(G,i,:pref_neighbors))
            preference_indices = vcat(preference_indices, num_admissible_neighbors + 1)
        else
            pref_index = get_edge_number_from_graph_pref_index(G, i, gp)
            preference_indices = vcat(preference_indices, pref_index)
        end
    end

    return GridState(start_state, false, pomdp.true_goal_index,
                     start_A, start_b, preference_indices)
end



function initialize_prior_belief(p::MapWorldSetup,pos::GridPosition,A::Matrix,b::Vector)
    # Uniform prior on which goal is the intended one
    G = p.hyperplane_graph
    vtx_id = pos_to_region_index(pos, G)

    num_neighbors = length(get_prop(G,vtx_id,:pref_neighbors))
    belief_intention = zeros(p.n_possible_goals, 1+num_neighbors)

    # Construct prior belief for each goal
    for (g_idx, goal) in enumerate(p.goal_options)
        # rows where the goal belong to the region ([0.0, 0.0, 1/N])
        if is_in_region(A, b, goal)
            belief_intention[g_idx, num_neighbors+1] = 1.0/p.n_possible_goals
        # rows where the goal is outside
        else
            shortest_path_distance = get_shortest_path_length(p, pos, goal)
            for pref=1:num_neighbors
                # Previous version: shortest path calculation to get p(theta|g).
                # pos_index = p.K_map[pos[1], pos[2]]
                # goal_index = p.K_map[goal[1], goal[2]]
                #
                # preference_constraint = NeighborConstraint(A,b,pref,true)
                #
                # shortest_path_constrained = Graphs.a_star(p.map_graph, pos_index, goal_index, preference_constraint, p.dist_matrix_DP)
                #
                # constrained_dist = traversal_time(shortest_path_constrained, p.map_graph)
                # diff = constrained_dist - shortest_path_distance
                # belief_intention[g_idx,pref] = exp(-0.1*diff)

                # New version: uniform distribution
                belief_intention[g_idx,pref] = 1.0
            end
            # normalize prior and divide by n_possible_goals (sum for each row should be 1/N)
            belief_intention[g_idx,:] = belief_intention[g_idx,:] / (sum(belief_intention[g_idx,:]) * p.n_possible_goals)
        end
    end
    return belief_intention
end

"""Re-construct prior belief after transition in between regions/sets"""
function reshape_prior_belief(p::MapWorld,pos::GridPosition,new_pos::GridPosition, A::Matrix,b::Vector, bel::GridBeliefState)
    """A and b are the matrices corresponding to the new position."""

    G = p.hyperplane_graph
    old_vtx_id = pos_to_region_index(pos, G)
    new_vtx_id = pos_to_region_index(new_pos, G)
    new_num_neighbors = length(get_prop(G,new_vtx_id,:pref_neighbors))

    old_A = get_prop(G,old_vtx_id,:A)
    old_b = get_prop(G,old_vtx_id,:b)


    # Initialize new belief
    belief_intention = zeros(p.n_possible_goals, 1+new_num_neighbors)

    # Collect marginals over goal
    marginals = sum(bel.belief_intention, dims=2)

    # Find the probabilities of doing the transition that it actually did
    sequence = old_A*(-0.1 .+ new_pos)-old_b
    transition_index = findfirst(.>(0),sequence)

    # Find the index of the edge it just transitioned to
    transition_edge_number = get_edge_number_from_index_in_A(G, old_vtx_id, transition_index)

    p_trans = sum(bel.belief_intention[:,transition_edge_number]) # marginalized over goals


    # Construct prior belief for each goal
    for (g_idx, goal) in enumerate(p.goal_options)
        if is_in_region(A, b, goal)
            belief_intention[g_idx, new_num_neighbors+1] = marginals[g_idx]#1.0/p.n_possible_goals
        else
            # shortest_path_distance = get_shortest_path_length(p, new_pos, goal)
            for pref=1:new_num_neighbors

                # Get the neighbor that this corresponds to (with edge number pref)
                pref_neighbor = get_pref_from_edge_number(G, new_vtx_id, pref)

                # Uniform prior
                # belief_intention[g_idx,pref] = 1.0#exp(-0.1*diff)

                # Using marginals in case we already visited region
                belief_intention[g_idx,pref] = bel.preference_marginals[new_vtx_id][pref]*new_num_neighbors


                # If the preference corresponds to where we come from, use prior
                if pref_neighbor.neighbor_vertex_id == old_vtx_id
                    belief_intention[g_idx,pref] = 1.0 - p_trans
                end

                # if sequence[pref] > 0
                #     belief_intention[g_idx,pref] = 1.0 - p_trans[g_idx]
                # end

                # This shouldn't happen
                # if is_obstacle_index(new_pos,pref)
                if pos_to_region_index(new_pos, G) == -1 # If infeasible new pos
                    println("Robot moved inside an obstacle?")
                    @show new_pos
                    @show pref
                    belief_intention[g_idx,pref] = 0.0
                end

                # Is there a valid path to the goal going through the new pref?
                pos_index = p.K_map[new_pos[1], new_pos[2]]
                goal_index = p.K_map[goal[1], goal[2]]
                preference_constraint = NeighborConstraint(A,b,pref_neighbor.idx_in_A,true)
                shortest_path_constrained = Graphs.a_star(p.map_graph, pos_index, goal_index, preference_constraint, p.dist_matrix_DP)
                # If pref leads to nowhere, set belief to 0.
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


function sample_human_action(p::MapWorld, s::GridState, rng::AbstractRNG)
    actions = ['N', 'S', 'E', 'W', '0', '1', '2', '3']
    measurement_likelihoods = fill(0.0,length(actions))
    goal_in_region = is_in_region(s.neighbor_A, s.neighbor_b, p.goal_options[p.true_goal_index])
    current_vtx_id = pos_to_region_index(s.position, p.hyperplane_graph)
    for (i,a) in enumerate(actions)
        if goal_in_region
            measurement_likelihoods[i] = compute_measurement_likelihood(p,s.position,p.goal_options[p.true_goal_index],a)
        else
            measurement_likelihoods[i] = compute_measurement_likelihood(p,
                                 s.position,
                                 p.goal_options[p.true_goal_index],
                                 a,
                                 s.neighbor_A, s.neighbor_b, s.intentions[current_vtx_id])
        end
    end
    measurement_likelihoods = measurement_likelihoods ./ sum(measurement_likelihoods)
    action_index = findfirst(cumsum(measurement_likelihoods) .>= rand(rng))
    heading = actions[action_index]
    return heading_to_angle(heading)
end


function compute_measurement_likelihood(p::MapWorld, pos::GridPosition, goal::GridPosition, heading::Char)

    distance_to_goal = get_shortest_path_length(p,pos,goal)

    human_intended_position = update_position(p,pos,heading)

    distance_to_intended_position = get_shortest_path_length(p, pos, human_intended_position)

    if distance_to_intended_position==0 # The human banged against the wall
        wall_banging_penalty=4 # The human gets extra penalty for its damage
        return exp(-p.BII_gamma * wall_banging_penalty)
    end

    distance_to_goal_complying = distance_to_intended_position + get_shortest_path_length(p, human_intended_position, goal)

    diff = distance_to_goal_complying - distance_to_goal

    return exp(-p.BII_gamma * diff)
end

function compute_measurement_likelihood(p::MapWorld, pos::GridPosition,
                                        goal::GridPosition, heading::Char,
                                        neighbor_A::Matrix, neighbor_b::Vector,
                                        preference_index::Int64)
    """Returns the probability that the human created observation "heading"
    given the tuple (goal, neighbor_preference)."""

    # println("Entered compute measurement likelihood function")
    # println("")
    # @show heading
    # @show pos
    # @show goal
    # @show preference_index


    # 1. Set up preference constraint ----------------------------------

    # Preference index is the edge number. To turn that in a constraint,
    # get the corresponding index in A that should be flipped.

    G = p.hyperplane_graph
    v_idx = pos_to_region_index(pos, G)

    # preference_index = intentions[v_idx]

    pref = get_pref_from_edge_number(G, v_idx, preference_index)

    # Create preference constraint
    preference_constraint = NeighborConstraint(neighbor_A,neighbor_b,
                                               pref.idx_in_A,true)



    # 2. Get dist(pos->intended position) ------------------------------

    # Map observation to intended future state l_t
    human_intended_position = update_position(p,pos,heading)
    # println("Found human intended position: ", human_intended_position)

    distance_to_intended_position = get_shortest_path_length(
                                   p, pos, human_intended_position)

    if distance_to_intended_position==0 # The human banged against the wall
       wall_banging_penalty=4 # The human gets extra penalty for its damage
       return exp(-p.BII_gamma * wall_banging_penalty)
    end


    # 3. get delta(pos->goal|preference) -------------------------------------

    # Get vertex indices on the mapgraph corresponding to current pos and goal
    pos_index = p.K_map[pos[1], pos[2]]
    goal_index = p.K_map[goal[1], goal[2]]

    # Compute shortest path from pos to goal constrained by preference
    path_to_goal = Graphs.a_star(p.map_graph, pos_index, goal_index,
                                 preference_constraint, p.dist_matrix_DP)
    # @show path_to_goal
    # @show [[get_prop(p.map_graph, e.src, :x), get_prop(p.map_graph, e.src, :y)] for e in path_to_goal]
    distance_to_goal = traversal_time(path_to_goal, p.map_graph)
    # @show distance_to_goal


    # 3. Did the human intend to change region? -------------------------------

    # Is the human's intended location in a different region?
    # same_region = all(<=(0), neighbor_A*human_intended_position-neighbor_b)
    same_region = is_in_region(neighbor_A, neighbor_b, human_intended_position)


    # 4. Compute delta(pos->goal|preference) complying with intention ---------

    if same_region==0 # Different region!
        # If the human wanted to transition to another region passing by lt,
        # they would need to backtrack back into the region first.
        # likelihood propto 2*dist(pos,intention) + delta(pos->goal|preference)
        distance_to_goal_complying = 2*distance_to_intended_position + distance_to_goal
    else
        # Get index in the mapgraph of the intended pose
        human_intended_index = p.K_map[human_intended_position[1], human_intended_position[2]]

        # @show human_intended_index

        path_to_goal_from_intended_index = Graphs.a_star(p.map_graph,
                                            human_intended_index, goal_index,
                                            preference_constraint,
                                            p.dist_matrix_DP)

        # @show path_to_goal_from_intended_index
        # @show [[get_prop(p.map_graph, e.src, :x), get_prop(p.map_graph, e.src, :y)] for e in path_to_goal_from_intended_index]

        distance_to_goal_from_intended_index = traversal_time(path_to_goal_from_intended_index, p.map_graph)
        # @show distance_to_goal_from_intended_index

        distance_to_goal_complying = distance_to_intended_position + distance_to_goal_from_intended_index
    end

    # @show distance_to_goal_complying

    diff = distance_to_goal_complying - distance_to_goal

    return exp(-p.BII_gamma * diff)
end


function update_position(p::MapWorld, pos::GridPosition, a::Char)

    curr_pos = pos

    # if a == 'S'
    #     new_pos = GridPosition(max(1, curr_pos[1]-1), curr_pos[2])
    # elseif a == 'N'
    #     new_pos = GridPosition(min(p.grid_side, curr_pos[1]+1), curr_pos[2])
    # elseif a == 'W'
    #     new_pos = GridPosition(curr_pos[1], max(1, curr_pos[2]-1))
    # elseif a == 'E'
    #     new_pos = GridPosition(curr_pos[1], min(p.grid_side, curr_pos[2]+1))
    # elseif a == '0'
    #     new_pos = GridPosition(max(1, curr_pos[1]-1), max(1, curr_pos[2]-1))
    # elseif a == '1'
    #     new_pos = GridPosition(max(1, curr_pos[1]-1), min(p.grid_side, curr_pos[2]+1))
    # elseif a == '2'
    #     new_pos = GridPosition(min(p.grid_side, curr_pos[1]+1), max(1, curr_pos[2]-1))
    # elseif a == '3'
    #     new_pos = GridPosition(min(p.grid_side, curr_pos[1]+1), min(p.grid_side, curr_pos[2]+1))
    # else
    #     new_pos = curr_pos
    # end

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

    # The robot is also not allowed to cut through polytope corners
    G = p.hyperplane_graph
    curr_v_idx = curr_v_idx = pos_to_region_index(pos, G)
    A = get_prop(G,curr_v_idx,:A)
    b = get_prop(G,curr_v_idx,:b)
    same_region = is_in_region(A, b, new_pos)
    if ~same_region
        sequence = A*(-0.1 .+ new_pos)-b
        num_jumps = count(>(0),sequence)
        transition_A_index = findfirst(.>(0),sequence)[1]

        pref_neighbors = get_prop(G, curr_v_idx, :pref_neighbors)
        admissible_A_transition_indices = [pn.idx_in_A for pn in pref_neighbors]

        if (num_jumps != 1) || ~(transition_A_index in admissible_A_transition_indices)
            new_pos = curr_pos
        end
    end

    return new_pos
end

function get_shortest_path_length(p::Union{MapWorld, MapWorldSetup}, start::GridPosition,finish::GridPosition)
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
                    Dict(:x=>i,
                    :y=>j)
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

# Previous version before changing definition of coordinates
# function initialize_grid_graph(obstacle_map::Matrix{Bool})
#     """Returns a grid graph that represents a 2D environment defined by the
#         obstacle map matrix.
#         Considers that the nominal time to move from one edge to another is 1 if
#         the robot moves right, left, up or down, and 1.41 for movements along
#         the diagonals.
#         The output is a MetaGraph"""
#     Ap = obstacle_map
#     K = zeros(Int,size(Ap))
#     G = MetaGraph()
#
#     k = 0
#     for i in 1:size(Ap,1)
#         for j in 1:size(Ap,2)
#             if Ap[i,j] == false
#                 k += 1
#                 add_vertex!(G,
#                     Dict(:x=>(i-1),
#                     :y=>(j-1))
#                     )
#                 add_edge!(G,nv(G),nv(G))
#                 set_prop!(G, Edge(nv(G),nv(G)), :weight, 1.0)
#                 K[i,j] = k
#             end
#         end
#     end
#     for i in 1:size(Ap,1)
#         for j in 1:size(Ap,2)
#             if Ap[i,j] == false
#                 if j < size(Ap,2)
#                     add_edge!(G,K[i,j],K[i,j+1])
#                     set_prop!(G, Edge(K[i,j],K[i,j+1]), :weight, 1.0)
#                 end
#                 if j > 1
#                     add_edge!(G,K[i,j],K[i,j-1])
#                     set_prop!(G, Edge(K[i,j],K[i,j-1]), :weight, 1.0)
#                 end
#                 if i < size(Ap,1)
#                     add_edge!(G,K[i,j],K[i+1,j])
#                     set_prop!(G, Edge(K[i,j],K[i+1,j]), :weight, 1.0)
#                 end
#                 if i > 1
#                     add_edge!(G,K[i,j],K[i-1,j])
#                     set_prop!(G, Edge(K[i,j],K[i-1,j]), :weight, 1.0)
#                 end
#                 if j < size(Ap,2) && i < size(Ap,1)
#                     if (Ap[i,j+1]==false) & (Ap[i+1,j]==false)
#                         add_edge!(G,K[i,j],K[i+1,j+1])
#                         set_prop!(G, Edge(K[i,j],K[i+1,j+1]), :weight, 1.41)
#                     end
#                 end
#                 if j > 1 && i > 1
#                     if (Ap[i,j-1]==false) & (Ap[i-1,j]==false)
#                         add_edge!(G,K[i,j],K[i-1,j-1])
#                         set_prop!(G, Edge(K[i,j],K[i-1,j-1]), :weight, 1.41)
#                     end
#                 end
#                 if j < size(Ap,2) && i > 1
#                     if (Ap[i,j+1]==false) & (Ap[i-1,j]==false)
#                         add_edge!(G,K[i,j],K[i-1,j+1])
#                         set_prop!(G, Edge(K[i,j],K[i-1,j+1]), :weight, 1.41)
#                     end
#                 end
#                 if j > 1 && i < size(Ap,1)
#                     if (Ap[i,j-1]==false) & (Ap[i+1,j]==false)
#                         add_edge!(G,K[i,j],K[i+1,j-1])
#                         set_prop!(G, Edge(K[i,j],K[i+1,j-1]), :weight, 1.41)
#                     end
#                 end
#             end
#         end
#     end
#     return G,K
# end


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
