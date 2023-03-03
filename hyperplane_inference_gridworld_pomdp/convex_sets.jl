using Graphs, MetaGraphs
using JuMP, GLPK
using DataStructures
using Infiltrator

"""An obstacle is a polyhedron defined by Ax <= b.
point can be any coordinate located inside the obstacle."""
struct obstacle
    A::Matrix
    b::Array
    point::Array
end

"""A contour defines the bounds of the environment."""
struct contour
    A::Matrix
    b::Array
end

"""Structure representing the hyperplane arrangement"""
struct hyperplane_arrangement
    obstacles::Array{obstacle}
    root_point::Array
    Graph::MetaGraph
    root_node
end

struct waitlisted_polytope
    H_A::Matrix
    H_b::Array
    state::Array
    source_vertex::Int64
    # shared_face_index_source::Int64
    shared_face_index_global::Int64
    flipped_index_in_reduced_source::Int64
end

# Each vertex has a list of those which are possible preferences
struct pref_neighbor
    edge_index::Int64
    idx_in_A::Int64
    neighbor_vertex_id::Int64
end

"""Tests whether a point is located within the bounds"""
function is_within_limits(x::Array, limits::contour)
    # If all the elements of Ax-b are negative, we are inside, else outside
    return all(<=(0), limits.A*x-b)
end

"""Finds reduced representation of a polytope by removing redundant constraints"""
function find_reduced_representation(H_A::Matrix, H_b::Array, num_obstacle_hyperplanes::Int64)
    n=2

    reduced_A = nothing
    reduced_b = nothing
    contour_indices = []
    reduced_idx_dict = Dict() # Mapping from reduced indices to corresponding index in graph

    println("Finding reduced representation")

    # Find which indices are redundant by solving an LP for each
    for i in 1:length(H_b)
        A_tilde = H_A[1:end .!= i,:]
        b_tilde = H_b[1:end .!= i]

        a_i = H_A[i,:]
        b_i = H_b[i]

        # --- is i redundant? ---

        # JuMP optimization model
        model = Model(GLPK.Optimizer)

        @variable(model, x[1:n])
        @constraint(model, A_tilde * x - b_tilde .<= 0)
        @objective(model, Max, a_i' * x)

        optimize!(model)

        obj = objective_value(model)

        # If objective > bi, constraint is not redundant
        if (obj > b_i || Int(termination_status(model))!=1)
            println("Constraint ", i, "is not redundant")
            reduced_A = vcat(reduced_A, a_i')
            reduced_b = vcat(reduced_b, b_i)
            # If we are adding a contour, store the information
            if i > num_obstacle_hyperplanes
                push!(contour_indices, length(reduced_b)-1)
            end
            reduced_idx_dict[length(reduced_b)-1] = i
        end
    end

    return reduced_A[2:end,:], reduced_b[2:end], contour_indices, reduced_idx_dict
end

function get_idx_in_A_from_graph_pref_index(G::MetaGraph, v_idx::Int64, graph_pref_vtx_id::Int64)
    """Gets the index in the local graph relevant to v_idx of the preference corresponding to
    a transition to neighbor with global vertex id graph_pref."""

    neighbors_of_v = get_prop(G,v_idx,:pref_neighbors)

    for n in neighbors_of_v
        if n.neighbor_vertex_id == graph_pref_vtx_id
            return n.idx_in_A
        end
    end

    @show v_idx
    @show graph_pref_vtx_id
    println("(A index query) Didn't find the desired graph_pref in v_idx neighbors!")
    return nothing
end

function get_edge_number_from_graph_pref_index(G::MetaGraph, v_idx::Int64, graph_pref_vtx_id::Int64)
    """Gets the index in the local graph relevant to v_idx of the preference corresponding to
    a transition to neighbor with global vertex id graph_pref."""

    neighbors_of_v = get_prop(G,v_idx,:pref_neighbors)

    for n in neighbors_of_v
        if n.neighbor_vertex_id == graph_pref_vtx_id
            return n.edge_index
        end
    end

    @show v_idx
    @show graph_pref_vtx_id
    println("(edge index query) Didn't find the desired graph_pref in v_idx neighbors!")
    return nothing
end

function get_edge_number_from_index_in_A(G::MetaGraph, v_idx::Int64, idx_in_A::Int64)
    """Gets the index in the local graph relevant to v_idx of the preference corresponding to
    a transition to neighbor with global vertex id graph_pref."""

    neighbors_of_v = get_prop(G,v_idx,:pref_neighbors)

    for n in neighbors_of_v
        if n.idx_in_A == idx_in_A
            return n.edge_index
        end
    end

    @show v_idx
    @show idx_in_A
    println("(edge number from idx_in_A query) Didn't find the desired graph_pref in v_idx neighbors!")
    return nothing
end

function get_pref_from_edge_number(G::MetaGraph, v_idx::Int64, edge_number::Int64)
    """Gets the index in the local graph relevant to v_idx of the preference corresponding to
    a transition to neighbor with global vertex id graph_pref."""

    neighbors_of_v = get_prop(G,v_idx,:pref_neighbors)

    for n in neighbors_of_v
        if n.edge_index == edge_number
            return n
        end
    end

    @show v_idx
    @show edge_number
    println("(edge index query) Didn't find the desired graph_pref in v_idx neighbors!")
    return nothing
end

# struct node_property
#     node_A::Matrix
#     node_b::Array
#     neighbor_list::Array # Ordered list of corresponding neighbor vertices
# end
"""Vertex properties:
root_node (Bool)
A (reduced)
b (reduced)
contour_indices
state   --
mapping -- dict mapping index in b (reduced) to index in list of all hyperplanes
"""

"""Function creating a hyperplane arrangement structure from obstacle inputs"""
function create_hyperplane_arrangement(root_point::Array, obstacles, limits::contour, obstacle_map)

    n=2 # Dimension of the problem

    G = MetaGraph()
    vtx_id = 1 # Keeps track of the ID of the vertex created
               # (can't find any better way to do this in Julia...)
    add_vertex!(G,Dict(:root_node=>true))

    state_to_vertex_dict = Dict() # maps a state to the vertex index

    # This dictionary maps state vectors to vertices in the graph
    vtx_map = Dict()

    # Queue of nodes to add to the graph
    openset = Queue{waitlisted_polytope}()

    obstacle_indices_in_A = Dict()

    # Get global A,b by stacking obstacles
    A = nothing
    b = nothing
    for obs in obstacles
        @show obs
        for i=1:length(obs.b) # Go through each obstacle hyp. to remove duplicates
            # Determine whether the row is duplicate
            row_already_in_A = false
            if ~isnothing(b)

                obs_A_i = obs.A[i,:]
                obs_b_i = obs.b[i]
                for j=2:length(b)
                    # global A
                    # global b
                    # global row_already_in_A
                    if ((obs_A_i == A[j,:] && obs_b_i == b[j]) || (obs_A_i == -1 .* A[j,:] && obs_b_i == -b[j]))
                        row_already_in_A = true
                    end
                end
            end
            # If the row is not a duplicate, add it
            if ~row_already_in_A
                A = vcat(A, obs.A[i,:]')
                b = vcat(b, obs.b[i])
                obs_indices = get(obstacle_indices_in_A, obs, nothing)
                if ~isnothing(obs_indices)
                    obstacle_indices_in_A[obs] = vcat(obs_indices, length(b)-1)
                end
            end
        end
        # A = vcat(A, obs.A)
        # b = vcat(b, obs.b)
    end
    # (debug) remove nothing from beginning of lists?!
    A = A[2:end,:]
    b = b[2:end]

    # track all hyperplanes that are not contours
    num_obstacle_hyperplanes = length(b)

    # Add boundaries
    A = vcat(A, limits.A)
    b = vcat(b, limits.b)

    # track all polytopes that correspond to obstacles
    obstacle_states = []
    for obs in obstacles
        obs_values = A*obs.point-b
        obs_state = [elt <= 0 for elt in obs_values]
        push!(obstacle_states, obs_state)
    end

    # find the correct signs for these for the root point to verify Ax <= b
    root_node_values = A*root_point-b
    root_signs = [elt>0 for elt in root_node_values]

    # State encoding the root node polytope
    H_state = [elt<=0 for elt in root_node_values]
    state_to_vertex_dict[H_state] = vtx_id

    # compute H-representation
    H_A = A - 2*A.*root_signs # Turn lines a of A into -a whenever ax-b > 0
    H_b = b - 2*b.*root_signs # same for b

    # Remove redundant constraints by solving LP (Joe's paper)
    reduced_A, reduced_b, contour_indices, reduced_idx_dict = find_reduced_representation(H_A, H_b, num_obstacle_hyperplanes)

    # Now that we found faces of the polytope, add them to the node properties
    set_prop!(G,vtx_id, :A, reduced_A)
    set_prop!(G,vtx_id, :b, reduced_b)
    set_prop!(G,vtx_id, :contour_indices, contour_indices)
    set_prop!(G,vtx_id, :state, H_state)
    set_prop!(G,vtx_id, :mapping, reduced_idx_dict)
    set_prop!(G,vtx_id, :pref_neighbors, []) # contains neighbor preference info
    vtx_map[H_state] = vtx_id # store vertex in state dictionary


    # Use neighbor to open set to figure out what neighbors to add!
    for i in 1:length(reduced_b)
        if i ∉ contour_indices

            j=reduced_idx_dict[i]
            # create neighbor for waitlist
            neighbor_H_A = copy(H_A)
            neighbor_H_b = copy(H_b)
            neighbor_state = copy(H_state)
            # Flip index j
            neighbor_H_A[j,:] = -neighbor_H_A[j,:]
            neighbor_H_b[j] = -neighbor_H_b[j]
            neighbor_state[j] = ~neighbor_state[j] # way to flip booleans in julia

            # @infiltrate

            point_in_neighbor = get_point_belonging_to_set(10, 10,neighbor_H_A,neighbor_H_b)

            # verify that the neighbor is not an obstacle
            # if neighbor_state ∉ obstacle_states
            if ~obstacle_map[point_in_neighbor[1],point_in_neighbor[2]]
                # Add the neighbor to the queue
                neighbor_polytope = waitlisted_polytope(neighbor_H_A, neighbor_H_b, neighbor_state, vtx_id, j, i)
                enqueue!(openset, neighbor_polytope)
            end
        end
    end

    while ~isempty(openset)

        poly = dequeue!(openset)

        # Does the new vertex already exist?
        if poly.state ∈ collect(keys(state_to_vertex_dict))
           # Get destination vertex
           dest = state_to_vertex_dict[poly.state]

           # Add connecting edge (will not happen if edge already exists)
           add_edge!(G,poly.source_vertex, dest, Dict(:shared_face=>poly.shared_face_index_global))

           # ---------- Set preference info ---------------
           # How many vertices does the source now have? (also index of new edge)
           edge_index = length(get_prop(G,poly.source_vertex,:pref_neighbors)) + 1
           # Index of the source's reduced A that was flipped to get here
           idx_in_A = poly.flipped_index_in_reduced_source
           # Index of the destination
           neighbor_vertex_id = dest

           # Container of the potential preferred neighbors
           pref_data = pref_neighbor(edge_index, idx_in_A, neighbor_vertex_id)

           # Add neighbor to existing list and update vertex property
           d = get_prop(G,poly.source_vertex,:pref_neighbors)
           d = vcat(d,pref_data)
           set_prop!(G,poly.source_vertex,:pref_neighbors, d)
           # -----------------------------------------------

        else
           # Add new vertex to the graph
           vtx_id += 1
           add_vertex!(G, Dict(:root_node=>false))
           state_to_vertex_dict[poly.state]=vtx_id

           # Add connecting edge
           add_edge!(G,poly.source_vertex, vtx_id, Dict(:shared_face=>poly.shared_face_index_global))

           # Get reduced representation of new vertex
           reduced_A, reduced_b, contour_indices, reduced_idx_dict = find_reduced_representation(poly.H_A, poly.H_b, num_obstacle_hyperplanes)

           # Set vertex properties
           set_prop!(G,vtx_id, :A, reduced_A)
           set_prop!(G,vtx_id, :b, reduced_b)
           set_prop!(G,vtx_id, :contour_indices, contour_indices)
           set_prop!(G,vtx_id, :state, poly.state)
           set_prop!(G, vtx_id, :mapping, reduced_idx_dict)
           set_prop!(G,vtx_id, :pref_neighbors, []) # contains neighbor preference info
           vtx_map[poly.state] = vtx_id # store vertex in state dictionary


           # ---------- Set preference info for source ----------------
           # How many vertices does the source now have? (also index of new edge)
           edge_index = length(get_prop(G,poly.source_vertex,:pref_neighbors)) + 1
           # Index of the source's reduced A that was flipped to get here
           idx_in_A = poly.flipped_index_in_reduced_source
           # Index of the destination
           neighbor_vertex_id = vtx_id

           # Container of the potential preferred neighbors
           pref_data = pref_neighbor(edge_index, idx_in_A, neighbor_vertex_id)

           # Add neighbor to existing list and update vertex property
           d = get_prop(G,poly.source_vertex,:pref_neighbors)
           # @show d
           d = vcat(d,pref_data)
           set_prop!(G,poly.source_vertex,:pref_neighbors, d)
           # ----------------------------------------------- -----------


           # Add neighbors into queue
           for i in 1:length(reduced_b)
               if i ∉ contour_indices

                   j=reduced_idx_dict[i]
                   # create neighbor for waitlist
                   neighbor_H_A = copy(poly.H_A)
                   neighbor_H_b = copy(poly.H_b)
                   neighbor_state = copy(poly.state)
                   # Flip index j
                   neighbor_H_A[j,:] = -neighbor_H_A[j,:]
                   neighbor_H_b[j] = -neighbor_H_b[j]
                   neighbor_state[j] = ~neighbor_state[j] # way to flip booleans in julia

                   point_in_neighbor = get_point_belonging_to_set(20, 20,neighbor_H_A,neighbor_H_b)

                   # verify that the neighbor is not an obstacle
                   # if neighbor_state ∉ obstacle_states
                   if ~obstacle_map[point_in_neighbor[1],point_in_neighbor[2]]
                       # Add the neighbor to the queue
                       neighbor_polytope = waitlisted_polytope(neighbor_H_A, neighbor_H_b, neighbor_state, vtx_id, j, i)
                       enqueue!(openset, neighbor_polytope)
                   end
               end
           end
        end
    end
    return G
end

function get_point_belonging_to_set(bdry_x, bdry_y,A,b)
    """set verifies (global_A*x-global_b)"""
    for x=1:bdry_x
        for y=1:bdry_y
            # point_sequence =[elt>0 for elt in global_A*[x,y]-global_b]
            # if point_sequence==global_sequence
            #     return [x,y]
            # end
            if is_in_region(A,b,[x,y])
                return [x,y]
            end
        end
    end
    return nothing
end

# struct waitlisted_polytope
#     H_A::Matrix
#     H_b::Array
#     state::Array
#     source_index::Int64
#     shared_face_index_global::Int64
# end

function test()
    # Obstacle: square containing [5,5], [5,6], [6,5] and [6,6]
    obstacle_A = [1 0
                  -1 0
                  0 1
                  0 -1]
    obstacle_b = [7.0, -5.0, 7.0, -5.0]
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

# test()
