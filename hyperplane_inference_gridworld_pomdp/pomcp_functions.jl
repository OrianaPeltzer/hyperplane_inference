"""Sample random belief state"""
function Base.rand(rng::AbstractRNG, bs::GridBeliefState)
    # We need to _sample_ state instances from a belief state

    # Just copy over position (observable)
    pos = bs.position
    done = bs.done

    # get marginals
    belief_goals = sum(bs.belief_intention, dims=2)
    # Sample a new goal from probabilities
    goal_index = findfirst(cumsum(belief_goals, dims=1) .>= rand(rng))[1]
    # goal_index = sample(1:length(belief_goals),Weights([elt[i] for elt=1:]))

    goal_position = bs.goal_options[goal_index]

    G = bs.hyperplane_graph
    index_of_current_region = pos_to_region_index(bs.position, G)
    prefs = []

    for v in vertices(G)
        A = get_prop(G,v,:A)
        b = get_prop(G,v,:b)
        num_neighbors = length(get_prop(G,v,:pref_neighbors))

        # If the goal is in the set, no need to sample the pref
        if is_in_region(A,b,goal_position)
            prefs = vcat(prefs, num_neighbors+1)
        # If current region, sample from the latest belief dist
        elseif v == index_of_current_region
            w = bs.belief_intention[goal_index,:] ./ sum(bs.belief_intention[goal_index,:])
            sample_pref = sample(1:length(bs.belief_intention[goal_index,:]), Weights(w))
            prefs = vcat(prefs, sample_pref)
        else
            # sample from the marginals
            marginals = bs.preference_marginals[v]
            w = marginals ./ sum(marginals)
            sample_pref = sample(1:length(marginals), Weights(w))
            prefs = vcat(prefs, sample_pref)
        end
    end


    # # Sample a new preference from the goal and position
    # w = bs.belief_intention[goal_index,:] ./ sum(bs.belief_intention[goal_index,:])
    # pref = sample(1:length(bs.belief_intention[goal_index,:]), Weights(w))

    return GridState(pos, done, goal_index, bs.neighbor_A, bs.neighbor_b, prefs)
end

"""Sample random belief state"""
function Base.rand(rng::AbstractRNG, bs::GridBeliefStateGoal)
    # We need to _sample_ state instances from a belief state

    # Just copy over position (observable)
    pos = bs.position
    done = bs.done

    # get marginals
    belief_goals = bs.belief_intention
    # Sample a new goal from probabilities
    # goal_index = findfirst(cumsum(belief_goals, dims=1) .>= rand(rng))[1]
    goal_index = sample(1:length(belief_goals),Weights(belief_goals))

    goal_position = bs.goal_options[goal_index]

    G = bs.hyperplane_graph
    index_of_current_region = pos_to_region_index(bs.position, G)
    prefs = []

    for v in vertices(G)
        A = get_prop(G,v,:A)
        b = get_prop(G,v,:b)
        num_neighbors = length(get_prop(G,v,:pref_neighbors))

        # If the goal is in the set, no need to sample the pref
        if is_in_region(A,b,goal_position)
            prefs = vcat(prefs, num_neighbors+1)
        # If current region, sample from the latest belief dist
        elseif v == index_of_current_region
            w = fill(1.0/num_neighbors, num_neighbors)
            sample_pref = sample(1:num_neighbors, Weights(w))
            prefs = vcat(prefs, sample_pref)
        else
            # sample from the marginals
            w = fill(1.0/num_neighbors, num_neighbors)
            sample_pref = sample(1:num_neighbors, Weights(w))
            prefs = vcat(prefs, sample_pref)
        end
    end


    # # Sample a new preference from the goal and position
    # w = bs.belief_intention[goal_index,:] ./ sum(bs.belief_intention[goal_index,:])
    # pref = sample(1:length(bs.belief_intention[goal_index,:]), Weights(w))

    return GridState(pos, done, goal_index, bs.neighbor_A, bs.neighbor_b, prefs)
end




POMDPs.actions(pomdp::MapWorld) = ['N', 'S', 'E', 'W', '0', '1', '2', '3']
POMDPs.discount(pomdp::MapWorld) = pomdp.discount_factor

# For now set isterminal to false and use your own outer condition to play around
POMDPs.isterminal(pomdp::MapWorld) = false

# NOTE: Simple boilerplate stuff for rollouts (ignore now)
struct MapWorldBelUpdater <: Updater
    pomdp::MapWorld
end

function POMDPs.update(updater::MapWorldBelUpdater, b::GridBeliefState, a::Char, o::HumanInputObservation)
    b_updated_goal = update_goal_belief(updater.pomdp, b, o)
    return update_pose_belief(updater.pomdp, b_updated_goal, a) # no observation in this model!
end

function POMDPs.update(updater::MapWorldBelUpdater, b::GridBeliefStateGoal, a::Char, o::HumanInputObservation)
    b_updated_goal = update_goal_belief(updater.pomdp, b, o)
    return update_pose_belief(updater.pomdp, b_updated_goal, a) # no observation in this model!
end

function POMDPs.initialize_belief(updater, d)
    return initial_belief_state(updater.pomdp)
end


# Finally define your generative model
function POMDPs.gen(p::MapWorld, s::GridState, a::Char, rng::AbstractRNG=Random.GLOBAL_RNG)
    """generative model
    Receives the current state and world map, action taken (and random rng)
    Returns a new state, a reward and an observation
    """

    curr_pos = s.position

    G = p.hyperplane_graph
    curr_v_idx = pos_to_region_index(curr_pos, G)

    goal = p.goal_options[s.goal_index]

    # We suppose the human won't give any more observations
    o = HumanInputObservation(0.0,false,false)


    # Get reward if we reached the goal, otherwise get penalty
    if curr_pos == goal
        r = p.reward
        new_state = GridState(curr_pos, true, s.goal_index, s.neighbor_A,
                              s.neighbor_b, s.intentions)
        return (sp=new_state,r=r,o=o)
    elseif curr_pos in p.goal_options # We know if we visited the wrong goal
        o = HumanInputObservation(0.0,false,true)
    end


    # We did not reach the goal, so we get a penalty
    r = p.penalty

    # Set robot position, and get reward or penalty based on true state of next cell
    # I've implicitly bounded in grid here


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
        r = p.diag_penalty
    elseif a == '2' # NW
        new_pos = GridPosition(max(1, curr_pos[1]-1), min(p.grid_side, curr_pos[2]+1))
        r = p.diag_penalty
    elseif a == '1' # SE
        new_pos = GridPosition(min(p.grid_side, curr_pos[1]+1), max(1, curr_pos[2]-1))
        r = p.diag_penalty
    elseif a == '3' # NE
        new_pos = GridPosition(min(p.grid_side, curr_pos[1]+1), min(p.grid_side, curr_pos[2]+1))
        r = p.diag_penalty
    else
        new_pos = curr_pos
    end

    # # Assuming FIRST coordinate is y, i.e. rows number
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
    #     r = p.diag_penalty
    # elseif a == '1'
    #     new_pos = GridPosition(max(1, curr_pos[1]-1), min(p.grid_side, curr_pos[2]+1))
    #     r = p.diag_penalty
    # elseif a == '2'
    #     new_pos = GridPosition(min(p.grid_side, curr_pos[1]+1), max(1, curr_pos[2]-1))
    #     r = p.diag_penalty
    # elseif a == '3'
    #     new_pos = GridPosition(min(p.grid_side, curr_pos[1]+1), min(p.grid_side, curr_pos[2]+1))
    #     r = p.diag_penalty
    # else
    #     new_pos = curr_pos
    # end

    # If we hit an obstacle, we don't actually get to move
    if p.obstacle_map[new_pos...] == true
        new_pos = curr_pos
        r = p.penalty # If we failed to do a diagonal mvt, set the penalty back
    end

    # If the new state is the goal, also get reward straight away
    if new_pos == goal
        r = p.reward
        new_state = GridState(new_pos, true, s.goal_index, s.neighbor_A,
                              s.neighbor_b, s.intentions)
        return (sp=new_state,r=r,o=o)
    end

    # The robot is also not allowed to cut through polytope corners
    same_region = is_in_region(s.neighbor_A, s.neighbor_b, new_pos)
    if ~same_region
        sequence = s.neighbor_A*(-0.1 .+ new_pos)-s.neighbor_b
        # @show s.neighbor_A
        # @show s.neighbor_b
        # @show new_pos
        # @show sequence
        num_jumps = count(>(0),sequence)
        transition_A_index = findfirst(.>(0),sequence)[1]

        pref_neighbors = get_prop(G, curr_v_idx, :pref_neighbors)
        admissible_A_transition_indices = [pn.idx_in_A for pn in pref_neighbors]

        if (num_jumps != 1) || ~(transition_A_index in admissible_A_transition_indices)
            new_pos = curr_pos
            same_region = true
            r = p.penalty # If we failed to do a diagonal mvt, set the penalty back
        end
    end


    # Case where the new position is in a new set: transition penalty
    if ~same_region
        sequence = s.neighbor_A*(-0.1 .+ new_pos)-s.neighbor_b

        # @show s.neighbor_A
        # @show s.neighbor_b
        # @show new_pos
        # @show sequence

        transition_A_index = findfirst(.>(0),sequence)[1]

        # if curr_v_idx==3 && transition_A_index==1
        #     @show curr_pos
        #     @show new_pos
        # end

        transition_edge_num = get_edge_number_from_index_in_A(G, curr_v_idx, transition_A_index)


        if transition_edge_num != s.intentions[curr_v_idx]
            r += p.incorrect_transition_penalty
        else
            r += p.correct_transition_reward
        end
    end

    # If we changed region, A, b and the true preference must change
    if ~same_region
        # @show new_pos
        # find new A, b
        new_A, new_b = pos_to_neighbor_matrices(new_pos, p.hyperplane_graph)

        # @show new_A
        # @show new_b

        # # find new preference
        # new_node_index = pos_to_region_index(new_pos, p.hyperplane_graph)
        # # @show new_node_index
        #
        # # MISTAKE!!! NEED TO GET THAT FROM THE STATE!
        # # new_pref_index_global = p.true_preference[new_node_index]
        # new_pref_index_global = s.intentions[new_node_index]
        # # @show new_pref_index_global
        #
        # if new_pref_index_global == -1 # means the goal is in the region
        #     num_admissible_neighbors = length(get_prop(G,new_node_index,:pref_neighbors))
        #     new_pref_index = num_admissible_neighbors + 1
        # else
        #     new_pref_index = get_edge_number_from_graph_pref_index(G,
        #                                       new_node_index, new_pref_index_global)
        # end


        new_state = GridState(new_pos, false, s.goal_index, new_A,
                              new_b, s.intentions)
    else
        new_state = GridState(new_pos, false, s.goal_index, s.neighbor_A,
                              s.neighbor_b, s.intentions)
    end


    return (sp=new_state, r=r, o=o)
end

"""Update belief with received observation
Currently not used - we update pose and goal belief separately instead"""
# function update_belief(pomdp::MapWorld, b::GridBeliefState, a::Char, o::HumanInputObservation)
#
#     # Just copy over done but update position
#     pos = update_position(pomdp, b.position, a)
#     done = b.done
#
#     changed = all(<=(0), b.neighbor_A*pos-b.neighbor_b)
#
#
#
#     # If region changed, reconstruct belief from marginals
#     if changed==1
#         n_A, n_b = pos_to_neighbor_matrices(pos)
#         prior_belief = GridBeliefState(pos, done, n_A, n_b, reshape_prior_belief(pomdp,b.position,pos,n_A,n_b, b))
#     else
#         prior_belief = b
#     end
#
#
#
#     # now only update belief if you received an observation
#     if o.received_observation == true
#
#         measurement_likelihoods = zeros(pomdp.n_possible_goals, length(prior_belief.neighbor_b)+1)
#
#         # Go through all goals and update one by one
#         for goal_id=1:pomdp.n_possible_goals
#
#             goal = pomdp.goal_options[goal_id]
#             if is_in_region(prior_belief.neighbor_A, prior_belief.neighbor_b,goal)
#
#                 # update measurement likelihood with no constraints (previous method)
#                 measurement_likelihoods[goal_id,length(prior_belief.neighbor_b)+1] = compute_measurement_likelihood(pomdp,pos,goal,o.heading)
#             else
#                 # use preference constraints
#                 for pref=1:length(b)
#                     measurement_likelihoods[goal_id,pref] = compute_measurement_likelihood(pomdp,pos,goal,o.heading, prior_belief.neighbor_A, prior_belief.neighbor_b, pref)
#                 end
#             end
#         end
#         # @show measurement_likelihoods
#
#         # Update prior - not normalized yet
#         new_belief_intentions = measurement_likelihoods .* prior_belief.belief_goals
#
#     else
#         return GridBeliefState(pos, done, prior_belief.neighbor_A, prior_belief.neighbor_b, prior_belief.belief_goals)
#     end
#
#
#     # If we visited a hypothetical goal and it wasn't one, our belief goes to 0
#     if (o.visited_wrong_goal == true) & (pos ∈ pomdp.goal_options)
#         println("Visited wrong goal: now updating belief and setting p to 0")
#         wrong_goal_id = findfirst(isequal(pos), pomdp.goal_options)
#         new_belief_intentions[wrong_goal_id,:] .= 0.0
#     end
#
#
#     # Normalize here!
#     new_belief_intentions = new_belief_intentions/sum(new_belief_intentions)
#
#     return GridBeliefState(pos, done, prior_belief.neighbor_A, prior_belief.neighbor_b, new_belief_intentions)
# end

"""Update belief with received observation"""
function update_goal_belief(pomdp::MapWorld, b::GridBeliefState, o::HumanInputObservation)

    # Just copy over
    pos = b.position
    done = b.done

    prior_belief = b

    # now only update belief if you received an observation
    if o.received_observation == true


        n_goals, n_prefs = size(prior_belief.belief_intention)

        measurement_likelihoods = fill(0.0, n_goals, n_prefs)
        #zeros(pomdp.n_possible_goals, length(prior_belief.neighbor_b)+1)

        # Go through all goals and update one by one
        for goal_id=1:n_goals

            goal = pomdp.goal_options[goal_id]

            if is_in_region(prior_belief.neighbor_A, prior_belief.neighbor_b, goal)

                # update measurement likelihood with no constraints (previous method)
                measurement_likelihoods[goal_id,n_prefs] = compute_measurement_likelihood(pomdp,pos,goal,o.heading)
            else
                # use preference constraints
                for pref=1:(n_prefs-1) # don't include last elt
                    measurement_likelihoods[goal_id,pref] = compute_measurement_likelihood(pomdp,pos,goal,o.heading, prior_belief.neighbor_A, prior_belief.neighbor_b, pref)
                end
            end
        end

        # @show measurement_likelihoods
        # @show prior_belief.belief_intention

        # Update prior - not normalized yet
        new_belief_intentions = measurement_likelihoods .* prior_belief.belief_intention

        # @show new_belief_intentions

    else
        new_belief_intentions = prior_belief.belief_intention
        # If we visited a hypothetical goal and it wasn't one, our belief goes to 0
        if (o.visited_wrong_goal == true) & (pos ∈ pomdp.goal_options)
            println("Visited wrong goal: now updating belief and setting p to 0")
            wrong_goal_id = findfirst(isequal(pos), pomdp.goal_options)
            new_belief_intentions[wrong_goal_id,:] .= 0.0
        end

        # Normalize here!
        new_belief_intentions = new_belief_intentions ./ sum(new_belief_intentions)

        return GridBeliefState(pos, done, prior_belief.neighbor_A,
                               prior_belief.neighbor_b,
                               prior_belief.goal_options,
                               prior_belief.hyperplane_graph,
                               new_belief_intentions,
                               prior_belief.preference_marginals)
    end

    # If we visited a hypothetical goal and it wasn't one, our belief goes to 0
    if (o.visited_wrong_goal == true) & (pos ∈ pomdp.goal_options)
        println("Visited wrong goal: now updating belief and setting p to 0")
        wrong_goal_id = findfirst(isequal(pos), pomdp.goal_options)
        new_belief_intentions[wrong_goal_id,:] .= 0.0
    end

    # Normalize here!
    new_belief_intentions = new_belief_intentions ./ sum(new_belief_intentions)

    # println("After normalization:")
    # @show new_belief_intentions

    return GridBeliefState(pos, done, prior_belief.neighbor_A,
                           prior_belief.neighbor_b,
                           prior_belief.goal_options,
                           prior_belief.hyperplane_graph,
                           new_belief_intentions,
                           prior_belief.preference_marginals)
end

function update_goal_belief(pomdp::MapWorld, b::GridBeliefStateGoal, o::HumanInputObservation)

    # Just copy over
    pos = b.position
    done = b.done

    prior_belief = b

    # now only update belief if you received an observation
    if o.received_observation == true

        n_goals = length(prior_belief.belief_intention)

        measurement_likelihoods = fill(0.0, n_goals)
        #zeros(pomdp.n_possible_goals, length(prior_belief.neighbor_b)+1)

        # Go through all goals and update one by one
        for goal_id=1:n_goals

            goal = pomdp.goal_options[goal_id]
            measurement_likelihoods[goal_id] = compute_measurement_likelihood(pomdp,pos,goal,o.heading)

        end

        # Update prior - not normalized yet
        new_belief_intentions = measurement_likelihoods .* prior_belief.belief_intention



    else
        new_belief_intentions = prior_belief.belief_intention
        # If we visited a hypothetical goal and it wasn't one, our belief goes to 0
        if (o.visited_wrong_goal == true) & (pos ∈ pomdp.goal_options)
            println("Visited wrong goal: now updating belief and setting p to 0")
            wrong_goal_id = findfirst(isequal(pos), pomdp.goal_options)
            new_belief_intentions[wrong_goal_id] = 0.0
        end

        # Normalize here!
        new_belief_intentions = new_belief_intentions ./ sum(new_belief_intentions)

        return GridBeliefStateGoal(pos, done, prior_belief.neighbor_A,
                               prior_belief.neighbor_b,
                               prior_belief.goal_options,
                               prior_belief.hyperplane_graph,
                               new_belief_intentions)
    end

    # If we visited a hypothetical goal and it wasn't one, our belief goes to 0
    if (o.visited_wrong_goal == true) & (pos ∈ pomdp.goal_options)
        println("Visited wrong goal: now updating belief and setting p to 0")
        wrong_goal_id = findfirst(isequal(pos), pomdp.goal_options)
        new_belief_intentions[wrong_goal_id] = 0.0
    end

    # Normalize here!
    new_belief_intentions = new_belief_intentions ./ sum(new_belief_intentions)

    # println("After normalization:")
    # @show new_belief_intentions

    return GridBeliefStateGoal(pos, done, prior_belief.neighbor_A,
                           prior_belief.neighbor_b,
                           prior_belief.goal_options,
                           prior_belief.hyperplane_graph,
                           new_belief_intentions)
end



function update_pose_belief(pomdp::MapWorld, b::GridBeliefState, a::Char)
    println("Entered update pose belief")

    # Just copy over done but update position
    pos = update_position(pomdp, b.position, a)
    done = b.done

    # True if we are staying in the same region
    same = all(<=(0), b.neighbor_A*(-0.1 .+ pos) - b.neighbor_b)

    # @show pos
    # @show same
    # @show b.neighbor_A*pos-b.neighbor_b

    # If region changed, reconstruct belief from marginals
    # New: the "going back to previous set" probability is (1-p(transition))
    if same==0
        n_A, n_b = pos_to_neighbor_matrices(pos, pomdp.hyperplane_graph)

        # Step #1: store all the prior information collected into the belief marginals
        old_vertex_index = pos_to_region_index(b.position, pomdp.hyperplane_graph)
        pref_marginals = b.preference_marginals
        pref_marginals[old_vertex_index] = sum(b.belief_intention, dims=1)[1:end-1]

        # Step #2: update the belief using marginals
        # println("TRANSITION DETECTED! RESHAPING PRIOR BELIEF")
        updated_belief = GridBeliefState(pos, done, n_A, n_b,
                                         b.goal_options, b.hyperplane_graph,
                                         reshape_prior_belief(pomdp,b.position,pos, n_A,n_b, b), pref_marginals)
    else
        # println("NO TRANSITION - NO RESHAPING")
        updated_belief = GridBeliefState(pos, b.done, b.neighbor_A, b.neighbor_b,
                                         b.goal_options, b.hyperplane_graph,
                                         b.belief_intention, b.preference_marginals)
    end

    return updated_belief
end

function update_pose_belief(pomdp::MapWorld, b::GridBeliefStateGoal, a::Char)
    println("Entered update pose belief")

    # Just copy over done but update position
    pos = update_position(pomdp, b.position, a)
    done = b.done

    # True if we are staying in the same region
    same = all(<=(0), b.neighbor_A*(-0.1 .+ pos) - b.neighbor_b)

    # @show pos
    # @show same
    # @show b.neighbor_A*pos-b.neighbor_b

    # If region changed, reconstruct belief from marginals
    # New: the "going back to previous set" probability is (1-p(transition))
    if same==0
        n_A, n_b = pos_to_neighbor_matrices(pos, pomdp.hyperplane_graph)

        # Step #2: update the belief using marginals
        # println("TRANSITION DETECTED! RESHAPING PRIOR BELIEF")
        updated_belief = GridBeliefStateGoal(pos, done, n_A, n_b,
                                         b.goal_options, b.hyperplane_graph,
                                         b.belief_intention)
    else
        # println("NO TRANSITION - NO RESHAPING")
        updated_belief = GridBeliefStateGoal(pos, b.done, b.neighbor_A, b.neighbor_b,
                                         b.goal_options, b.hyperplane_graph,
                                         b.belief_intention)
    end

    return updated_belief
end



# # Ignore - not used in this problem
# POMDPs.actions(pomdp::MapWorld) = ['N', 'S', 'E', 'W', '0', '1', '2', '3']
# POMDPs.discount(pomdp::MapWorld) = pomdp.discount_factor
#
# # For now set isterminal to false and use your own outer condition to play around
# POMDPs.isterminal(pomdp::MapWorld) = false
