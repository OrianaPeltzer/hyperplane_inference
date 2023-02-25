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

    # Sample a new preference from the goal and position
    w = bs.belief_intention[goal_index,:] ./ sum(bs.belief_intention[goal_index,:])
    pref = sample(1:length(bs.belief_intention[goal_index,:]), Weights(w))

    return GridState(pos, done, goal_index, bs.neighbor_A, bs.neighbor_b, pref)
end

# struct GridState
#     position::GridPosition
#     done::Bool # are we in a terminal state?
#     goal_index::Int64 # This is the goal
#     neighbor_A::Matrix # A*current_state - b <= 0
#     neighbor_b::Array
#     intention_index::Int64 # This is the index of the preferred neighbor
# end

# @with_kw struct GridBeliefState
#     position::GridPosition
#     done::Bool # are we in a terminal state?
#     neighbor_A::Matrix # A*current_state - b <= 0
#     neighbor_b::Array
#     belief_intention::Matrix{Float64} # size #goals*(1+#neighbors)
# end


# G_h = loadgraph("maps/"*map_name*".mg", MGFormat())
#
# function Base.rand(rng::AbstractRNG, bs::GridBeliefState)
#     # We need to _sample_ state instances from a belief state
#
#     # Just copy over position (observable)
#     pos = bs.position
#     done = bs.done
#
#     A, b = pos_to_neighbor_matrices(pos, G_h)
#
#     sum=0.0
#     limit = rand(rng)
#
#     for goal in 1:length(bs.belief_intention[:,1])
#         for pref in 1:length(bs.belief_intention[1,:])
#             sum += bs.belief_intention[goal, pref]
#             if sum >= limit
#                 return GridState(pos, done, goal, A, b, pref)
#             end
#         end
#     end
#
#     println("Wasn't able to sample a state!!!")
#     return nothing
#
#     # # Sample a new goal from probabilities
#     # goal_index = findfirst(cumsum(bs.belief_goals) .>= rand(rng))
#     #
#     # return GridState(pos, done, goal_index)
# end










POMDPs.actions(pomdp::MapWorld) = ['N', 'S', 'E', 'W', '0', '1', '2', '3']
POMDPs.discount(pomdp::MapWorld) = pomdp.discount_factor

# For now set isterminal to false and use your own outer condition to play around
POMDPs.isterminal(pomdp::MapWorld) = false

# NOTE: Simple boilerplate stuff for rollouts (ignore now)
struct MapWorldBelUpdater <: Updater
    pomdp::MapWorld
end

function POMDPs.update(updater::MapWorldBelUpdater, b::GridBeliefState, a::Char, o::HumanInputObservation)
    return update_pose_belief(updater.pomdp, b, a) # no observation in this model!
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

    goal = p.goal_options[s.goal_index]

    # We suppose the human won't give any more observations
    o = HumanInputObservation(0.0,false,false)

    pref = s.intention_index

    # Get reward if we reached the goal, otherwise get penalty
    if curr_pos == goal
        r = p.reward
        new_state = GridState(curr_pos, true, s.goal_index, s.neighbor_A,
                              s.neighbor_b, s.intention_index)
        return (sp=new_state,r=r,o=o)
    end


    # We did not reach the goal, so we get a penalty
    r = p.penalty

    # Set robot position, and get reward or penalty based on true state of next cell
    # I've implicitly bounded in grid here
    # Assuming FIRST coordinate is y, i.e. rows number
    if a == 'S'
        new_pos = GridPosition(max(1, curr_pos[1]-1), curr_pos[2])
    elseif a == 'N'
        new_pos = GridPosition(min(p.grid_side, curr_pos[1]+1), curr_pos[2])
    elseif a == 'W'
        new_pos = GridPosition(curr_pos[1], max(1, curr_pos[2]-1))
    elseif a == 'E'
        new_pos = GridPosition(curr_pos[1], min(p.grid_side, curr_pos[2]+1))
    elseif a == '0'
        new_pos = GridPosition(max(1, curr_pos[1]-1), max(1, curr_pos[2]-1))
        r = p.diag_penalty
    elseif a == '1'
        new_pos = GridPosition(max(1, curr_pos[1]-1), min(p.grid_side, curr_pos[2]+1))
        r = p.diag_penalty
    elseif a == '2'
        new_pos = GridPosition(min(p.grid_side, curr_pos[1]+1), max(1, curr_pos[2]-1))
        r = p.diag_penalty
    elseif a == '3'
        new_pos = GridPosition(min(p.grid_side, curr_pos[1]+1), min(p.grid_side, curr_pos[2]+1))
        r = p.diag_penalty
    else
        new_pos = curr_pos
    end

    # If we hit an obstacle, we don't actually get to move
    if p.obstacle_map[new_pos...] == true
        new_pos = curr_pos
        r = p.penalty # If we failed to do a diagonal mvt, set the penalty back
    end


    # Case where the new position is in a new set: transition penalty
    same_region = is_in_region(s.neighbor_A, s.neighbor_b, new_pos)
    if ~same_region
        sequence = s.neighbor_A*(-0.1 .+ new_pos)-s.neighbor_b
        transition_index = findfirst(.>(0),sequence)[1]

        if transition_index != s.intention_index
            r += p.incorrect_transition_penalty
        end
    end

    # If we changed region, A, b and the true preference must change
    if ~same_region
        @show new_pos
        # find new A, b
        new_A, new_b = pos_to_neighbor_matrices(new_pos, p.hyperplane_graph)

        @show new_A
        @show new_b

        # find new preference
        new_node_index = pos_to_region_index(new_pos, p.hyperplane_graph)
        @show new_node_index

        new_pref_index_global = p.true_preference[new_node_index]

        @show new_pref_index_global

        new_pref_index = get_local_pref_from_graph_pref_index(p.hyperplane_graph,
                                          new_node_index, new_pref_index_global)

        new_state = GridState(new_pos, false, s.goal_index, new_A,
                              new_b, new_pref_index)
    else
        new_state = GridState(new_pos, false, s.goal_index, s.neighbor_A,
                              s.neighbor_b, s.intention_index)
    end


        # struct GridState
        #     position::GridPosition
        #     done::Bool # are we in a terminal state?
        #     goal_index::Int64 # This is the goal
        #     neighbor_A::Matrix # A*current_state - b <= 0
        #     neighbor_b::Array
        #     intention_index::Int64 # This is the index of the preferred neighbor
        # end

    # # over-write new state
    # new_state = GridState(new_pos, false, s.goal_index)

    # # If we are at a state that is not the goal, mention it in observation
    # if (new_pos in p.goal_options) && (new_pos != goal)
    #     new_obs = HumanInputObservation(sample_direction(new_pos,goal,p.human_input_std), true, true)
    # else
    #     new_obs = HumanInputObservation(sample_direction(new_pos,goal,p.human_input_std), true, false)
    # end

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

        # If we made a transition, skip the likelihood update (except for wrong
        # goal part) and return current beliefs.
        # human_intended_position = update_position(pomdp,pos,o.heading)
        # same = all(<=(0), b.neighbor_A*human_intended_position-b.neighbor_b)
        # if same==0
        #     # println("Observation is mapped to transition in another set!")
        #     new_belief_intentions = b.belief_intention
        #     # If we visited a hypothetical goal and it wasn't one, our belief goes to 0
        #     if (o.visited_wrong_goal == true) & (pos ∈ pomdp.goal_options)
        #         # println("Visited wrong goal: now updating belief and setting p to 0")
        #         wrong_goal_id = findfirst(isequal(pos), pomdp.goal_options)
        #         new_belief_intentions[wrong_goal_id,:] .= 0.0
        #     end
        #     # println("EXITED IF")
        #
        #     # Normalize here!
        #     new_belief_intentions = new_belief_intentions ./ sum(new_belief_intentions)
        #
        #     # println("After normalization:")
        #     # @show new_belief_intentions
        #
        #     return GridBeliefState(pos, done, prior_belief.neighbor_A, prior_belief.neighbor_b, new_belief_intentions)
        # end


        measurement_likelihoods = zeros(pomdp.n_possible_goals, length(prior_belief.neighbor_b)+1)

        # Go through all goals and update one by one
        for goal_id=1:pomdp.n_possible_goals

            goal = pomdp.goal_options[goal_id]

            if is_in_region(prior_belief.neighbor_A, prior_belief.neighbor_b, goal)

                # update measurement likelihood with no constraints (previous method)
                measurement_likelihoods[goal_id,length(prior_belief.neighbor_b)+1] = compute_measurement_likelihood(pomdp,pos,goal,o.heading)
            else
                # use preference constraints
                for pref=1:length(prior_belief.neighbor_b)
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
        return GridBeliefState(pos, done, prior_belief.neighbor_A, prior_belief.neighbor_b, prior_belief.belief_intention)
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

    return GridBeliefState(pos, done, prior_belief.neighbor_A, prior_belief.neighbor_b, new_belief_intentions)
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
        # println("TRANSITION DETECTED! RESHAPING PRIOR BELIEF")
        updated_belief = GridBeliefState(pos, done, n_A, n_b, reshape_prior_belief(pomdp,b.position,pos, n_A,n_b, b))
    else
        # println("NO TRANSITION - NO RESHAPING")
        updated_belief = GridBeliefState(pos, b.done, b.neighbor_A, b.neighbor_b, b.belief_intention)
    end

    return updated_belief
end



# # Ignore - not used in this problem
# POMDPs.actions(pomdp::MapWorld) = ['N', 'S', 'E', 'W', '0', '1', '2', '3']
# POMDPs.discount(pomdp::MapWorld) = pomdp.discount_factor
#
# # For now set isterminal to false and use your own outer condition to play around
# POMDPs.isterminal(pomdp::MapWorld) = false
