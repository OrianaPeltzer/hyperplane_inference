"""Sample random belief state"""
function Base.rand(rng::AbstractRNG, bs::GridBeliefState)
    # We need to _sample_ state instances from a belief state

    # Just copy over position (observable)
    pos = bs.position
    done = bs.done

    # get marginals
    belief_goals = sum(bs.belief_intention, dims=2)
    # Sample a new goal from probabilities
    goal_index = findfirst(cumsum(belief_goals) .>= rand(rng))

    # Sample a new preference from the goal and position
    pref = sample(1:length(bs.belief_intention[goal_index,:]), Weights(bs.belief_intention[goal_index,:]))

    return GridState(pos, done, goal_index, bs.neighbor_A, bs.neighbor_b, pref)
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

    # Just copy over done but update position
    pos = b.position
    done = b.done

    prior_belief = b

    # now only update belief if you received an observation
    if o.received_observation == true

        # If we made a transition, skip the likelihood update (except for wrong
        # goal part) and return current beliefs.
        human_intended_position = update_position(pomdp,pos,o.heading)
        same = all(<=(0), b.neighbor_A*human_intended_position-b.neighbor_b)
        if same==0
            # println("DETECTED TRANSITION!")
            new_belief_intentions = b.belief_intention
            # If we visited a hypothetical goal and it wasn't one, our belief goes to 0
            if (o.visited_wrong_goal == true) & (pos ∈ pomdp.goal_options)
                # println("Visited wrong goal: now updating belief and setting p to 0")
                wrong_goal_id = findfirst(isequal(pos), pomdp.goal_options)
                new_belief_intentions[wrong_goal_id,:] .= 0.0
            end
            # println("EXITED IF")

            # Normalize here!
            new_belief_intentions = new_belief_intentions ./ sum(new_belief_intentions)

            # println("After normalization:")
            # @show new_belief_intentions

            return GridBeliefState(pos, done, prior_belief.neighbor_A, prior_belief.neighbor_b, new_belief_intentions)
        end


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
    same = all(<=(0), b.neighbor_A*pos-b.neighbor_b)

    # @show pos
    # @show same
    # @show b.neighbor_A*pos-b.neighbor_b

    # If region changed, reconstruct belief from marginals
    if same==0
        n_A, n_b = pos_to_neighbor_matrices(pos)
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
