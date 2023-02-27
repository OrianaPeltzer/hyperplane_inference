
function compute_probability_correct_goal(p::MapWorld, b::GridBeliefState)

    marginals_over_goals = sum(b.belief_intention, dims=2)

    return marginals_over_goals[p.true_goal_index]
end

function compute_probability_correct_goal(p::MapWorld, b::GridBeliefStateGoal)
    return b.belief_intention[p.true_goal_index]
end

function compute_entropy_goal_distribution(p::MapWorld, b::GridBeliefState)

    marginals_over_goals = sum(b.belief_intention, dims=2)

    p_logp = [x*log(x+0.0000001) for x in marginals_over_goals]

    return -sum(p_logp)
end

function compute_entropy_goal_distribution(p::MapWorld, b::GridBeliefStateGoal)
    p_logp = [x*log(x+0.0000001) for x in b.belief_intention]
    return -sum(p_logp)
end

function violated_preferences(p::MapWorld, pos::GridPosition, a::Char)

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

    # If we hit an obstacle, we don't actually get to move: no violation
    if p.obstacle_map[new_pos...] == true
        return false
    end

    # Cutting through polytope corners or to non-existing sets also results in no mvt
    G = p.hyperplane_graph
    curr_v_idx = pos_to_region_index(curr_pos, G)
    A = get_prop(G,curr_v_idx,:A)
    b = get_prop(G,curr_v_idx,:b)
    same_region = is_in_region(A, b, new_pos)
    if ~same_region
        sequence = A*(-0.1 .+ new_pos)-b
        num_jumps = count(>(0),sequence)
        transition_A_index = findfirst(.>(0),sequence)[1]

        pref_neighbors = get_prop(G, curr_v_idx, :pref_neighbors)
        admissible_A_transition_indices = [pn.idx_in_A for pn in pref_neighbors]

        actual_preference_graph = p.true_preference[curr_v_idx]


        # If we're in the goal region, there's no preference and any transition
        # is a violation
        # if actual_preference_edgenumber==-1
        #     return true
        # end



        # if isnothing(actual_preference)
        #     return true
        # end

        if (num_jumps != 1) || ~(transition_A_index in admissible_A_transition_indices)
            # new_pos = curr_pos
            # same_region = true
            return false
        elseif actual_preference_graph==-1 # We did a valid thing but we weren't supposed to transition
            return true
        end
        actual_preference_edgenumber = get_edge_number_from_graph_pref_index(G, curr_v_idx, actual_preference_graph)
        actual_preference = get_pref_from_edge_number(G, curr_v_idx, actual_preference_edgenumber)
        # We transitioned to an admissible region that was not the correct one
        if transition_A_index != actual_preference.idx_in_A
            # Violation
            return true
        end
    end



    return false
end
