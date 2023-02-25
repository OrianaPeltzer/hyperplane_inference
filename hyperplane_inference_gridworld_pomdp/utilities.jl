"""(Utilities) Returns the angle difference b-a"""
function angdiff(a, b)
    r = abs(b - a)
    if r ≥ pi
        r -= 2*pi
    end
    return r
end

"""(Utilities) computes the angle traced by the line to the target"""
function compute_direction(pos::GridPosition, target::GridPosition)
    return atan(target[2]-pos[2], target[1]-pos[1])
end

function is_in_region(A::Matrix, b::Vector, pos::GridPosition)
    """Returns true if and only if the specified position is inside the region
    denoted by A and b."""
    adjustment = 0.1
    sequence = A*[pos[1]-adjustment, pos[2]-adjustment] - b
    return all(<=(0), sequence)
end

function pos_to_neighbor_matrices(pos::GridPosition, G::MetaGraph;
                             adjustment::Float64=0.1)
    """Returns matrices A and b corresponding to A*pos - b <= 0 on the graph."""

    # @infiltrate

    for i in vertices(G)
        # Matrices that include the contour
        Ac = get_prop(G, i, :A)
        bc = get_prop(G, i, :b)
        contour_indices = get_prop(G, i, :contour_indices)

        # println(Ac)
        # println(bc)

        sequence = Ac*[pos[1]-adjustment, pos[2]-adjustment] - bc
        # If all elements of A*pos-b are negative, we found the set
        if all(<=(0), sequence)
            # # Remove the contour from Ac and bc (no inference over boundaries)
            # A = [nothing nothing]
            # b = nothing
            # for j in 1:length(bc)
            #     if j ∉ contour_indices
            #         A = vcat(A, transpose(Ac[j,:]))
            #         b = vcat(b, bc[j])
            #     end
            # end
            # # println("Done with for loop")
            # # @show A
            # # @show b
            # # (debug) remove nothing from beginning of lists?!
            # A = A[2:end,:]
            # b = b[2:end]
            #
            # # @show A
            # # @show b

            return Ac, bc
        end
    end

    return [], []
end

function pos_to_region_index(pos::GridPosition, G::MetaGraph;
                             adjustment::Float64=0.1)
    """Returns index of vertex corresponding to matrices A and b for which
    A*pos - b <= 0 on the graph.
    Adjustment is only used for rendering the obstacles in visualization script"""

    for i in vertices(G)
        # Matrices that include the contour
        Ac = get_prop(G, i, :A)
        bc = get_prop(G, i, :b)
        # contour_indices = get_prop(G, i, :contour_indices)

        sequence = Ac*[pos[1]-adjustment, pos[2]-adjustment] - bc
        # If all elements of A*pos-b are negative, we found the set
        if all(<=(0), sequence)
            return i
        end
    end
    println("Did not find region of the graph corresponding to color!")
    return -1
end

function pos_to_neighbor_matrices_no_obs(pos::GridPosition)
    """Hand-made for the moment. Joe's method should go in here.
    Returns A and b corresponding to the robot's current location and the
    neighboring regions. Ax-b <= 0 indicates that the robot is inside.

    Current: obstacle at (5,5) to (6,6), size 2x2
    """
    # Rectangular obstacle limits
    Lx = 5
    Ly = 5
    Ux = 6
    Uy = 6

    lowerleftA = [1 0
                  0 1]
    lowerleftb = [Lx-1,Ly-1]

    leftA = [1 0
             0 -1]
    leftb = [Lx-1,-Ly]

    upperleftA = [1 0
                  0 -1]
    upperleftb = [Lx-1, -Uy-1]

    upA = [1 0
           -1 0]
    upb = [Ux, -Lx]

    upperrightA = [-1 0
                   0 -1]
    upperrightb = [-Ux-1, -Uy-1]

    rightA = [0 1
              0 -1]
    rightb = [Uy, -Ly]

    lowerrightA = [-1 0
                   0 1]
    lowerrightb = [-Ux-1, Ly-1]

    lowA = [1 0
            -1 0]
    lowb = [Ux, -Lx]

    As = [lowerleftA, leftA, upperleftA, upA, upperrightA, rightA, lowerrightA, lowA]
    bs = [lowerleftb, leftb, upperleftb, upb, upperrightb, rightb, lowerrightb, lowb]

    for i=1:8
        sequence = As[i]*[pos[1], pos[2]] - bs[i]
        if all(<=(0), sequence)
            return As[i], bs[i]
        end
    end

    return [], []
end

function pos_to_neighbor_matrices(pos::GridPosition)
    """Hand-made for the moment. Joe's method should go in here.
    Returns A and b corresponding to the robot's current location and the
    neighboring regions. Ax-b <= 0 indicates that the robot is inside.

    Current: obstacle at (5,5) to (6,6), size 2x2
    """
    # Rectangular obstacle limits
    Lx = 5
    Ly = 5
    Ux = 6
    Uy = 6

    lowerleftA = [1 0
                  0 1]
    lowerleftb = [Lx-1,Ly-1]

    leftA = [1 0
             0 1
             0 -1]
    leftb = [Lx-1,Uy,-Ly]

    upperleftA = [1 0
                  0 -1]
    upperleftb = [Lx-1, -Uy-1]

    upA = [1 0
           -1 0
           0 -1]
    upb = [Ux, -Lx, -Uy-1]

    upperrightA = [-1 0
                   0 -1]
    upperrightb = [-Ux-1, -Uy-1]

    rightA = [-1 0
              0 1
              0 -1]
    rightb = [-Ux-1, Uy, -Ly]

    lowerrightA = [-1 0
                   0 1]
    lowerrightb = [-Ux-1, Ly-1]

    lowA = [1 0
            -1 0
            0 1]
    lowb = [Ux, -Lx, Ly-1]

    As = [lowerleftA, leftA, upperleftA, upA, upperrightA, rightA, lowerrightA, lowA]
    bs = [lowerleftb, leftb, upperleftb, upb, upperrightb, rightb, lowerrightb, lowb]

    for i=1:8
        sequence = As[i]*[pos[1], pos[2]] - bs[i]
        if all(<=(0), sequence)
            return As[i], bs[i]
        end
    end

    return [], []
end

function pos_to_lines(pos::GridPosition)

    Lx = 5
    Ly = 5
    Ux = 6
    Uy = 6

    # left of vertical, below hz
    lowerleftA = [1 0
                  0 1]
    lowerleftb = [Lx-1,Ly-1]
    lowerleftlines = [[(4,0),(4,4)], [(0,4),(4,4)]]

    # left of v, below upper, above lower
    leftA = [1 0
             0 1
             0 -1]
    leftb = [Lx-1,Uy,-Ly]
    leftlines = [[(4,4),(4,6)], [(0,6),(4,6)], [(0,4),(4,4)]]

    # left of v, above lower
    upperleftA = [1 0
                  0 -1]
    upperleftb = [Lx-1, -Uy-1]
    upperleftlines = [[(4,6),(4,10)], [(0,6),(4,6)]]

    # left of right, right of left, above h
    upA = [1 0
           -1 0
           0 -1]
    upb = [Ux, -Lx, -Uy-1]
    uplines = [[(6,6),(6,10)], [(4,6),(4,10)], [(4,6),(6,6)]]

    # right of v, above h
    upperrightA = [-1 0
                   0 -1]
    upperrightb = [-Ux-1, -Uy-1]
    upperrightlines = [[(6,6),(6,10)], [(6,6),(10,6)]]

    # right of v, below h, above h
    rightA = [-1 0
              0 1
              0 -1]
    rightb = [-Ux-1, Uy, -Ly]
    rightlines = [[(6,4),(6,6)], [(6,6),(10,6)], [(6,4),(10,4)]]

    # right of v, below h
    lowerrightA = [-1 0
                   0 1]
    lowerrightb = [-Ux-1, Ly-1]
    lowerrightlines = [[(6,0),(6,4)], [(6,4),(10,4)]]

    # left of right, right of left, below h
    lowA = [1 0
            -1 0
            0 1]
    lowb = [Ux, -Lx, Ly-1]
    lowlines = [[(6,0),(6,4)], [(4,0),(4,4)], [(4,4),(6,4)]]

    As = [lowerleftA, leftA, upperleftA, upA, upperrightA, rightA, lowerrightA, lowA]
    bs = [lowerleftb, leftb, upperleftb, upb, upperrightb, rightb, lowerrightb, lowb]
    lines = [lowerleftlines, leftlines, upperleftlines, uplines, upperrightlines, rightlines, lowerrightlines, lowlines]

    for i=1:8
        sequence = As[i]*[pos[1], pos[2]] - bs[i]
        if all(<=(0), sequence)
            return lines[i]
        end
    end

    return []
end


function is_obstacle_index(pos::GridPosition, pref::Int)
    """Hand-made for the moment. Joe's method should go in here.
    Returns A and b corresponding to the robot's current location and the
    neighboring regions. Ax-b <= 0 indicates that the robot is inside.

    Current: obstacle at (5,5) to (6,6), size 2x2
    """
    # Rectangular obstacle limits
    Lx = 5
    Ly = 5
    Ux = 6
    Uy = 6

    lowerleftA = [1 0
                  0 1]
    lowerleftb = [Lx-1,Ly-1]

    leftA = [1 0
             0 1
             0 -1]
    leftb = [Lx-1,Uy,-Ly]

    upperleftA = [1 0
                  0 -1]
    upperleftb = [Lx-1, -Uy-1]

    upA = [1 0
           -1 0
           0 -1]
    upb = [Ux, -Lx, -Uy-1]

    upperrightA = [-1 0
                   0 -1]
    upperrightb = [-Ux-1, -Uy-1]

    rightA = [-1 0
              0 1
              0 -1]
    rightb = [-Ux-1, Uy, -Ly]

    lowerrightA = [-1 0
                   0 1]
    lowerrightb = [-Ux-1, Ly-1]

    lowA = [1 0
            -1 0
            0 1]
    lowb = [Ux, -Lx, Ly-1]

    As = [lowerleftA, leftA, upperleftA, upA, upperrightA, rightA, lowerrightA, lowA]
    bs = [lowerleftb, leftb, upperleftb, upb, upperrightb, rightb, lowerrightb, lowb]
    obstacle_indices = [-1, 1, -1, 3, -1, 1, -1, 3]

    for i=1:8
        sequence = As[i]*[pos[1], pos[2]] - bs[i]
        if all(<=(0), sequence)
            if obstacle_indices[i]==pref
                return true
            else
                return false
            end
        end
    end

    return false
end
