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
    sequence = A*[pos[1], pos[2]] - b
    return all(<=(0), sequence)
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
