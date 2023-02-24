"""Returns an occupancy grid of the environment"""
function occupancy_grid()
    # Generate a true 10x10 false/true map and ensure start position is empty (false)
    grid_side = 10
    OBSTACLE_MAP = fill(false, grid_side, grid_side)
    println("OBSTACLE MAP: ", OBSTACLE_MAP)
    return OBSTACLE_MAP
end
