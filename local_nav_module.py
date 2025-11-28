import time

# Constatns
THRESH_ENTRY = 2000      # Threshold to enter local nav
THRESH_WALL_LOST = 400   # Threshold for "losing" the wall
TARGET_DIST = 2000       # Distance to the wall of the obstacle
BASE_SPEED = 100
P_GAIN = 0.1

# Times
CLEARANCE_DURATION = 1.5 # Time to turn the corner (step forward)
TURN_DURATION = 1.5      # Time to turn

# Debug flag to enable debug print
DEBUG_PRINT = True

# State variables at module level (!= main_loop)
current_state = "GLOBAL"
timer_start = 0
previous_wall_side = None 

# Helper functions
def _clamp(val):
    # motor speed capped at 500
    return max(min(int(val), 500), -500)

def reset_local_state():
    # resets local nav state, called when entering local nav from main
    # to have always the same starting state
    global current_state, timer_start, previous_wall_side
    current_state = "GLOBAL"
    timer_start = 0
    previous_wall_side = None

def check_obstacle_trigger(prox_horizontal):
    # checks if an obstacle is close enough to trigger local nav
    # true iff max_prx > THRESH_ENTRY
    front_all = prox_horizontal[0:5]
    
    if max(front_all) > THRESH_ENTRY:
        return True
    return False

def local_nav_log(message):
    if DEBUG_PRINT:
        print(f"[LOCAL_NAV] {message}")

# MAIN LOGIC
def local_nav_update(prox_horizontal):
    #returns left_speed, right_speed
    # global to modify them
    global current_state, timer_start, previous_wall_side
    
    # sensors
    front_all = prox_horizontal[0:5]
    front_center = prox_horizontal[2]
    left_side = prox_horizontal[0]
    right_side = prox_horizontal[4]
    
    # STATE 1: in global we move forward until we see the obstacle
    if current_state == "GLOBAL":
        # If frontal obstacle
        if max(front_all) > THRESH_ENTRY:
            local_nav_log("Obstacle detected. Choosing the best side to continue.")
            
            # Determine side (sum of left vs right sensors)
            sum_left = prox_horizontal[0] + prox_horizontal[1] 
            sum_right = prox_horizontal[3] + prox_horizontal[4]
            
            if sum_left > (sum_right + 50):
                # Wall Left -> Follow Left
                current_state = "FOLLOW_LEFT"
                previous_wall_side = "LEFT"
            else:
                # Wall on the Right -> Follow Right
                current_state = "FOLLOW_RIGHT"
                previous_wall_side = "RIGHT"
            return 0, 0
        
        # No obstacle
        return BASE_SPEED, BASE_SPEED

    # STATE 2 follow left wall
    elif current_state == "FOLLOW_LEFT":
        # if we hit the front (inside corner), turn right
        if front_center > THRESH_ENTRY:
            return 100, -100 

        # exit if the wall is finished
        if left_side < THRESH_WALL_LOST:
            local_nav_log("Wall finished. Going away.")
            current_state = "CLEARANCE"
            timer_start = time.time()
            return BASE_SPEED, BASE_SPEED

        # mantaining distance from wall
        error = left_side - TARGET_DIST
        correction = int(error * P_GAIN)
        return _clamp(BASE_SPEED + correction), _clamp(BASE_SPEED - correction)

    # STATE 3: follow rigth wall
    elif current_state == "FOLLOW_RIGHT":
        if front_center > THRESH_ENTRY:
            return -100, 100 

        # exit
        if right_side < THRESH_WALL_LOST:
            local_nav_log("Wall finished. Going away.")
            current_state = "CLEARANCE"
            timer_start = time.time()
            return BASE_SPEED, BASE_SPEED

        # mantaining distance from wall
        error = right_side - TARGET_DIST
        correction = int(error * P_GAIN)
        return _clamp(BASE_SPEED - correction), _clamp(BASE_SPEED + correction)

    # STATE 4: we go straight to be sure to surpass obstacle)
    elif current_state == "CLEARANCE":
        # If we hit something while moving forward, we go back to follow
        if max(front_all) > THRESH_ENTRY:
            current_state = f"FOLLOW_{previous_wall_side}"
            return 0, 0

        # If timer finished, we try to realign ourselves
        if (time.time() - timer_start) >= CLEARANCE_DURATION:
            local_nav_log("Realigning angle.")
            current_state = "REALIGN_ANGLE"
            timer_start = time.time()
            return 0, 0
        
        return BASE_SPEED, BASE_SPEED

    # STATE 5
    elif current_state == "REALIGN_ANGLE":
        # Timer, finish turning?
        if (time.time() - timer_start) >= TURN_DURATION:
            
            # now we are realigned, is there any obstacle in front of us?
            if max(front_all) > THRESH_ENTRY:
                local_nav_log("Obstacle still present.")
                current_state = "GLOBAL" 
            else:
                local_nav_log("No obstacle detected. Returning to Global Navigation!")
                current_state = "GLOBAL"
            
            return 0, 0

        # Rotation
        if previous_wall_side == "LEFT":
            return -100, 100 # Rotate to the left
        else:
            return 100, -100 # Rotate to the right

    return 0, 0