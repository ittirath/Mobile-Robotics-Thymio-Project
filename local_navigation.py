import time
from global_variables import *

# --- Logic Functions ---


def _clamp(val):
    return max(min(int(val), 500), -500)

def reset_local_state():
    global current_state, timer_start, previous_wall_side
    current_state = "GLOBAL"
    timer_start = 0
    previous_wall_side = None

def check_obstacle_trigger(prox_horizontal):
    front_all = prox_horizontal[0:5]
    if max(front_all) > THRESH_ENTRY:
        return True
    return False

def local_nav_log(message):
    if DEBUG_PRINT:
        print(f"[LOCAL_NAV] {message}")

def local_nav_update(prox_horizontal):
    """
    Pure logic: takes sensors, returns speed (left, right).
    """
    global current_state, timer_start, previous_wall_side
    
    # sensors
    front_all = prox_horizontal[0:5]
    front_center = prox_horizontal[2]
    left_side = prox_horizontal[0]
    right_side = prox_horizontal[4]
    
    # STATE 1: GLOBAL
    if current_state == "GLOBAL":
        if max(front_all) > THRESH_ENTRY:
            local_nav_log("Obstacle detected. Choosing side.")
            sum_left = prox_horizontal[0] + prox_horizontal[1] 
            sum_right = prox_horizontal[3] + prox_horizontal[4]
            
            if sum_left > (sum_right + 50):
                current_state = "FOLLOW_LEFT"
                previous_wall_side = "LEFT"
            else:
                current_state = "FOLLOW_RIGHT"
                previous_wall_side = "RIGHT"
            return 0, 0
        return BASE_SPEED, BASE_SPEED

    # STATE 2: FOLLOW LEFT
    elif current_state == "FOLLOW_LEFT":
        if DEBUG_PRINT:
            print(f"[LOCAL_NAV] Following left")
        if front_center > THRESH_ENTRY:
            return 100, -100 
        if left_side < THRESH_WALL_LOST:
            local_nav_log("Wall finished. Clearance.")
            current_state = "CLEARANCE"
            # timer_start = time.time()
            timer_start = time.perf_counter()
            return BASE_SPEED, BASE_SPEED
        error = left_side - TARGET_DIST
        correction = int(error * P_GAIN)
        return _clamp(BASE_SPEED + correction), _clamp(BASE_SPEED - correction)

    # STATE 3: FOLLOW RIGHT
    elif current_state == "FOLLOW_RIGHT":
        if DEBUG_PRINT:
            print(f"[LOCAL_NAV] Following right")
        if front_center > THRESH_ENTRY:
            return -100, 100 
        if right_side < THRESH_WALL_LOST:
            local_nav_log("Wall finished. Clearance.")
            current_state = "CLEARANCE"
            # timer_start = time.time()
            timer_start = time.perf_counter()
            return BASE_SPEED, BASE_SPEED
        error = right_side - TARGET_DIST
        correction = int(error * P_GAIN)
        return _clamp(BASE_SPEED - correction), _clamp(BASE_SPEED + correction)

    # STATE 4: CLEARANCE
    elif current_state == "CLEARANCE":
        if DEBUG_PRINT:
            print(f"[LOCAL_NAV] Clearance")
        if max(front_all) > THRESH_ENTRY:
            current_state = f"FOLLOW_{previous_wall_side}"
            return 0, 0
        if (time.perf_counter() - timer_start) >= CLEARANCE_DURATION:
            print("time diff", time.perf_counter() - timer_start)
            local_nav_log("Realigning.")
            current_state = "REALIGN_ANGLE"
            timer_start = time.perf_counter()
            return 0, 0
        return BASE_SPEED, BASE_SPEED

    # STATE 5: REALIGN
    elif current_state == "REALIGN_ANGLE":
        if DEBUG_PRINT:
            print(f"[LOCAL_NAV] Realigning angle")
        if (time.perf_counter() - timer_start) >= TURN_DURATION:
            print("time turn diff", time.perf_counter() - timer_start)
            if max(front_all) > THRESH_ENTRY:
                local_nav_log("Obstacle still present.")
                current_state = "GLOBAL" 
            else:
                local_nav_log("Path clear. Global.")
                current_state = "GLOBAL"
            return 0, 0
        
        if previous_wall_side == "LEFT":
            return -100, 100 
        else:
            return 100, -100 

    return 0, 0