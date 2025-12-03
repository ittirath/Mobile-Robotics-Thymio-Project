from global_variables import *
from dataclasses import dataclass

@dataclass
class LocalNav:
    turn_direction: str          
    state_local_nav: str     

    turn_margin_counter: int
    turn_counter: int
    advance_counter: int

    obstacle_cleared: bool
    realign_done: bool

def check_obstacle_trigger(prox_horizontal):
    front_all = prox_horizontal[0:5]
    if max(front_all) > THRESH_ENTRY:
        return True
    return False

def reset_local_nav(nav: LocalNav):
    nav.turn_direction = ""
    nav.state_local_nav = "TURN"
    nav.turn_margin_counter = 0
    nav.turn_counter = 0
    nav.advance_counter = 0
    nav.obstacle_cleared = False
    nav.realign_done = False

def motor_command(direction: str):
    if direction == "LEFT":
        return BASE_SPEED, -BASE_SPEED
    elif direction == "RIGHT":
        return -BASE_SPEED, BASE_SPEED
    elif direction == "FORWARD":
        return BASE_SPEED, BASE_SPEED
    else:
        return 0, 0

def local_nav_FSM(nav: LocalNav, prox_values):
    """
    Input: nav: LocalNav object, prox_values array
    Output: left_speed, right_speed, nav.obstacle_cleared
    """    
    # sensors
    front_all = prox_values[0:5]
    sum_left = prox_values[0] + prox_values[1] 
    sum_right = prox_values[3] + prox_values[4] 
    #print("prox values = ",prox_values)
    # STATE 1: TURN
    if nav.state_local_nav == "TURN":
        if (max(front_all) > THRESH_ENTRY): # still obstacle in front and haven't started turning
            #print("check direction", sum_left, sum_right)
            if (sum_left > sum_right): # obstacle on left so turn right
                nav.turn_direction = "RIGHT"
                return motor_command("RIGHT")
            else: # obstacle on right so turn left
                nav.turn_direction = "LEFT"
                return motor_command("LEFT")
        else: # path is clear
            nav.advance_counter = 0
            nav.state_local_nav = "ADVANCE"
            return motor_command("STOP")
        
            # UNCOMMENT and comment above IF MARGIN IS NEEDED
            # print("check margin", nav.turn_margin_counter)
            # nav.turn_margin_counter += 1
            # if nav.turn_margin_counter > MARGIN: # margin passed -> CHECK VALUE 
            #     nav.advance_counter = 0
            #     nav.state_local_nav = "ADVANCE"
            #     return motor_command("STOP")
            # elif nav.turn_direction == "RIGHT":
            #     return motor_command("RIGHT")
            # elif nav.turn_direction == "LEFT": 
            #     return motor_command("LEFT")
            # else:
            #     print("ERROR: No turn direction set in TURN state.")
            #     return motor_command("STOP")
            
            
    # STATE 2: ADVANCE
    elif nav.state_local_nav == "ADVANCE":
        nav.advance_counter += 1
        if nav.advance_counter > FIVE_CM: # advanced enough
            nav.state_local_nav = "CHECK_CLEARANCE"
            return motor_command("STOP")
        else: # keep advancing
            return motor_command("FORWARD")
    
    # STATE 3: CHECK CLEARANCE
    elif nav.state_local_nav == "CHECK_CLEARANCE":
        if (max(front_all) > THRESH_ENTRY): # obstacle detected while turning
            #print("TURN COUNTER AT REALIGN ", nav.turn_counter)
            if nav.turn_counter >= NINETY_DEGREES: # obstacle detected after turning at least 90 degrees
                nav.obstacle_cleared = True
                #print("Obstacle cleared.")
            else: 
                nav.obstacle_cleared = False # obstacle detected before turning 90 degrees
            nav.state_local_nav = "REALIGN"
            return motor_command("STOP")

        else: # turn untill obstacle detected
            #print("Turning to check clearance", nav.turn_counter)
            if nav.turn_direction == "LEFT": # turned left so check obstacle on right
                nav.turn_counter += 1
                return motor_command("RIGHT")
            elif nav.turn_direction == "RIGHT": # turned right so check obstacle on left
                nav.turn_counter += 1
                return motor_command("LEFT")
            
    # STATE 4: REALIGN
    elif nav.state_local_nav == "REALIGN":
        # nav.turn_counter -= 1
        # if nav.turn_counter-NINETY_DEGREES <= 0: # realigned with obstacle at 90 degrees w.r.t heading direction
        if (max(front_all) < THRESH_ENTRY): # path is clear
            if nav.obstacle_cleared:
                nav.realign_done = True
                return motor_command("STOP")
            else:
                nav.state_local_nav = "ADVANCE"
                nav.advance_counter = 0
                return motor_command("STOP")
            
        else: # keep realigning
            if nav.turn_direction == "LEFT": # turned left so realign left
                return motor_command("LEFT")
            elif nav.turn_direction == "RIGHT": # turned right so realign right
                return motor_command("RIGHT")
            
    print ("ERROR: Local Navigation FSM reached undefined state.")
    return motor_command("STOP")

