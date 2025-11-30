import numpy as np
import cv2
import math

from global_variables import *

# COMPUTER VISION HELPER FUNCTIONS ----------------------------------------

def path_world_to_camera(frame_size, path_world):
    H = frame_size[1]
    path_cam = path_world.copy()
    path_cam[1, :] = H - path_cam[1, :]
    return path_cam

# Compute center of the ArUco marker
def compute_center(top_left,bottom_right):
    center_x = int((top_left[0] + bottom_right[0]) / 2.0)
    center_y = int((top_left[1] + bottom_right[1]) / 2.0)
    return center_x,center_y

# Compute angle of vector = (bottom_left -> top_left) w.r.t horizontal axis
def compute_angle(top_left,bottom_left):
    x_diff = int(top_left[0] - bottom_left[0])
    y_diff = int(top_left[1] - bottom_left[1])
    return math.atan2(y_diff,x_diff)

def detect_red_polygons(frame,
                        min_area=300,       # filter tiny blobs
                        eps_factor=0.02):   # approximation factor for vertices
    """
    Returns:
      polygons: list of (N,2) int arrays with polygon vertices in image coords
      mask: binary mask used to detect red regions
    """
    # BGR -> HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Red wraps around 0°, so use two ranges
    lower_red1 = np.array([0,   80,  80])
    upper_red1 = np.array([10,  255, 255])
    lower_red2 = np.array([170, 80,  80])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask  = cv2.bitwise_or(mask1, mask2)

    # Cleanup (optional but usually helpful)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=2)

    # Contours --> polygons
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    polygons = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, eps_factor * peri, True)  # (N,1,2)
        verts = approx.reshape(-1, 2)                            # (N,2)
        polygons.append(verts.astype(int))

    return polygons, mask

def detect_red_polygons_in_map(frame, p_tl, p_br,min_area=300, eps_factor=0.02):
    x0, y0 = p_tl
    x1, y1 = p_br

    # Ensure proper ordering
    x0, x1 = sorted((x0, x1))
    y0, y1 = sorted((y0, y1))

    roi = frame[y0:y1, x0:x1]

    polygons_roi, mask_roi = detect_red_polygons(roi,min_area=min_area,eps_factor=eps_factor)
    # Shift vertices back to full-frame coordinates
    polygons_full = []
    offset = np.array([[x0, y0]])
    for verts in polygons_roi:
        polygons_full.append(verts + offset)

    #return polygons_full, mask_roi
    return polygons_full

# map a single image point (u,v) to world (X,Y)
def img_to_world(u,v,H):
    uv1 = np.array([u,v, 1.0], dtype=np.float32) # z = 1
    XY1 = H @ uv1
    XY1 /= XY1[2]  # scale back to world coordinates where z = 1
    return float(XY1[0]), float(XY1[1])

# helper function to display text
def putText(frame, text, x, y):
    cv2.putText(frame,str(text),(int(x), int(y)),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0, 100, 150),2)

def draw_global_path(frame, H, global_path):
    if global_path is None or global_path.shape[1] < 2:
        return

    # If H maps image -> world, then inv(H) maps world -> image
    try:
        Hinv = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return  # H not invertible

    # global_path is 2xN (x_world, y_world). Convert to Nx1x2 for OpenCV.
    pts_world = global_path[:2, :].T.astype(np.float32).reshape(-1, 1, 2)

    # Map world -> image using Hinv
    pts_img = cv2.perspectiveTransform(pts_world, Hinv).reshape(-1, 2)

    # Draw polyline
    for p1, p2 in zip(pts_img[:-1], pts_img[1:]):
        if not (np.isfinite(p1).all() and np.isfinite(p2).all()):
            continue
        x1, y1 = map(int, np.round(p1))
        x2, y2 = map(int, np.round(p2))
        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)


# COMPUTER VISION GLOBAL FUNCTIONS ----------------------------------------

def get_camera_measurement(frame, only_thymio=False, plot_view=False):
    """
    Returns:
        thymio_x, thymio_y, thymio_theta: position (x,y) in m and orientation theta in radians
    If only_thymio is False then in addition to above returns:
        goal_x, goal_y: position (x,y) in m
        list of (N,2) int arrays with polygon vertices in world coords
    """
    # Convert to grayscale (Aruco works on grayscale)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # gray is an array of shape (height, width, 1) type uint8

    # Detect markers
    #corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)
    detector = cv2.aruco.ArucoDetector(dictionary, detector_params)
    corners, ids, rejected = detector.detectMarkers(gray)
    
    # Initialize outputs
    thymio_start = goal = np.array([1e8,1e8])
    thymio_theta = 0
    polygons_real_world = H = None

    # ---------------- Marker detection and map definition ----------------
    if ids is not None and len(ids) >= 6: # all markers must be detected

        # 0: TL, 1: TR, 2: BR, 3: BL, 4: thymio, 5: goal
        marker0_TL = marker1_TR = marker2_BR = marker3_BL = marker4_thymio = marker5_goal = 0

        ids_flat = ids.flatten()
        for i, marker_id in enumerate(ids_flat):
            if marker_id == 0:
                marker0_TL = i
            elif marker_id == 1:
                marker1_TR = i
            elif marker_id == 2:
                marker2_BR = i
            elif marker_id == 3:
                marker3_BL = i
            elif marker_id == 4:
                marker4_thymio = i
            elif marker_id == 5:
                marker5_goal = i

        # Interior corner of each map marker (assuming OpenCV order TL=0, TR=1, BR=2, BL=3)
        tl_pt = corners[marker0_TL][0][2]  # BR of TL marker (closest to map interior)
        tr_pt = corners[marker1_TR][0][3]  # BL of TR marker
        br_pt = corners[marker2_BR][0][0]  # TL of BR marker
        bl_pt = corners[marker3_BL][0][1]  # TR of BL marker

        # Four corners of the virtual map (axis aligned)
        p_tl = (int(tl_pt[0]), int(tl_pt[1]))
        p_br = (int(br_pt[0]), int(br_pt[1]))
        p_tr = (int(tr_pt[0]), int(tr_pt[1]))
        p_bl = (int(bl_pt[0]), int(bl_pt[1]))

        # ---------------- Position/orientation of thymio and goal in image space ----------------
        thymio_x_img, thymio_y_img = compute_center(
            corners[marker4_thymio][0][0],  # TL
            corners[marker4_thymio][0][2]   # BR
        )
        goal_x_img, goal_y_img = compute_center(
            corners[marker5_goal][0][0],    # TL
            corners[marker5_goal][0][2]     # BR
        )
        thymio_theta = compute_angle(
            corners[marker4_thymio][0][0],  # TL
            corners[marker4_thymio][0][3]   # BL
        )

        # ---------------- Homography: image -> world ----------------
        # Image/pixel coordinates of the 16 corners (4 markers × 4 corners)
        pts_img_TL = corners[marker0_TL][0].astype(np.float32)  # (4,2)
        pts_img_TR = corners[marker1_TR][0].astype(np.float32)  # (4,2)
        pts_img_BR = corners[marker2_BR][0].astype(np.float32)  # (4,2)
        pts_img_BL = corners[marker3_BL][0].astype(np.float32)  # (4,2)

        pts_img = np.vstack([pts_img_TL, pts_img_TR, pts_img_BR, pts_img_BL]).astype(np.float32)  # (16,2)

        # Homography matrix (pts_world must be shape (16,2), matching the order above)
        H, inliers = cv2.findHomography(pts_img, pts_world)  # method=0: DLT

        # Compute position of thymio / goal / obstacles in real world (origin at top left corner of map)
        thymio_x, thymio_y = img_to_world(thymio_x_img, thymio_y_img, H)
        thymio_start = np.array([thymio_x, thymio_y])


        if(not only_thymio):
            goal_x, goal_y = img_to_world(goal_x_img, goal_y_img, H)
            goal = np.array([goal_x, goal_y])

            polygons_img_world = detect_red_polygons_in_map(frame, p_tl, p_br)

            polygons_real_world = []
            # Map polygon vertices to real world positions
            for verts in polygons_img_world:  # verts: (N,2) in image coords
                cnt = verts.reshape(-1, 1, 2).astype(np.float32)    # cv2 uses (N,1,2) format
                cnt_world = cv2.perspectiveTransform(cnt, H)        # same as img_to_world but batched
                polygons_real_world.append(cnt_world.reshape(-1, 2) * 1e-2)  # list of (N,2) world coords
            #print(polygons_real_world)
            #polygons_real_world = np.array(polygons_real_world)  # (M,N,2) where M is the number of polygons

        if plot_view:
          #  ---------------- Display Map,Thymio,Goal,Polygons ----------------

          # Draw virtual map
          cv2.line(frame, p_tl, p_tr, (0, 0, 0), 2)
          cv2.line(frame, p_tr, p_br, (0, 0, 0), 2)
          cv2.line(frame, p_br, p_bl, (0, 0, 0), 2)
          cv2.line(frame, p_bl, p_tl, (0, 0, 0), 2)

          cv2.aruco.drawDetectedMarkers(frame, corners, ids)

          # # Thymio position / angle
          putText(frame, f"{thymio_theta:.2f}", thymio_x_img, thymio_y_img)
          putText(frame, f"({thymio_x:.2f}, {thymio_y:.2f})",thymio_x_img, thymio_y_img + 15)  # +15 to move a bit down

          if(not only_thymio):
              # Display Goal position
              #putText(frame, f"({goal_x:.2f}, {goal_y:.2f})", goal_x_img, goal_y_img)

              # Display detected polygons in image space
              for verts in polygons_img_world:
                  cv2.polylines(frame, [verts], True, (255, 0, 255), 2)  # connect vertices
                  for (x, y) in verts: #  verts is (N,2) where N is the number of vertices in a polygon
                      cv2.circle(frame, (int(x), int(y)), 4, (0, 0, 255), -1)  # circle at each vertex

    if only_thymio:
        return thymio_start * 1e-2, thymio_theta
    else:
        return thymio_start * 1e-2, thymio_theta, goal * 1e-2, polygons_real_world, H
    


def get_pose_from_frame(frame, only_thymio=False):
    """
    Returns:
        thymio_x, thymio_y, thymio_theta: position (x,y) in cm and orientation theta in radians
    If only_thymio is False then in addition to above returns:
        goal_x, goal_y: position (x,y) in cm
        list of (N,2) int arrays with polygon vertices in world coords
    """
    # Convert to grayscale (Aruco works on grayscale)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # gray is an array of shape (height, width, 1) type uint8

    # Detect markers
    # corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)
    detector = cv2.aruco.ArucoDetector(dictionary, detector_params)
    corners, ids, rejected = detector.detectMarkers(gray)


    # Initialize outputs
    thymio_start = thymio_theta = goal = polygons_real_world = H = None

    # 0: TL, 1: TR, 2: BR, 3: BL, 4: thymio, 5: goal
    marker0_TL = marker1_TR = marker2_BR = marker3_BL = marker4_thymio = marker5_goal = None

    ids_flat = ids.flatten()
    for i, marker_id in enumerate(ids_flat):
        if marker_id == 0:
            marker0_TL = i
        elif marker_id == 1:
            marker1_TR = i
        elif marker_id == 2:
            marker2_BR = i
        elif marker_id == 3:
            marker3_BL = i
        elif marker_id == 4:
            marker4_thymio = i
        elif marker_id == 5:
            marker5_goal = i
    essential_markers_detected = (marker0_TL is not None) and (marker1_TR is not None) and \
                                 (marker2_BR is not None) and (marker3_BL is not None) and (marker4_thymio is not None) 

    # ---------------- Marker detection and map definition ----------------
    if ids is not None and essential_markers_detected: #

        # 0: TL, 1: TR, 2: BR, 3: BL, 4: thymio, 5: goal
        marker0_TL = marker1_TR = marker2_BR = marker3_BL = marker4_thymio = marker5_goal = None
    
        ids_flat = ids.flatten()
        for i, marker_id in enumerate(ids_flat):
            if marker_id == 0:
                marker0_TL = i
            elif marker_id == 1:
                marker1_TR = i
            elif marker_id == 2:
                marker2_BR = i
            elif marker_id == 3:
                marker3_BL = i
            elif marker_id == 4:
                marker4_thymio = i
            elif marker_id == 5:
                marker5_goal = i
                
        # Interior corner of each map marker (assuming OpenCV order TL=0, TR=1, BR=2, BL=3)
        tl_pt = corners[marker0_TL][0][2]  # BR of TL marker (closest to map interior)
        tr_pt = corners[marker1_TR][0][3]  # BL of TR marker
        br_pt = corners[marker2_BR][0][0]  # TL of BR marker
        bl_pt = corners[marker3_BL][0][1]  # TR of BL marker

        # Four corners of the virtual map (axis aligned)
        p_tl = (int(tl_pt[0]), int(tl_pt[1]))
        p_br = (int(br_pt[0]), int(br_pt[1]))
        p_tr = (int(tr_pt[0]), int(tr_pt[1]))
        p_bl = (int(bl_pt[0]), int(bl_pt[1]))

        # ---------------- Position/orientation of thymio and goal in image space ----------------
        thymio_x_img, thymio_y_img = compute_center(
            corners[marker4_thymio][0][0],  # TL
            corners[marker4_thymio][0][2]   # BR
        )
        thymio_theta = compute_angle(
            corners[marker4_thymio][0][0],  # TL
            corners[marker4_thymio][0][3]   # BL
        )
        if marker5_goal is not None:
            goal_x_img, goal_y_img = compute_center(
                corners[marker5_goal][0][0],    # TL
                corners[marker5_goal][0][2]     # BR
            )

        # ---------------- Homography: image -> world ----------------
        # Image/pixel coordinates of the 16 corners (4 markers × 4 corners)
        pts_img_TL = corners[marker0_TL][0].astype(np.float32)  # (4,2)
        pts_img_TR = corners[marker1_TR][0].astype(np.float32)  # (4,2)
        pts_img_BR = corners[marker2_BR][0].astype(np.float32)  # (4,2)
        pts_img_BL = corners[marker3_BL][0].astype(np.float32)  # (4,2)

        pts_img = np.vstack([pts_img_TL, pts_img_TR, pts_img_BR, pts_img_BL]).astype(np.float32)  # (16,2)

        # Homography matrix (pts_world must be shape (16,2), matching the order above)
        H, inliers = cv2.findHomography(pts_img, pts_world)  # method=0: DLT

        # Compute position of thymio / goal / obstacles in real world (origin at top left corner of map)
        thymio_x, thymio_y = img_to_world(thymio_x_img, thymio_y_img, H)
        thymio_start = np.array([thymio_x, thymio_y])
        
        if(not only_thymio):
            if marker5_goal is not None:
                goal_x, goal_y = img_to_world(goal_x_img, goal_y_img, H)
                goal = np.array([goal_x, goal_y])
            
            polygons_img_world = detect_red_polygons_in_map(frame, p_tl, p_br)

            polygons_real_world = []
            # Map polygon vertices to real world positions
            for verts in polygons_img_world:  # verts: (N,2) in image coords
                cnt = verts.reshape(-1, 1, 2).astype(np.float32)    # cv2 uses (N,1,2) format
                cnt_world = cv2.perspectiveTransform(cnt, H)        # same as img_to_world but batched
                polygons_real_world.append(cnt_world.reshape(-1, 2))  # list of (N,2) world coords
            
        #  ---------------- Display Map,Thymio,Goal,Polygons ----------------

        # Draw virtual map
        cv2.line(frame, p_tl, p_tr, (0, 0, 0), 2)
        cv2.line(frame, p_tr, p_br, (0, 0, 0), 2)
        cv2.line(frame, p_br, p_bl, (0, 0, 0), 2)
        cv2.line(frame, p_bl, p_tl, (0, 0, 0), 2)

        #cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        # # Thymio position / angle
        putText(frame, f"{thymio_theta:.2f}", thymio_x_img, thymio_y_img)
        putText(frame, f"({thymio_x:.2f}, {thymio_y:.2f})",thymio_x_img, thymio_y_img + 15)  # +15 to move a bit down

        if(not only_thymio):
            # Display Goal position
            #putText(frame, f"({goal_x:.2f}, {goal_y:.2f})", goal_x_img, goal_y_img)
            
            # Display detected polygons in image space
            for verts in polygons_img_world:
                cv2.polylines(frame, [verts], True, (255, 0, 255), 2)  # connect vertices
                for (x, y) in verts: #  verts is (N,2) where N is the number of vertices in a polygon
                    cv2.circle(frame, (int(x), int(y)), 4, (0, 0, 255), -1)  # circle at each vertex

    if only_thymio:
        return thymio_start, thymio_theta
    else:
        return thymio_start, thymio_theta, goal, polygons_real_world, H