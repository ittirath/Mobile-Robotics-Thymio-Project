import warnings
import time

import numpy as np
import cv2
import math
import matplotlib
import matplotlib.pyplot as plt
import heapq

from global_variables import *


# PATH PLANNING HELPER FUNCTIONS ----------------------------------------

def build_poly_edges(polygons):
    poly_edges = []
    for verts in polygons:
        n = len(verts)
        edges_for_poly = []
        for i in range(n):
            x1, y1 = verts[i]
            x2, y2 = verts[(i + 1) % n]   # wrap-around
            edges_for_poly.append([x1, y1, x2, y2])
        poly_edges.append(edges_for_poly)
    return poly_edges

def offset_polygon(vertices, L):
    """
    Agrandit un polygone CCW d'un offset L vers l'extérieur.
    
    vertices : liste de (x, y)
    L        : distance d'offset
    retourne : nouvelle liste de sommets (x, y)
    """
    V = np.asarray(vertices, dtype=float)
    n = len(V)
    out = []

    for i in range(n):
        p_prev = V[(i - 1) % n]
        p_curr = V[i]
        p_next = V[(i + 1) % n]

        # Vecteurs des 2 arêtes autour de p_curr
        e1 = p_curr - p_prev
        e2 = p_next - p_curr

        n1 = np.linalg.norm(e1)
        n2 = np.linalg.norm(e2)
        if n1 < 1e-12 or n2 < 1e-12:
            # Cas dégénéré : on recopie tel quel
            out.append(tuple(p_curr))
            continue

        t1 = e1 / n1  # direction de l'arête (p_prev -> p_curr)
        t2 = e2 / n2  # direction de l'arête (p_curr -> p_next)

        # Normales extérieures pour un polygone CCW (droite de la direction)
        n_1 = np.array([t1[1], -t1[0]])
        n_2 = np.array([t2[1], -t2[0]])

        # Points sur les droites offsetées
        Q1 = p_curr + L * n_1
        Q2 = p_curr + L * n_2

        # Résoudre Q1 + s*t1 = Q2 + t*t2
        A = np.column_stack((t1, -t2))
        b = Q2 - Q1

        try:
            s, _ = np.linalg.solve(A, b)
            new_pt = Q1 + s * t1
        except np.linalg.LinAlgError:
            # Arêtes quasi-parallèles : on moyenne les normales
            n_avg = n_1 + n_2
            if np.linalg.norm(n_avg) < 1e-12:
                n_avg = n_1
            n_avg = n_avg / np.linalg.norm(n_avg)
            new_pt = p_curr + L * n_avg

        out.append(new_pt)

    return np.array(out)

def inflate_object(polygon, scale):
    """
    Uniformly scales a polygon around its centroid.

    polygon : (N, 2) ndarray  (vertices in order)
    scale   : float > 1       (how much larger you want it)

    returns : (N, 2) ndarray of inflated vertices
    """
    poly = np.asarray(polygon, dtype=float)
    center = poly.mean(axis=0)          # centroid
    #inflated = center + scale * (poly - center)
    inflated = center + scale * (poly - center)

    return inflated

# -------------------- Geometry helpers --------------------

def point_on_segment(px, py, x1, y1, x2, y2, eps=1e-9):
    # check if P is on segment [P1,P2]
    vx, vy = x2 - x1, y2 - y1
    wx, wy = px - x1, py - y1

    # cross product == 0  -> colinear
    if abs(vx * wy - vy * wx) > eps:
        return False

    # dot product between 0 and |v|^2 -> within segment bounds
    dot = vx * wx + vy * wy
    if dot < -eps:
        return False

    v2 = vx * vx + vy * vy
    if dot > v2 + eps:
        return False

    return True


def point_in_polygon_strict(point, verts, eps=1e-9):
    """
    True if point is strictly inside polygon (not on boundary).
    verts: (N,2) array.
    """
    x, y = point

    # 1) boundary check: if on any edge -> not strictly inside
    n = len(verts)
    for i in range(n):
        x1, y1 = verts[i]
        x2, y2 = verts[(i + 1) % n]
        if point_on_segment(x, y, x1, y1, x2, y2, eps):
            return False

    # 2) standard ray-casting for inside
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = verts[i]
        xj, yj = verts[j]
        # edge crosses horizontal ray?
        if ((yi > y) != (yj > y)):
            x_int = xi + (xj - xi) * (y - yi) / (yj - yi)
            if x < x_int:
                inside = not inside
        j = i
    return inside


def segment_intersection_params(e1, e2, eps=1e-9):
    """
    Return (t, s) if segments e1 and e2 intersect, else None.
    e1, e2 are [x1, y1, x2, y2].
    t is parameter on e1, s on e2.
    """
    x1, y1, x2, y2 = e1
    x3, y3, x4, y4 = e2

    dx1, dy1 = x2 - x1, y2 - y1
    dx2, dy2 = x4 - x3, y4 - y3

    A = np.array([[dx1, -dx2],
                  [dy1, -dy2]], dtype=float)
    b = np.array([x3 - x1,
                  y3 - y1], dtype=float)

    det = np.linalg.det(A)
    if abs(det) < eps:
        # parallel / colinear: ignore for visibility test
        return None

    t, s = np.linalg.solve(A, b)

    if -eps <= t <= 1.0 + eps and -eps <= s <= 1.0 + eps:
        return t, s
    return None

def all_intersections_on_candidate(candidate, obstacle_edges, eps=1e-6):
    """
    Return list of unique t in [0,1] where candidate intersects any obstacle edge.
    """
    ts = []
    for e in obstacle_edges:
        res = segment_intersection_params(candidate, e)
        if res is None:
            continue
        t, s = res

        # snap to exact endpoints if very close
        if abs(t) < eps:
            t = 0.0
        elif abs(t - 1.0) < eps:
            t = 1.0

        # deduplicate by t
        if not any(abs(t - t0) < eps for t0 in ts):
            ts.append(t)
    return ts

def is_valid_visibility_edge(candidate, obstacle_edges, polygons, eps=1e-6):
    """
    Edge is valid if:
      - its interior does not intersect any obstacle edge, AND
      - its interior does not lie inside any polygon.
    Intersections at endpoints are allowed.
    """
    # 1) boundary intersections (same as before)
    ts = all_intersections_on_candidate(candidate, obstacle_edges, eps)
    for t in ts:
        if eps < t < 1.0 - eps:   # interior intersection with boundary
            return False

    # 2) interior point inside any polygon?
    x1, y1, x2, y2 = candidate
    mx, my = (x1 + x2) / 2.0, (y1 + y2) / 2.0   # midpoint

    for poly in polygons:
        if point_in_polygon_strict((mx, my), poly, eps):
            return False

    return True

# -------------------- Dijkstra --------------------
def dijkstra(adj, start_idx, goal_idx):
    n = len(adj)
    INF = 1e18
    dist = [INF] * n
    prev = [-1] * n

    dist[start_idx] = 0.0
    pq = [(0.0, start_idx)]

    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u] + 1e-12:
            continue
        if u == goal_idx:
            break
        for v, w in adj[u]:
            nd = d + w
            if nd < dist[v] - 1e-12:
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))

    if dist[goal_idx] >= INF:
        return None, dist

    # reconstruct path
    path = []
    u = goal_idx
    while u != -1:
        path.append(u)
        u = prev[u]
    path.reverse()
    return path, dist
    

# PATH PLANNING GLOBAL FUNCTIONS ----------------------------------------

def get_objective_waypoints(start, goal, polygons):
  # Returns a list of the coordinates of the N waypoints that the robot much reach from the start coordinates, the goal coordinates and the polygon coordinates vector ()
  # The list is ordered in reverse order of visit of waypoints (ie the goal is the first set of coordinates, then the before last waypoint and so on)
  # Example get_objective_waypoints() = np.array([[x_N,y_N], [x_N-1,y_N-1], ..., [x_1,y_1]])
  # Inputs : start (2x1 numpy array of floats), goal (2x1 numpy array of floats), polygons (Px1 numpy array of numpy array of coordinates)
  # Output : waypoints (list of N arrays of floats)

  if polygons is None or len(polygons) == 0:
      return np.array([[start[0], goal[0]],
                       [start[1], goal[1]]])
    
  offset = 10.0 # dont know if we need to pu it in cm
  #inflated_polygons = [offset_polygon(poly,offset) for poly in polygons]
  inflated_polygons = [inflate_object(poly, scale_factor) for poly in polygons]
  polygons = inflated_polygons

  poly_edges = build_poly_edges(polygons)
  obstacle_edges = [e for poly in poly_edges for e in poly]

  # Nodes: start, goal, and all polygon vertices
  points = [start, goal] + [v for poly in polygons for v in poly]
  n_points = len(points)

  # Build graph
  valid_edges = []            # list of [x1,y1,x2,y2] for plotting
  adj = {i: [] for i in range(n_points)}  # adjacency list

  for i in range(n_points):
      for j in range(i + 1, n_points):
          x1, y1 = points[i]
          x2, y2 = points[j]
          candidate = [x1, y1, x2, y2]

          if is_valid_visibility_edge(candidate, obstacle_edges, polygons):
              valid_edges.append(candidate)
              w = float(np.linalg.norm(np.array(points[i]) - np.array(points[j])))
              adj[i].append((j, w))
              adj[j].append((i, w))

  # -------------------- Shortest path (Dijkstra) --------------------
  start_idx = 0  # start
  goal_idx  = 1  # goal
  path_indices, dist = dijkstra(adj, start_idx, goal_idx)
  print("Shortest distance:", dist[goal_idx])
  print("Path indices:", path_indices)

  objective_waypoints = []

  while len(path_indices) > 1:
    i = path_indices.pop()
    objective_waypoints.append(points[i])

  return objective_waypoints


def get_global_path(start, goal, polygons, plot=False):
    """
    Compute the shortest collision-free path between a start point and a goal point
    using a visibility graph with inflated obstacle polygons.

    Parameters
    ----------
    start : array-like of shape (2,)
        Starting point [x, y].
    goal : array-like of shape (2,)
        Goal point [x, y].
    polygons : list of ndarray
        List of polygons; each polygon is an array of shape (M,2) with its vertices.
    plot : bool, optional
        If True, displays the visibility graph and highlights the shortest path.

    Returns
    -------
    global_path : ndarray of shape (2, K) or None
        The optimal path as two rows:
            global_path[0, :] = x coordinates
            global_path[1, :] = y coordinates
        K is the number of waypoints on the shortest path.
        Returns None if no path exists.
    """
        
    if start is None or goal is None:
        return None  

    if polygons is None or len(polygons) == 0:
        return np.array([[start[0], goal[0]],
                         [start[1], goal[1]]])
    
    # offset = 10.0 # cm
    # inflated_polygons = [offset_polygon(poly,offset) for poly in polygons]
    inflated_polygons = [inflate_object(poly, scale_factor) for poly in polygons]
    polygons = inflated_polygons

    poly_edges = build_poly_edges(polygons)
    obstacle_edges = [e for poly in poly_edges for e in poly]

    # Nodes: start, goal, and all polygon vertices
    points = [start, goal] + [v for poly in polygons for v in poly]
    n_points = len(points)

    # Build graph
    valid_edges = []            # list of [x1,y1,x2,y2] for plotting
    adj = {i: [] for i in range(n_points)}  # adjacency list

    for i in range(n_points):
        for j in range(i + 1, n_points):
            x1, y1 = points[i]
            x2, y2 = points[j]
            candidate = [x1, y1, x2, y2]

            if is_valid_visibility_edge(candidate, obstacle_edges, polygons):
                valid_edges.append(candidate)
                w = float(np.linalg.norm(points[i] - points[j]))
                adj[i].append((j, w))
                adj[j].append((i, w))

    # -------------------- Shortest path (Dijkstra) --------------------
    start_idx = 0  # start
    goal_idx  = 1  # goal
    path_indices, dist = dijkstra(adj, start_idx, goal_idx)
    # print("Shortest distance:", dist[goal_idx])
    # print("Path indices:", path_indices)
    
    # -------------------- Plot --------------------
    if(plot == True):
        plt.figure()
        # Plot polygons (closed)
        for verts in polygons:
            xs = np.append(verts[:, 0], verts[0, 0])
            ys = np.append(verts[:, 1], verts[0, 1])
            plt.plot(xs, ys)

        # Plot start and goal
        plt.scatter(start[0], start[1], marker='o', color='C0')
        plt.scatter(goal[0],  goal[1],  marker='x', color='C1')
        plt.text(start[0] + 0.1, start[1], "start")
        plt.text(goal[0] + 0.1,  goal[1],  "goal")

        # Plot all valid visibility edges
        for x1, y1, x2, y2 in valid_edges:
            plt.plot([x1, x2], [y1, y2], 'g--', linewidth=0.5)

    global_path = None
    # Highlight shortest path
    if path_indices is not None:
        global_path = np.zeros((2, len(path_indices)))
        for k in range(len(path_indices) - 1):
            i = path_indices[k]
            j = path_indices[k + 1]
            x1, y1 = points[i]
            x2, y2 = points[j]
            if(plot == True):
                plt.plot([x1, x2], [y1, y2], 'r-', linewidth=2.5)
            # print(f"Path segment: ({x1:.2f}, {y1:.2f}) -> ({x2:.2f}, {y2:.2f})")
            global_path[:, k] = [x1, y1]
        global_path[:, -1] = points[path_indices[-1]] # last point

    if(plot == True):
        plt.gca().set_aspect('equal', adjustable='box')
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("Visibility graph with shortest path")
        plt.grid(True)
        plt.gca().invert_yaxis() # match image coords with origin at top-left (y increases downward)
        plt.savefig("navigation_path.png", dpi=200)

    return global_path