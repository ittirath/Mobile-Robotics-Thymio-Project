from extremitypathfinder import PolygonEnvironment
from extremitypathfinder.plotting import PlottingEnvironment


environment = PolygonEnvironment()
# counter clockwise vertex numbering!
boundary_coordinates = [(0.0, -2.0), (10.0, -2.0), (9.0, 5.0), (10.0, 10.0), (0.0, 10.0)]
# clockwise numbering!
list_of_holes = [
    [
        (3.0, 7.0),
        (5.0, 9.0),
        (4.5, 7.0),
        (5.0, 4.0),
    ],
    [
        (3.0,0.0),
        (4.0,2.0),
        (6.0,0.0),
    ],
]


environment = PlottingEnvironment(plotting_dir="path/to/plots")
environment.store(boundary_coordinates, list_of_holes, validate=False)
start_coordinates = (4.5, -1.0)
goal_coordinates = (4.0, 9)
path, length = environment.find_shortest_path(start_coordinates, goal_coordinates)