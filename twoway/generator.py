import numpy as np
import subprocess

net_file= 'twoway/two_intersection/two_intersection.net.xml'
routes_file = 'twoway/two_intersection/randomRoutes.rou.xml' # Path to the SUMO routes file.
trips_file = 'twoway/two_intersection/randomTrips.trips.xml' # Path to the SUMO routes file.

class TrafficGenerator:
    def __init__(self, max_steps):
        # self._n_cars_generated = n_cars_generated  # how many cars per episode
        self._max_steps = max_steps

    def generate_routes(self, begin_time=0, trip_probability=1):
        end_time = begin_time + self._max_steps
        random_trips_script = 'twoway/randomTrips.py'

        # Step 1: Generate random trips
        generate_trips_command = [
            'python', random_trips_script,
            '-n', net_file,
            '-r', routes_file,
            '--seed', str(np.random.randint(0, 10000)),
            '-b', str(begin_time),
            '-e', str(end_time),
            '-p', str(trip_probability)
        ]

        # Execute the command to generate random trips
        subprocess.run(generate_trips_command, check=True)

        # # Step 2: Convert trips to routes
        # duarouter_command = [
        #     'duarouter',
        #     '-n', net_file,
        #     '-t', trips_file,
        #     '-o', routes_file
        # ]
        #
        # # Execute the command to convert trips to routes
        # subprocess.run(duarouter_command, check=True)
