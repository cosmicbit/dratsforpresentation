import os
import sys

import gym
import numpy as np

from gym import spaces

# import save_data
from twoway.visualization import Visualization

# Set SUMO_HOME and update system path (adjust the path if needed)
if "SUMO_HOME" not in os.environ:
    os.environ["SUMO_HOME"] = "/usr/share/sumo"
sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))
import traci


class FixedTimeEnv(gym.Env):
    """
    A SUMO-based Gym environment that uses SUMO's built-in fixed-time (traditional) traffic light logic.

    Assumes a four-way intersection with:
      - Four incoming lanes with IDs: "edge_n_in_0", "edge_s_in_0", "edge_e_in_0", and "edge_w_in_0".
      - A traffic light junction "TL1" defined in your network file.

    The state vector is:
        [veh_n, veh_s, veh_e, veh_w, current_phase]
    where veh_* is the vehicle count on the corresponding incoming lane,
    and current_phase is the current phase of the traffic light.

    Since the fixed-time controller does not intervene (the tlLogic is predefined),
    no actions are taken. The simulation simply advances.

    Reward (optional): negative sum of vehicle counts (for logging purposes).
    """

    def __init__(self, sumo_cmd, max_steps):
        super(FixedTimeEnv, self).__init__()
        # Restart simulation if already connected.

        self.step_count = 0
        self.max_steps = max_steps
        self.sumo_cmd = sumo_cmd
        # Assume each intersection contributes 4 halting counts + 1 phase indicator (total=5 per intersection).
        # For 2 intersections, the state length is 10.
        self.observation_space = spaces.Box(low=0, high=100, shape=(10,), dtype=np.float32)
        # Joint action space: 4 discrete actions (0 to 3)
        self.action_space = spaces.Discrete(4)

        # Define phase dictionary for both intersections (assuming identical phases)
        self.phase_dict = {
            0: "GGGggrrrrrGGGggrrrrr",
            1: "rrrrrGGGggrrrrrGGGgg"
        }
        self.step_count = 0

    def reset(self):
        if traci.isLoaded():
            traci.close()
        # TG = TrafficGenerator(max_steps=self.max_steps)
        # TG.generate_routes()
        traci.start(self.sumo_cmd)
        for i in range(50):
            traci.simulationStep()
            self.step_count += 1
        return self._get_state()

    def _get_state(self):
        """
        Constructs the state vector by iterating over all traffic lights.
        For each light, collects halting vehicle counts from the controlled lanes and its current phase.
        """
        trafficlights = traci.trafficlight.getIDList()
        halting_numbers_all = []
        phases_all = []

        for tl_id in trafficlights:
            controlled_lanes = set(traci.trafficlight.getControlledLanes(tl_id))
            edge_ids = set(traci.lane.getEdgeID(lane) for lane in controlled_lanes)
            edge_ids = list(edge_ids)
            edge_ids.sort()
            halting_numbers = [traci.edge.getLastStepHaltingNumber(edge) for edge in edge_ids]
            halting_numbers_all.extend(halting_numbers)

            color_phase = traci.trafficlight.getRedYellowGreenState(tl_id)
            phase = next((k for k, v in self.phase_dict.items() if v == color_phase), -1)
            phases_all.append(phase)

        # Concatenate halting numbers and phase indicators into one state vector.
        state = halting_numbers_all + phases_all
        return np.array(state, dtype=np.float32)

    def step(self, action=None):
        traci.simulationStep()
        self.step_count += 1
        next_state = self._get_state()
        reward = self._get_reward(next_state)
        done = self.step_count >= self.max_steps

        return next_state, reward, done, {}

    def _get_reward(self, state):
        """
        Computes reward based on the halting vehicles from both intersections.
        The reward is the negative sum of halting counts plus a penalty proportional to the standard deviation,
        and an additional heavy penalty for extreme imbalances.
        """
        # Assuming each intersection contributes 4 halting counts.
        num_intersections = 2
        num_counts_per_intersection = 4
        halting_vehicles = []
        for i in range(num_intersections):
            start_idx = i * num_counts_per_intersection
            end_idx = start_idx + num_counts_per_intersection
            halting_vehicles.extend(state[start_idx:end_idx])

        halting_vehicles = np.array(halting_vehicles)
        total_halt = np.sum(halting_vehicles)
        std_dev_halt = np.std(halting_vehicles)
        beta = 0.9

        reward = - total_halt - beta * std_dev_halt

        # Apply a heavy penalty if there is an extreme imbalance.
        if np.max(halting_vehicles) - np.min(halting_vehicles) > 30:
            reward -= 100

        # save_data.save_state_to_csv(self.step_count, state, total_halt, std_dev_halt, beta, reward,
        #                             filename="data_from_ttl.csv")
        return reward

    def close(self):
        if traci.isLoaded():
            traci.close()


def run_simulation(sumo_cmd = ["sumo-gui", "-c", "twoway/two_intersection/two_intersection.sumocfg"], max_steps=2000, print_interval=50):
    env = FixedTimeEnv(sumo_cmd, max_steps=max_steps)
    state = env.reset()
    rewards=[]
    waiting_time_list = []
    print("Starting Fixed-Time (TTL) Simulation")
    for step in range(max_steps):
        state, reward, done, _ = env.step()
        rewards.append(reward)
        veh_ids = traci.vehicle.getIDList()
        if veh_ids:
            avg_wait = np.mean([traci.vehicle.getWaitingTime(veh) for veh in veh_ids])
        else:
            avg_wait = 0.0
        waiting_time_list.append(avg_wait)

        if step % print_interval == 0:
            print(f"Step {step}: State = {state}, Reward = {reward}")
        if done:
            break
        # Sleep a little if you want to slow down the simulation
        #time.sleep(0.05)
    env.close()
    print("Simulation Completed.")
    print(f"Total reward of simulation : {np.sum(rewards)}")
    print(f"Mean reward of simulation: {np.mean(rewards)}")
    viz = Visualization()
    # Plot the reward at each step
    viz.save_data_and_plot(data=rewards, filename='ROT TTL', xlabel='Time Steps', ylabel='Total rewards')

    return rewards, waiting_time_list

if __name__ == "__main__":
    # Set the SUMO command to use SUMO-gui (for visualization)
    # Ensure that "fourway_intersection.sumocfg" exists in your working directory.
    sumo_cmd = ["sumo-gui", "-c", "two_intersection/two_intersection.sumocfg"]
    run_simulation(sumo_cmd, total_steps=1000, print_interval=50)
