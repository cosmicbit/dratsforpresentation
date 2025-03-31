import os
import sys
import numpy as np
from gym import spaces
import traci


# Set SUMO_HOME and update system path (adjust the path if needed)
if "SUMO_HOME" not in os.environ:
    os.environ["SUMO_HOME"] = "/usr/share/sumo"
sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))



class FixedTimeEnv():

    def __init__(self, sumo_cmd, max_steps):
        self.sumo_cmd = sumo_cmd
        self.max_steps = max_steps
        self.observation_space = spaces.Box(low=0, high=100, shape=(5,), dtype=np.float32)
        # Action space is dummy; no actions will be taken in fixed-time control.
        self.action_space = spaces.Discrete(2)
        self.phase_dict = {
            0: "ggggrrrrggggrrrr",
            1: "rrrrggggrrrrgggg"
        }
        self.step_count = 0

    def reset(self):
        if traci.isLoaded():
            traci.close()
        traci.start(self.sumo_cmd)
        self.step_count = 0
        # Let the simulation run for some steps to initialize.
        for i in range(50):
            traci.simulationStep()
        return self._get_state()

    def _get_state(self):
        trafficlights = traci.trafficlight.getIDList()
        densities = []  # Normalized traffic density for each lane

        # Estimated average vehicle length in meters (adjust as needed)
        avg_vehicle_length = 5.0

        # For simplicity, assume one traffic light (as in your case)
        # Get the lane IDs controlled by the traffic light
        controlled_lanes = list(set(traci.trafficlight.getControlledLanes(trafficlights[0])))
        controlled_lanes.sort()  # Optional: sort for consistency

        for lane in controlled_lanes:

            # Get the total number of vehicles on this lane
            vehicle_count = traci.lane.getLastStepVehicleNumber(lane)
            # Get lane length in meters
            lane_length = traci.lane.getLength(lane)
            # Estimate capacity as lane length divided by average vehicle length
            capacity = lane_length / avg_vehicle_length if lane_length > 0 else 1.0
            # Compute normalized density
            density = vehicle_count / capacity
            densities.append(density)

        # Get the phase for the traffic light (only one value)
        color_phase = traci.trafficlight.getRedYellowGreenState(trafficlights[0])
        phase = next((k for k, v in self.phase_dict.items() if v == color_phase), -1)

        # Construct state vector: for each lane, add halted count and density, then add phase
        state_vector = densities + [phase]

        return np.array(state_vector, dtype=np.float32)

    def step(self, action=None):
        traci.simulationStep()
        self.step_count += 1
        next_state = self._get_state()
        reward = self._get_reward(next_state)
        done = self.step_count >= self.max_steps
        return next_state, reward, done, {}

    def _get_reward(self, state):
        # Assume state layout: [halted_counts for num_lanes, densities for num_lanes, current_phase, current_phase_time]
        num_lanes = 8  # adjust as needed
        # halted_counts = np.array(state[:num_lanes])
        densities = np.array(state[:num_lanes])
        # total_halt = np.sum(halted_counts)
        avg_density = np.mean(densities) if densities.size > 0 else 0.0
        reference_density = 0.3
        std_dev = np.std(densities) if len(densities) > 0 else 0.0
        beta = 0.5  # reduced from 0.5 to soften the penalty slightly
        imbalance_penalty = beta * std_dev
        gamma = 0.9
        congestion_penalty = avg_density * gamma / reference_density
        # Combine all reward components:
        reward = (- imbalance_penalty - congestion_penalty)

        return reward

    def close(self):
        if traci.isLoaded():
            traci.close()


def run_simulation(sumo_cmd = ["sumo-gui", "-c", "oneway/one_intersection/first.sumocfg"], total_steps=1000):
    env = FixedTimeEnv(sumo_cmd, max_steps=total_steps)
    state = env.reset()
    rewards = []
    waiting_time_list = []  # Average waiting time per step.
    print("Starting Fixed-Time (TTL) Simulation")
    for step in range(total_steps):
        _, reward, done, _ = env.step()
        rewards.append(reward)

        # Waiting time: Compute the average waiting time for all vehicles currently in simulation.
        veh_ids = traci.vehicle.getIDList()
        if veh_ids:
            avg_wait = np.mean([traci.vehicle.getWaitingTime(veh) for veh in veh_ids])
        else:
            avg_wait = 0.0
        waiting_time_list.append(avg_wait)
        if done:
            break

    env.close()
    total_reward = np.sum(rewards)
    mean_wait = np.mean(waiting_time_list)
    print("Simulation Completed.")
    print(f"Total Reward: {total_reward:.2f}")
    print(f"Mean Waiting Time: {mean_wait:.2f}")
    return rewards, waiting_time_list

if __name__ == "__main__":
    # Use SUMO-gui mode for visualization; adjust the config file as needed.
    sumo_cmd = ["sumo-gui", "-c", "one_intersection/first.sumocfg"]
    run_simulation(sumo_cmd, total_steps=1000, print_interval=50)
