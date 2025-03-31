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
        self._num_states= 17
        self._num_actions=6
        self.observation_space = spaces.Box(low=0, high=100, shape=(self._num_states,), dtype=np.float32)
        self.action_space = spaces.Discrete(6)
        self.phase_dict = {
            0: "ggggrrrrggggrrrr",
            1: "rrrrggggrrrrgggg"
        }
        self._step = 0

    def reset(self):
        if traci.isLoaded():
            traci.close()
        traci.start(self.sumo_cmd)
        self._step = 0
        # Let the simulation run for some steps to initialize.

        return {}

    def _get_state(self):
        trafficlights = traci.trafficlight.getIDList()
        halted_counts = []  # Halted vehicle counts for each lane
        moving_counts = []
        self.lanes = list(set(traci.trafficlight.getControlledLanes(trafficlights[0])))
        self.lanes.sort()  # Optional: sort for consistency
        for lane in self.lanes:
            halted_counts.append(traci.lane.getLastStepHaltingNumber(lane))
            moving_counts.append(traci.lane.getLastStepVehicleNumber(lane))

        # Get the phase for the traffic light (only one value)
        color_phase = traci.trafficlight.getRedYellowGreenState(trafficlights[0])
        phase = next((k for k, v in self.phase_dict.items() if v == color_phase), -1)

        # Construct state vector: for each lane, add halted count and density, then add phase
        state_vector = [phase] + halted_counts + moving_counts
        return np.array(state_vector, dtype=np.float32)

    def step(self, action=None):
        traci.simulationStep()
        self._step += 1
        self.current_state = self._get_state()
        reward = self._get_reward()
        done = self._step >= self.max_steps
        return self.current_state, reward, done, {}

    def _get_reward(self):
        # Assume state layout: [halted_counts for num_lanes, densities for num_lanes, current_phase, current_phase_time]
        num_lanes = len(self.lanes)  # adjust as needed
        phase = np.array(self.current_state[0])
        halted_counts = np.array(self.current_state[1:1 + num_lanes])
        vehicle_counts = np.array(self.current_state[1 + num_lanes:1 + num_lanes + num_lanes])
        sum_halted = 0
        non_zero_halted = []
        for i in halted_counts:
            sum_halted += i / 100
            if i > 0:
                non_zero_halted.append(i)
        std_dev = np.std(non_zero_halted) if len(non_zero_halted) > 0 else 0.0
        imbalance_penalty = std_dev
        congestion_penalty = sum_halted
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
        state, reward, done, _ = env.step()
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
