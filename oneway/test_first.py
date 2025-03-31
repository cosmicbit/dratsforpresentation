import numpy as np
import torch
from oneway.agent_first import DQNAgent
import gym
import traci
from gym import spaces


##############################
# SUMO-Based Gym Environment #
##############################
class SUMOTrafficEnv(gym.Env):

    def __init__(self, sumo_cmd, max_steps):
        super(SUMOTrafficEnv, self).__init__()
        self.sumo_cmd = sumo_cmd
        self.net_file = "one_intersection/network.net.xml"
        self.max_steps = max_steps
        # State: 4 vehicle counts + traffic light phase
        self.observation_space = spaces.Box(low=0, high=100, shape=(9,), dtype=np.float32)
        self.num_duration_bins = 3  # e.g., durations: 10, 15, 20, ..., 60 seconds
        self.action_space = spaces.MultiDiscrete([2, self.num_duration_bins])
        self.phase_dict = {
            0: "GGGggrrrrrGGGggrrrrr",
            1: "rrrrrGGGggrrrrrGGGgg"
        }
        # Define your discrete durations (e.g., 10 to 60 seconds in steps of 5)
        self.durations = [25, 30, 35]
        self.step_count = 0
        # self.writer = writer
        self.current_phase_time = 0
        self.current_phase = 0  # Track the active phase

    def reset(self):
        # If TraCI is already connected, close it.
        if traci.isLoaded():
            traci.close()
        # generate_route_file(self.net_file, filename="one_intersection/routes.rou.xml")
        traci.start(self.sumo_cmd)

        self.step_count = 0
        self.prev_total_halt = 0
        self.phase_switched = False
        # Set initial phase to 0 for traffic light "TL1"
        phase_str = self.phase_dict[1]
        traci.trafficlight.setRedYellowGreenState("TL1", phase_str)
        for i in range(50):
            traci.simulationStep()
        return self._get_state()

    def _get_state(self):
        trafficlights = traci.trafficlight.getIDList()
        halted_counts = []  # Halted vehicle counts for each lane
        densities = []  # Normalized traffic density for each lane

        # Estimated average vehicle length in meters (adjust as needed)
        avg_vehicle_length = 5.0

        # For simplicity, assume one traffic light (as in your case)
        # Get the lane IDs controlled by the traffic light
        controlled_lanes = list(set(traci.trafficlight.getControlledLanes(trafficlights[0])))
        controlled_lanes.sort()  # Optional: sort for consistency

        for lane in controlled_lanes:
            # Get the number of halted vehicles on this lane
            halted = traci.lane.getLastStepHaltingNumber(lane)
            halted_counts.append(halted)

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

    def step(self, action):
        # print("step:", self.step_count)
        self._apply_action(action)
        next_state = self._get_state()
        reward = self._get_reward(next_state)
        done = self.step_count >= self.max_steps
        info = {
            'step_count': self.step_count,
        }
        return next_state, reward, done, info

    def _apply_action(self, action):
        phase_decision, duration_index = action  # Unpack the multidiscrete action
        chosen_duration = self.durations[duration_index]

        self.phase_switched = False
        if phase_decision == 1:
            self.phase_switched = True
            color_phase = traci.trafficlight.getRedYellowGreenState("TL1")
            phase = next((k for k, v in self.phase_dict.items() if v == color_phase), -1)
            new_phase = (phase + 1) % 2  # Assuming two phases in the cycle
            phase_str = self.phase_dict[new_phase]
            traci.trafficlight.setRedYellowGreenState("TL1", phase_str)
            # Reset phase timer when switching phases
            self.current_phase_time = 0
            self.current_phase = new_phase
        else:
            # If not switching, you might want to extend or modify the phase duration
            # Here, we simply consider the chosen duration as the new "target" duration
            pass
        for i in range(chosen_duration):
            traci.simulationStep()
            self.step_count += 1
            self.current_phase_time += 1

    def _get_reward(self, state):
        # Assume state layout: [halted_counts for num_lanes, densities for num_lanes, current_phase, current_phase_time]
        num_lanes = 8  # adjust as needed
        densities = np.array(state[:num_lanes])
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


def test_agent(sumo_cmd = ["sumo-gui", "-c", "oneway/one_intersection/first.sumocfg"],
               episodes=1):
    """
    Runs the trained agent for a given number of episodes and measures:
      - Rewards (as before)
      - Throughput (cumulative count of vehicles that have arrived per environment step)
      - Average waiting time (averaged over the simulation steps within an environment step)
    """
    env = SUMOTrafficEnv(sumo_cmd, max_steps=1000)
    state_dim = env.observation_space.shape[0]
    phase_n = 2
    duration_n = 3
    # print("action_n", action_n)
    loaded_agent = DQNAgent(state_dim, phase_n, duration_n)
    loaded_agent.q_network.load_state_dict(torch.load("oneway/models/first_model.pth",
                                                      map_location=loaded_agent.device))
    loaded_agent.update_target_network()
    loaded_agent.epsilon = 0.0

    for ep in range(episodes):
        state = env.reset()
        rewards = []
        throughput_list = []  # List of cumulative arrivals per environment step
        cumulative_throughput = 0
        waiting_time_list = []  # List of average waiting times per environment step
        done = False
        step=0
        while not done:
            step += 1
            action = loaded_agent.act(state)
            state, reward, done, info = env.step(action)
            rewards.append(reward)
            veh_ids = traci.vehicle.getIDList()
            if veh_ids:
                avg_wait = np.mean([traci.vehicle.getWaitingTime(veh) for veh in veh_ids])
            else:
                avg_wait = 0.0
            waiting_time_list.append(avg_wait)

        total_reward = np.sum(rewards)
        print(f"Test Episode {ep + 1} completed.")
        print(f"Total Reward: {total_reward:.2f}")
        # print(f"Final Cumulative Throughput: {throughput_list[-1]}")
        print(f"Mean Waiting Time: {np.mean(waiting_time_list):.2f}")
    env.close()

    return rewards, waiting_time_list

if __name__ == "__main__":
    sumo_cmd = ["sumo-gui", "-c", "one_intersection/first.sumocfg"]
    rewards, waiting_time_list = test_agent(sumo_cmd, episodes=1)
    ttl.run_simulation(sumo_cmd, total_steps=1000, print_interval=50)

