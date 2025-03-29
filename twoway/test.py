# test_sumo.py
import numpy as np
import torch
from twoway.agent import DQNAgent  # Import classes from your training module

from twoway.visualization import Visualization
from twoway.generator import TrafficGenerator
import traci
import gym
from gym import spaces


class SUMOTrafficMultiEnv(gym.Env):
    """
    A SUMO-based Gym environment for a network with two intersections (TL1 and TL2).

    Assumes a SUMO network with two traffic light junctions "TL1" and "TL2".
    The state vector is created by concatenating the halting vehicle counts from all controlled lanes
    for each traffic light, followed by the current phase indicator for each intersection.

    Action space (Discrete(4)):
      0: Keep current phases for both TL1 and TL2.
      1: Toggle TL1 only.
      2: Toggle TL2 only.
      3: Toggle both TL1 and TL2.

    Reward: Negative sum of halting vehicle counts plus a penalty for imbalance.
    """

    def __init__(self, sumo_cmd, max_steps):
        super(SUMOTrafficMultiEnv, self).__init__()
        self.sumo_cmd = sumo_cmd
        self.max_steps = max_steps

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
        # Restart simulation if already connected.
        if traci.isLoaded():
            traci.close()
        TG = TrafficGenerator(max_steps=self.max_steps)
        TG.generate_routes()
        traci.start(self.sumo_cmd)
        self.step_count = 0

        # Set initial phase for both intersections.
        initial_phase = self.phase_dict[1]  # Starting phase.
        traci.trafficlight.setRedYellowGreenState("TL1", initial_phase)
        traci.trafficlight.setRedYellowGreenState("TL2", initial_phase)

        # Run a few simulation steps to stabilize the environment.
        for _ in range(50):
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

    def step(self, action):
        self._apply_action(action)
        next_state = self._get_state()
        reward = self._get_reward(next_state)
        done = self.step_count >= self.max_steps
        info = {'step_count': self.step_count}
        return next_state, reward, done, info

    def _apply_action(self, action):
        """
        Decodes the joint action and applies phase toggling.
          - action 0 (00 binary): Do nothing for both TL1 and TL2.
          - action 1 (01 binary): Toggle TL1 only.
          - action 2 (10 binary): Toggle TL2 only.
          - action 3 (11 binary): Toggle both TL1 and TL2.
        """
        toggle_TL1 = action & 1  # Least significant bit controls TL1.
        toggle_TL2 = (action >> 1) & 1  # Next bit controls TL2.

        if toggle_TL1:
            color_phase = traci.trafficlight.getRedYellowGreenState("TL1")
            phase = next((k for k, v in self.phase_dict.items() if v == color_phase), -1)
            new_phase = (phase + 1) % 2
            phase_str = self.phase_dict[new_phase]
            traci.trafficlight.setRedYellowGreenState("TL1", phase_str)

        if toggle_TL2:
            color_phase = traci.trafficlight.getRedYellowGreenState("TL2")
            phase = next((k for k, v in self.phase_dict.items() if v == color_phase), -1)
            new_phase = (phase + 1) % 2
            phase_str = self.phase_dict[new_phase]
            traci.trafficlight.setRedYellowGreenState("TL2", phase_str)

        # Advance the simulation for a fixed number of steps after applying the action.
        for _ in range(30):
            traci.simulationStep()
            self.step_count += 1

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

        # save_data.save_state_to_csv(self.step_count, state, total_halt, std_dev_halt, beta, reward, filename="data_from_model.csv")
        return reward

    def close(self):
        if traci.isLoaded():
            traci.close()

def test_agent(sumo_cmd = ["sumo-gui", "-c", "twoway/two_intersection/two_intersection.sumocfg"], episodes=1, max_steps = 2000,print_interval=50):
    """
    Runs the trained agent for a given number of episodes in the SUMO simulation,
    using SUMO-gui so you can visually inspect the simulation.
    """
    env = SUMOTrafficMultiEnv(sumo_cmd, max_steps=max_steps)
    # For a two-intersection setup, the state dimension might now be 10.
    state_dim = env.observation_space.shape[0]
    # The joint action space is discrete with 4 actions (0: do nothing, 1: toggle TL1, 2: toggle TL2, 3: toggle both).
    action_n = env.action_space.n
    # Create a new DQN agent instance.
    loaded_agent = DQNAgent(state_dim, action_n)

    # Load the saved model parameters (ensure "models/dqn_multi_intersection_model.pth" exists)
    loaded_agent.q_network.load_state_dict(
        torch.load("twoway/models/dqn_multi_intersection_model.pth", map_location=loaded_agent.device))
    loaded_agent.update_target_network()
    loaded_agent.epsilon = 0.0  # Set to greedy policy (no exploration)

    for ep in range(episodes):
        state = env.reset()
        rewards = []
        waiting_time_list = []
        done = False
        prev_step_count = -100
        while not done:
            state_tensor = torch.FloatTensor(state).to(loaded_agent.device)
            with torch.no_grad():
                action = int(torch.argmax(loaded_agent.q_network(state_tensor)).item())
            state, reward, done, info = env.step(action)
            step = info['step_count']
            rewards.append(reward)
            veh_ids = traci.vehicle.getIDList()
            if veh_ids:
                avg_wait = np.mean([traci.vehicle.getWaitingTime(veh) for veh in veh_ids])
            else:
                avg_wait = 0.0
            waiting_time_list.append(avg_wait)

            if prev_step_count + print_interval <= step:
                print(f"Step {step}: State = {state}, Reward = {reward}")
                prev_step_count = step
        print(f"Test Episode {ep + 1} completed.")
        print(f"Total reward of episode {ep + 1}: {np.sum(rewards)}")
        print(f"Mean reward of episode {ep + 1}: {np.mean(rewards)}")
    env.close()

    viz = Visualization()
    # Plot the reward at each step.
    viz.save_data_and_plot(data=rewards, filename='ROT_testing', xlabel='Timesteps',
                           ylabel='Cumulative negative reward')

    return rewards, waiting_time_list

if __name__ == "__main__":
    # Define the SUMO command in GUI mode (adjust the config file to your two-intersection setup)
    sumo_cmd = ["sumo-gui", "-c", "two_intersection/two_intersection.sumocfg"]

    # Run one or more test episodes.
    test_agent(sumo_cmd, episodes=1, max_steps=2000)
    ttl.run_simulation(sumo_cmd, total_steps=2000, print_interval=50)
