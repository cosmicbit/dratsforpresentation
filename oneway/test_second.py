import numpy as np
import torch
from oneway.agent_second import DQNAgent

import traci
from gym import spaces
from oneway.generator import generate_routefile

actions = [{
        "phase":"GGGggrrrrrGGGggrrrrr",
        "duration":25
    },{
        "phase":"GGGggrrrrrGGGggrrrrr",
        "duration": 30
    },{
        "phase":"GGGggrrrrrGGGggrrrrr",
        "duration": 35
    },{
        "phase":"rrrrrGGGggrrrrrGGGgg",
        "duration": 25
    },{
        "phase":"rrrrrGGGggrrrrrGGGgg",
        "duration": 30
    },{
        "phase":"rrrrrGGGggrrrrrGGGgg",
        "duration": 35
    }
]
##############################
# SUMO-Based Gym Environment #
##############################
class SUMOTrafficEnv():

    def __init__(self, sumo_cmd, max_steps):
        self.sumo_cmd = sumo_cmd
        self._max_steps = max_steps
        self._num_states = 17
        self._num_actions = 6
        self.observation_space = spaces.Box(low=0, high=100, shape=(self._num_states,), dtype=np.float32)
        self.action_space = spaces.Discrete(6)
        self.phase_dict = {
            0: "GGGggrrrrrGGGggrrrrr",
            1: "rrrrrGGGggrrrrrGGGgg"
        }
        # inits
        self._step = 0
        self._waiting_times = {}
        self._sum_neg_reward = 0
        self._sum_queue_length = 0
        self._sum_waiting_time = 0
        self.old_total_wait = 0
        self.current_total_wait = 0
        self.old_state = -1
        self.old_action = -1
        self.vehicles = dict()

    def reset(self):
        # If TraCI is already connected, close it.
        if traci.isLoaded():
            traci.close()
        # generate_route_file(self.net_file, filename="one_intersection/routes.rou.xml")
        traci.start(self.sumo_cmd)

        # inits
        self._step = 0
        self._waiting_times = {}
        self._sum_neg_reward = 0
        self._sum_queue_length = 0
        self._sum_waiting_time = 0
        self.old_total_wait = 0
        self.old_state = -1
        self.old_action = -1
        self.vehicles = dict()
        traci.trafficlight.setRedYellowGreenState("TL1", actions[0]["phase"])
        return self._get_state()

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
        # print("state_vector:", state_vector)
        return np.array(state_vector, dtype=np.float32)

    def step(self, action):
        # print("step:", self.step_count)
        self._apply_action(action)
        self.old_action = action
        self.current_state = self._get_state()
        self.old_state = self.current_state
        reward = self._get_reward()
        done = self._step >= self._max_steps
        return self.current_state, reward, done, {}

    def _apply_action(self, action):
        phase = actions[action]["phase"]
        duration = actions[action]["duration"]

        traci.trafficlight.setRedYellowGreenState("TL1", phase)

        self._simulate(duration)

    def _simulate(self, steps_todo):
        """
        Execute steps in sumo while gathering statistics
        """
        if (self._step + steps_todo) >= self._max_steps:  # do not do more steps than the maximum allowed number of steps
            steps_todo = self._max_steps - self._step

        while steps_todo > 0:
            traci.simulationStep()  # simulate 1 step in sumo
            self._step += 1  # update the step counter
            steps_todo -= 1

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


def test_agent(choices=None, sumo_cmd = None,
               episodes=1):
    """
    Runs the trained agent for a given number of episodes and measures:
      - Rewards (as before)
      - Throughput (cumulative count of vehicles that have arrived per environment step)
      - Average waiting time (averaged over the simulation steps within an environment step)
    """
    env = SUMOTrafficEnv(sumo_cmd, max_steps=1000)
    if choices is not None:
        generate_routefile(choices)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    loaded_agent = DQNAgent(state_dim, action_dim)
    loaded_agent.q_network.load_state_dict(torch.load("oneway/models/second_model.pth",
                                                      map_location=loaded_agent.device))
    loaded_agent.update_target_network()
    loaded_agent.epsilon = 0.0

    for ep in range(episodes):
        state = env.reset()
        rewards = []
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
        print(f"Mean Waiting Time: {np.mean(waiting_time_list):.2f}")
    env.close()


    return rewards, waiting_time_list

if __name__ == "__main__":
    sumo_cmd = ["sumo-gui", "-c", "one_intersection/simple_intersection.sumocfg"]
    rewards, waiting_time_list = test_agent(sumo_cmd, episodes=1)
    # ttl.run_simulation(sumo_cmd, total_steps=1000, print_interval=50)

