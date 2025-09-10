from env_logistic_v0 import TruckTrajectoryMAEnv
from utils import ma_train, ma_evaluate

from stable_baselines3 import PPO, DQN, A2C

if __name__ == "__main__":
    num_agents = 2

    render = True

    agent_1_algo = PPO
    agent_2_algo = PPO
 
    ma_env = TruckTrajectoryMAEnv(render=render, num_agents=num_agents)

    model_agent_1 = agent_1_algo.load('policies/agent_1_model_v0')
    model_agent_2 = agent_1_algo.load('policies/agent_2_model_v0')

    models = {'truck_1': model_agent_1, 'truck_2': model_agent_2}

    total_episodes = 10

    print(f"Evaluating models for {total_episodes} episodes...")
    avg_agent, avg_model = ma_evaluate(ma_env, models, total_episodes=total_episodes)

    print(f"Average rewards per agent:\n {avg_agent}")
    print(f"Average rewards per model:\n {avg_model}")