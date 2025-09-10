from env_logistic_v1 import TruckTrajectoryMAEnv
from utils import ma_train, ma_evaluate

from stable_baselines3 import PPO, DQN, A2C

if __name__ == "__main__":
    num_agents = 1

    # Entrenamiento.
    agent_1_algo = A2C
    agent_1_algo_params = {'policy': "MlpPolicy", 'verbose': 1, 'tensorboard_log': "./logs"}

    ma_env = TruckTrajectoryMAEnv(render=False, num_agents=num_agents)

    model_algo_map = {'truck_1': (agent_1_algo, agent_1_algo_params)}

    trained_models = ma_train(ma_env, model_algo_map=model_algo_map, models_to_train='all',
                              total_timesteps_per_model=200_000, training_iterations=1,
                              tb_log_suffix=f"trajectory_optim")
    
    ma_env.close()

    trained_models['truck_1'].save('policies/agent_1_model_single_v1')

    # Test.
    render = True
 
    ma_env = TruckTrajectoryMAEnv(render=render, num_agents=num_agents)

    model_agent_1 = agent_1_algo.load('policies/agent_1_model_single_v1')

    models = {'truck_1': model_agent_1}

    total_episodes = 4

    print(f"Evaluating models for {total_episodes} episodes...")
    avg_agent, avg_model = ma_evaluate(ma_env, models, total_episodes=total_episodes)

    print(f"Average rewards per agent:\n {avg_agent}")
    print(f"Average rewards per model:\n {avg_model}")