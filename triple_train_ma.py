from env_logistic_v1 import TruckTrajectoryMAEnv
from utils import ma_train, ma_evaluate

from stable_baselines3 import PPO, DQN, A2C

if __name__ == "__main__":
    num_agents = 3

    # Entrenamiento.
    agent_1_algo = A2C
    agent_1_algo_params = {'policy': "MlpPolicy", 'verbose': 1, 'tensorboard_log': "./logs"}

    agent_2_algo = A2C
    agent_2_algo_params = {'policy': "MlpPolicy", 'verbose': 1, 'tensorboard_log': "./logs"}

    agent_3_algo = A2C
    agent_3_algo_params = {'policy': "MlpPolicy", 'verbose': 1, 'tensorboard_log': "./logs"}

    ma_env = TruckTrajectoryMAEnv(render=False, num_agents=num_agents)

    model_algo_map = {'truck_1': (agent_1_algo, agent_1_algo_params), 'truck_2': (agent_2_algo, agent_2_algo_params),
                      'truck_3': (agent_3_algo, agent_3_algo_params)}

    trained_models = ma_train(ma_env, model_algo_map=model_algo_map, models_to_train='all',
                              total_timesteps_per_model=1_000_000, training_iterations=20,
                              tb_log_suffix=f"trajectory_optim")
    
    ma_env.close()

    trained_models['truck_1'].save('policies/test_triple_1')
    trained_models['truck_2'].save('policies/test_triple_2')
    trained_models['truck_3'].save('policies/test_triple_3')

    # Test.
    render = True
 
    ma_env = TruckTrajectoryMAEnv(render=render, num_agents=num_agents)

    model_agent_1 = agent_1_algo.load('policies/test_triple_1')
    model_agent_2 = agent_2_algo.load('policies/test_triple_2')
    model_agent_3 = agent_3_algo.load('policies/test_triple_3')

    models = {'truck_1': model_agent_1, 'truck_2': model_agent_2, 'truck_3': model_agent_3}

    total_episodes = 4

    print(f"Evaluating models for {total_episodes} episodes...")
    avg_agent, avg_model = ma_evaluate(ma_env, models, total_episodes=total_episodes)

    print(f"Average rewards per agent:\n {avg_agent}")
    print(f"Average rewards per model:\n {avg_model}")