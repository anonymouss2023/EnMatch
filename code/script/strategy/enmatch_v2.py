import os
import glob
import time
from datetime import datetime
from copy import deepcopy
import torch
import numpy as np

import gym
from enmatch.nets.enmatch.ppo import PPO

################################### Training ###################################
def train(config, env, env_copy):
    print("============================================================================================")

    ####### initialize environment hyperparameters ######

    env_name = 'SimpleMatch-v0'

    has_continuous_action_space = False  # continuous action space; else discrete

    max_ep_len = config['num_matches']                     # max timesteps in one episode
    max_training_timesteps = int(3e6)   # break training loop if timeteps > max_training_timesteps

    print_freq = max_ep_len * 10        # print avg reward in the interval (in num timesteps)
    log_freq = max_ep_len * 2           # log avg reward in the interval (in num timesteps)
    save_model_freq = int(1e5)          # save model frequency (in num timesteps)

    action_std = 0.6                    # starting std for action distribution (Multivariate Normal)
    action_std_decay_rate = 0.05        # linearly decay action_std (action_std = action_std - action_std_decay_rate)
    min_action_std = 0.1                # minimum action_std (stop decay after action_std <= min_action_std)
    action_std_decay_freq = int(2.5e5)  # action_std decay frequency (in num timesteps)
    #####################################################

    ## Note : print/log frequencies should be > than max_ep_len

    ################ PPO hyperparameters ################
    update_timestep = max_ep_len * 4      # update policy every n timesteps
    K_epochs = 80               # update policy for K epochs in one PPO update

    eps_clip = 0.2          # clip parameter for PPO
    gamma = 0.99            # discount factor

    lr_actor = 0.0003       # learning rate for actor network
    lr_critic = 0.001       # learning rate for critic network

    random_seed = 0         # set random seed if required (0 = no random seed)
    #####################################################

    print("training environment name : " + env_name)

    # env = gym.make(env_name)

    # state space dimension
    # state_dim = env.observation_space.shape[0]

    # action space dimension
    if has_continuous_action_space:
        action_dim = env.action_space.shape[0]
    else:
        action_dim = env.action_space.n

    ###################### logging ######################

    #### log files for multiple runs are NOT overwritten
    log_dir = "PPO_logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_dir = log_dir + '/' + env_name + '/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    #### get number of log files in log directory
    run_num = 0
    current_num_files = next(os.walk(log_dir))[2]
    run_num = len(current_num_files)

    #### create new log file for each run
    log_f_name = log_dir + '/PPO_' + env_name + "_log_" + str(run_num) + ".csv"

    print("current logging run number for " + env_name + " : ", run_num)
    print("logging at : " + log_f_name)
    #####################################################

    ################### checkpointing ###################
    run_num_pretrained = 0      #### change this to prevent overwriting weights in same env_name folder

    directory = "PPO_preTrained"
    if not os.path.exists(directory):
        os.makedirs(directory)

    directory = directory + '/' + env_name + '/'
    if not os.path.exists(directory):
        os.makedirs(directory)


    checkpoint_path = directory + "PPO_{}_{}_{}.pth".format(env_name, random_seed, run_num_pretrained)
    print("save checkpoint path : " + checkpoint_path)
    #####################################################


    ############# print all hyperparameters #############
    print("--------------------------------------------------------------------------------------------")
    print("max training timesteps : ", max_training_timesteps)
    print("max timesteps per episode : ", max_ep_len)
    print("model saving frequency : " + str(save_model_freq) + " timesteps")
    print("log frequency : " + str(log_freq) + " timesteps")
    print("printing average reward over episodes in last : " + str(print_freq) + " timesteps")
    print("--------------------------------------------------------------------------------------------")
    print("state space dimension : ", config['obs_size'])
    print("action space dimension : ", action_dim)
    print("--------------------------------------------------------------------------------------------")
    if has_continuous_action_space:
        print("Initializing a continuous action space policy")
        print("--------------------------------------------------------------------------------------------")
        print("starting std of action distribution : ", action_std)
        print("decay rate of std of action distribution : ", action_std_decay_rate)
        print("minimum std of action distribution : ", min_action_std)
        print("decay frequency of std of action distribution : " + str(action_std_decay_freq) + " timesteps")
    else:
        print("Initializing a discrete action space policy")
    print("--------------------------------------------------------------------------------------------")
    print("PPO update frequency : " + str(update_timestep) + " timesteps")
    print("PPO K epochs : ", K_epochs)
    print("PPO epsilon clip : ", eps_clip)
    print("discount factor (gamma) : ", gamma)
    print("--------------------------------------------------------------------------------------------")
    print("optimizer learning rate actor : ", lr_actor)
    print("optimizer learning rate critic : ", lr_critic)
    if random_seed:
        print("--------------------------------------------------------------------------------------------")
        print("setting random seed to ", random_seed)
        torch.manual_seed(random_seed)
        env.seed(random_seed)
        np.random.seed(random_seed)
    #####################################################

    print("============================================================================================")

    ################# training procedure ################

    # initialize a PPO agent
    ppo_agent = PPO(config['obs_size'], action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std, config)

    # pre_train via supervised learning from historical solved CO records
    # ppo_agent.pretrain()

    # track total training time
    start_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)

    print("============================================================================================")

    # logging file
    log_f = open(log_f_name,"w+")
    log_f.write('episode,timestep,reward\n')

    # printing and logging variables
    print_running_reward = 0
    print_running_episodes = 0

    log_running_reward = 0
    log_running_episodes = 0

    time_step = 0
    i_episode = 0

    # training loop
    while time_step <= max_training_timesteps:

        state = env.reset()
        state_copy = env_copy.reset()
        assert str(state) == str(state_copy)
        current_ep_reward = 0

        for t in range(1, max_ep_len+1):

            # select action with policy
            actions, heuristic_actions = ppo_agent.select_action(state)
            assert len(actions) == 2*config['num_per_team']
            for action in actions:
                state, reward, done, info = env.step(action)
            for action in heuristic_actions:
                heuristic_state, heuristic_reward, heuristic_done, heuristic_info = env_copy.step(action)
            # saving reward and is_terminals
            reward = (np.array(reward)+np.array(heuristic_reward))/2
            ppo_agent.buffer.rewards.append(reward)
            ppo_agent.buffer.is_terminals.append(done)

            time_step +=1
            current_ep_reward += np.mean(reward)

            # update PPO agent
            if time_step % update_timestep == 0:
                ppo_agent.update(max_ep_len, config['batch_size'])

            # if continuous action space; then decay action std of ouput action distribution
            if has_continuous_action_space and time_step % action_std_decay_freq == 0:
                ppo_agent.decay_action_std(action_std_decay_rate, min_action_std)

            # log in logging file
            if time_step % log_freq == 0:

                # log average reward till last episode
                log_avg_reward = log_running_reward / log_running_episodes
                log_avg_reward = round(log_avg_reward, 4)

                log_f.write('{},{},{}\n'.format(i_episode, time_step, log_avg_reward))
                log_f.flush()

                log_running_reward = 0
                log_running_episodes = 0

            # printing average reward
            if time_step % print_freq == 0:

                # print average reward till last episode
                print_avg_reward = print_running_reward / print_running_episodes
                print_avg_reward = round(print_avg_reward, 2)

                print("Episode : {} \t\t Timestep : {} \t\t Average Reward : {}".format(i_episode, time_step, print_avg_reward))

                print_running_reward = 0
                print_running_episodes = 0

            # save model weights
            if time_step % save_model_freq == 0:
                print("--------------------------------------------------------------------------------------------")
                print("saving model at : " + checkpoint_path)
                ppo_agent.save(checkpoint_path)
                print("model saved")
                print("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)
                print("--------------------------------------------------------------------------------------------")

            # break; if the episode is over
            # if done:
            #     break

        print_running_reward += current_ep_reward
        print_running_episodes += 1

        log_running_reward += current_ep_reward
        log_running_episodes += 1

        i_episode += 1

    log_f.close()
    env.close()

    # print total training time
    print("============================================================================================")
    end_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    print("Finished training at (GMT) : ", end_time)
    print("Total training time  : ", end_time - start_time)
    print("============================================================================================")


if __name__ == '__main__':
    import sys
    from script.strategy.utils import default_config
    from enmatch.env import *
    extra_config = eval(sys.argv[1]) if len(sys.argv) >= 2 else {}
    num_per_team = default_config['num_per_team']
    num_matches = default_config['num_matches']
    config = {
        **default_config,
        'epoch':100000,
        'batch_size':32,
        'num_players': 2*num_per_team*num_matches,
        'num_features': 1,
        'gpu':0,
        **extra_config
    }
    config['clipping'] = 10
    config['num_seeds'] = 1  # number of experimental seeds
    config['device'] = 'cuda'
    config['threshold'] = 0.0
    # config['train_size'] = int(4e4)
    # config['valid_size'] = int(4e3)
    config['embed_dim'] = 128
    config['hidden_dim'] = 128
    config['att_mode'] = "Dot"
    config['n_glimpses'] = 1
    # config['num_epochs'] = 100
    # config['batch_size'] = 128
    config['lr'] = 1e-4

    # config['num_matches'] = 1
    # config['num_per_team'] = 3
    config['reward_type'] = 'linear'
    # config['train_file'] = '/project/encom/enmatch/dataset/simplematch_6_3_100_100000_linear_opt.csv.2'
    # config['valid_file'] = '/project/encom/enmatch/dataset/simplematch_6_3_100_100000_linear_opt.csv.2'

    config['num_nodes'] = config['num_per_team'] * 2
    config['num_features'] = 1
    config['seq_len'] = config['num_nodes']
    config['input_dim'] = config['num_features']
    config['vocab_size'] = 10000 if config['num_per_team']<5 else 1000000
    config['action_size'] = config['max_steps'] = config['num_players'] = 2 * config['num_per_team'] * config['num_matches']
    print(config)
    config['obs_size'] = config['num_players'] * 3
    recsim = SeqSimpleMatchRecEnv(config = config, state_cls=SeqSimpleMatchState)
    env = RecEnvBase(recsim)
    env.seed(1)
    recsim_copy = SeqSimpleMatchRecEnv(config = config, state_cls=SeqSimpleMatchState)
    env_copy = RecEnvBase(recsim_copy)
    env_copy.seed(1)
    # env.reset()
    torch.autograd.set_detect_anomaly(True)
    train(config, env, env_copy)
