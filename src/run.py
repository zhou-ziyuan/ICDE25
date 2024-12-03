import datetime
import os
import pprint
import time
import threading
import torch as th
from types import SimpleNamespace as SN
from utils.logging import Logger
from utils.timehelper import time_left, time_str
from os.path import dirname, abspath

from learners import REGISTRY as le_REGISTRY
from runners import REGISTRY as r_REGISTRY
from controllers import REGISTRY as mac_REGISTRY
from components.episode_buffer import ReplayBuffer
from components.episode_buffer_fop import ReplayBuffer_FOP
from components.transforms import OneHot
from mappo.mappo_learner import MAPPO
# from fop.fop_modules.agents.rnn_agent import RNNAgent
from fop.fop_learners.fop_learner import FOP_Learner
from fop.fop_learners.fop_diffusion_learner import FOP_Diffusion_Learner

from fop.fop_controllers.cqmix_controller import CQMixMAC
from fop.fop_controllers.diffusion_controller import DiffusionMAC
from components.episode_buffer import DiffusionMemory


def run(_run, _config, _log):
    # check args sanity
    _config = args_sanity_check(_config, _log)

    args = SN(**_config)
    args.device = "cuda" if args.use_cuda else "cpu"

    # setup loggers
    logger = Logger(_log)

    _log.info("Experiment Parameters:")
    experiment_params = pprint.pformat(_config,
                                       indent=4,
                                       width=1)
    _log.info("\n\n" + experiment_params + "\n")

    # configure tensorboard logger
    unique_token = "{}__{}".format(args.name, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    args.unique_token = unique_token
    if args.use_tensorboard:
        tb_logs_direc = os.path.join(dirname(dirname(abspath(__file__))), "results", "tb_logs")
        tb_exp_direc = os.path.join(tb_logs_direc, "{}").format(unique_token)
        logger.setup_tb(tb_exp_direc)

    # sacred is on by default
    logger.setup_sacred(_run)

    # Run and train
    run_sequential(args=args, logger=logger)

    # Clean up after finishing
    print("Exiting Main")

    print("Stopping all threads")
    for t in threading.enumerate():
        if t.name != "MainThread":
            print("Thread {} is alive! Is daemon: {}".format(t.name, t.daemon))
            t.join(timeout=1)
            print("Thread joined")

    print("Exiting script")

    # Making sure framework really exits
    # os._exit(os.EX_OK)
    os._exit(0)  ########


def evaluate_sequential(args, runner, learner, logger,adv_learner=None):
    # print(args.test_nepisode)

    # for _ in range(args.test_nepisode):
    #     adv_test = False
    #     if args.Number_attack > 0:
    #         adv_test = True

    #     runner.run(test_mode=True, learner=learner, adv_test=adv_test,adv_learner=adv_learner)
    # # logger.log_stat("episode", args.test_nepisode-1, runner.t_env)
    # logger.print_recent_stats()

    # if args.save_replay:
    #     runner.save_replay()

    # runner.close_env()
    for _ in range(args.test_nepisode):
        adv_test = False
        if args.Number_attack > 0:
            adv_test = True
            
        runner.run(test_mode=True, learner=learner, adv_test=adv_test,adv_learner=adv_learner)
    # logger.log_stat("episode", args.test_nepisode-1, runner.t_env)
    logger.print_recent_stats()

    if args.save_replay:
        runner.save_replay()

    runner.close_env()


def run_sequential(args, logger):
    # Init runner so we can get env info
    runner = r_REGISTRY[args.runner](args=args, logger=logger)

    # Set up schemes and groups here
    env_info = runner.get_env_info()
    args.n_agents = env_info["n_agents"]
    args.n_actions = env_info["n_actions"]
    args.state_shape = env_info["state_shape"]
    args.obs_shape = env_info["obs_shape"]
    args.unit_dim = env_info["unit_dim"]

    # Default/Base scheme
    scheme = {
        "state": {"vshape": env_info["state_shape"]},
        "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "avail_actions": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.int},
        "reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": th.uint8},
        "perturbations":{"vshape": env_info["obs_shape"],"group": "agents"},# TODO 添加扰动字段
    }
    groups = {
        "agents": args.n_agents
    }
    preprocess = {
        "actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])
    }

    adv_learner=None
    learner=None
    if args.Number_attack > 0:
        if args.attack_method == "fgsm" or args.attack_method == "pgd" or args.attack_method == "rand_noise" or args.attack_method == "gaussian":
            buffer = ReplayBuffer(scheme, groups, args.buffer_size, env_info["episode_limit"] + 1,
                                  preprocess=preprocess,
                                  device="cpu" if args.buffer_cpu_only else args.device)
            # Setup multiagent controller here
            mac = mac_REGISTRY[args.mac](buffer.scheme, groups, args)
            # Give runner the scheme
            runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)
            # Learner adv_learner and agent_learner
            learner = le_REGISTRY[args.learner](mac, buffer.scheme, logger, args)
        elif args.attack_method == "adv_reg":
            buffer = ReplayBuffer(scheme, groups, args.buffer_size, env_info["episode_limit"] + 1,
                                  preprocess=preprocess,
                                  device="cpu" if args.buffer_cpu_only else args.device)
            reg_buffer = ReplayBuffer(scheme, groups, args.buffer_size, env_info["episode_limit"] + 1,
                                      preprocess=preprocess,
                                      device="cpu" if args.buffer_cpu_only else args.device)
            # Setup multiagent controller here
            mac = mac_REGISTRY[args.mac](buffer.scheme, groups, args)
            # Give runner the scheme
            runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)
            # Learner adv_learner and agent_learner
            learner = le_REGISTRY[args.learner](mac, buffer.scheme, logger, args)
        elif args.attack_method == "atla":
            buffer = ReplayBuffer(scheme, groups, args.buffer_size, env_info["episode_limit"] + 1,
                                  preprocess=preprocess,
                                  device="cpu" if args.buffer_cpu_only else args.device)
            adv_buffer = ReplayBuffer(scheme, groups, args.buffer_size, env_info["episode_limit"] + 1,
                                      preprocess=preprocess,
                                      device="cpu" if args.buffer_cpu_only else args.device)

            mac = mac_REGISTRY[args.mac](buffer.scheme, groups, args)
            runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)
            learner = le_REGISTRY[args.learner](mac, buffer.scheme, logger, args) #训练多智能体的learner
            args.obs_shape= env_info["obs_shape"]
            adv_learner=MAPPO(args) # 添加一个用于训练扰动的learner
        
        elif args.attack_method == "harl_adv_tar":
            buffer = ReplayBuffer(scheme, groups, args.buffer_size, env_info["episode_limit"] + 1,
                                  preprocess=preprocess,
                                  device="cpu" if args.buffer_cpu_only else args.device)
            # adv_buffer = OffPolicyBufferFP(
            #         {**algo_args["train"], **algo_args["model"], **algo_args["algo"]},
            #         self.envs.share_observation_space[0],
            #         self.num_agents,
            #         self.envs.observation_space,
            #         self.envs.action_space,
            #     )
            # #ReplayBuffer(scheme, groups, args.buffer_size, env_info["episode_limit"] + 1,
            #                           preprocess=preprocess,
            #                           device="cpu" if args.buffer_cpu_only else args.device)

            mac = mac_REGISTRY[args.mac](buffer.scheme, groups, args)
            runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)
            learner = le_REGISTRY[args.learner](mac, buffer.scheme, logger, args) #训练多智能体的learner
            args.obs_shape= env_info["obs_shape"]
            adv_learner=MAPPO(args) # 添加一个用于训练扰动的learner

        elif args.attack_method == "mer":
            # print(env_info)
            # args.action_spaces_mer = env_info["action_spaces"] 
            buffer = ReplayBuffer(scheme, groups, args.buffer_size, env_info["episode_limit"] + 1,
                                  preprocess=preprocess,
                                  device="cpu" if args.buffer_cpu_only else args.device)
            adv_buffer = ReplayBuffer_FOP(scheme, groups, args.buffer_size, env_info["episode_limit"] + 1,
                                      preprocess=preprocess,
                                      device="cpu" if args.buffer_cpu_only else args.device)

            mac = mac_REGISTRY[args.mac](buffer.scheme, groups, args)
            adv_mac = CQMixMAC(adv_buffer.scheme, groups, args)
            runner.setup_adv(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac, adv_mac=adv_mac)
            learner = le_REGISTRY[args.learner](mac, buffer.scheme, logger, args) #训练多智能体的learner
            args.obs_shape= env_info["obs_shape"]
            adv_learner=FOP_Learner(adv_mac, adv_buffer.scheme, logger, args) # 添加一个用于训练扰动的learner
            # xxx

        elif args.attack_method == "mer_diffusion":
            # print(env_info)
            # args.action_spaces_mer = env_info["action_spaces"] 
            buffer = ReplayBuffer(scheme, groups, args.buffer_size, env_info["episode_limit"] + 1,
                                  preprocess=preprocess,
                                  device="cpu" if args.buffer_cpu_only else args.device)
            adv_buffer = ReplayBuffer_FOP(scheme, groups, args.buffer_size, env_info["episode_limit"] + 1,
                                      preprocess=preprocess,
                                      device="cpu" if args.buffer_cpu_only else args.device)
            diffusion_buffer = DiffusionMemory(args.obs_shape*args.n_agents, args.obs_shape*args.n_agents, args.buffer_size, args.device)
            mac = mac_REGISTRY[args.mac](buffer.scheme, groups, args)
            # scheme, groups, diffusion_buffer, args
            adv_mac = DiffusionMAC(adv_buffer.scheme, groups, args)
            runner.setup_adv(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac, adv_mac=adv_mac)
            learner = le_REGISTRY[args.learner](mac, buffer.scheme, logger, args) #训练多智能体的learner
            args.obs_shape= env_info["obs_shape"]
            adv_learner=FOP_Diffusion_Learner(adv_mac, adv_buffer.scheme, logger, args, diffusion_buffer) # 添加一个用于训练扰动的learner
            # xxx
        elif args.attack_method == "adv_tar":
            buffer = ReplayBuffer(scheme, groups, args.buffer_size, env_info["episode_limit"] + 1,
                                  preprocess=preprocess,
                                  device="cpu" if args.buffer_cpu_only else args.device)
            adv_buffer = ReplayBuffer(scheme, groups, args.buffer_size, env_info["episode_limit"] + 1,
                                      preprocess=preprocess,
                                      device="cpu" if args.buffer_cpu_only else args.device)

            mac = mac_REGISTRY[args.mac](buffer.scheme, groups, args)
            adv_mac = mac_REGISTRY[args.mac](adv_buffer.scheme, groups, args)

            runner.setup_adv(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac, adv_mac=adv_mac)
            learner = le_REGISTRY[args.learner](mac, buffer.scheme, logger, args)
            adv_learner = le_REGISTRY[args.learner](adv_mac, adv_buffer.scheme, logger, args)

        elif args.attack_method == "fop_adv_tar":
            buffer = ReplayBuffer(scheme, groups, args.buffer_size, env_info["episode_limit"] + 1,
                                  preprocess=preprocess,
                                  device="cpu" if args.buffer_cpu_only else args.device)
            adv_buffer = ReplayBuffer_FOP(scheme, groups, args.buffer_size, env_info["episode_limit"] + 1,
                                      preprocess=preprocess,
                                      device="cpu" if args.buffer_cpu_only else args.device)
            

            mac = mac_REGISTRY[args.mac](buffer.scheme, groups, args)
            adv_mac = mac_REGISTRY[args.adv_mac](adv_buffer.scheme, groups, args)

            runner.setup_adv(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac, adv_mac=adv_mac)
            learner = le_REGISTRY[args.learner](mac, buffer.scheme, logger, args)
            adv_learner = le_REGISTRY[args.learner_mer](adv_mac, mac, adv_buffer.scheme, logger, args)

    else:
        buffer = ReplayBuffer(scheme, groups, args.buffer_size, env_info["episode_limit"] + 1,
                              preprocess=preprocess,
                              device="cpu" if args.buffer_cpu_only else args.device)

        # Setup multiagent controller here
        mac = mac_REGISTRY[args.mac](buffer.scheme, groups, args)

        # Give runner the scheme
        runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)

        # Learner adv_learner and agent_learner
        learner = le_REGISTRY[args.learner](mac, buffer.scheme, logger, args)

    if args.use_cuda:
        learner.cuda()
        if args.Number_attack > 0 and (args.attack_method == "adv_tar" or args.attack_method == "fop_adv_tar"):
            adv_learner.cuda()

    if args.checkpoint_path != "":

        timesteps = []
        timestep_to_load = 0

        if not os.path.isdir(args.checkpoint_path):
            logger.console_logger.info("Checkpoint directiory {} doesn't exist".format(args.checkpoint_path))
            return

        # Go through all files in args.checkpoint_path
        for name in os.listdir(args.checkpoint_path):
            full_name = os.path.join(args.checkpoint_path, name)
            # Check if they are dirs the names of which are numbers
            if os.path.isdir(full_name) and name.isdigit():
                timesteps.append(int(name))

        if args.load_step == 0:
            # choose the max timestep
            timestep_to_load = max(timesteps)
        else:
            # choose the timestep closest to load_step
            timestep_to_load = min(timesteps, key=lambda x: abs(x - args.load_step))
        # timestep_to_load=3800278
        model_path = os.path.join(args.checkpoint_path, str(timestep_to_load))

        logger.console_logger.info("Loading model from {}".format(model_path))
        learner.load_models(model_path)
        runner.t_env = timestep_to_load

    if args.adv_checkpoint_path != "" and args.Number_attack > 0 and (args.attack_method == "adv_tar" or args.attack_method == "fop_adv_tar" or args.attack_method == "atla" or args.attack_method == "mer" or args.attack_method == "mer_diffusion"):

        adv_timesteps = []
        adv_timestep_to_load = 0

        if not os.path.isdir(args.adv_checkpoint_path):
            logger.console_logger.info("Checkpoint directiory {} doesn't exist".format(args.adv_checkpoint_path))
            return

        # Go through all files in args.checkpoint_path
        for name in os.listdir(args.adv_checkpoint_path):
            adv_full_name = os.path.join(args.adv_checkpoint_path, name)
            # Check if they are dirs the names of which are numbers
            if os.path.isdir(adv_full_name) and name.isdigit():
                adv_timesteps.append(int(name))

        if args.adv_load_step == 0:
            # choose the max timestep
            adv_timestep_to_load = max(adv_timesteps)
        else:
            # choose the timestep closest to load_step
            adv_timestep_to_load = min(adv_timesteps, key=lambda x: abs(x - args.adv_load_step))

        adv_model_path = os.path.join(args.adv_checkpoint_path, str(adv_timestep_to_load))

        logger.console_logger.info("Loading model from {}".format(adv_model_path))
        adv_learner.load_models(adv_model_path)

    if args.evaluate or args.save_replay:
        evaluate_sequential(args, runner, learner, logger,adv_learner)
        return

    # start training
    episode = 0
    last_test_T = -args.test_interval - 1
    last_log_T = 0
    model_save_time = 0

    start_time = time.time()
    last_time = start_time

    logger.console_logger.info("Beginning training for {} timesteps".format(args.t_max))
    adv_test = False
    if args.Number_attack > 0:
        adv_test = True
    while runner.t_env <= args.t_max:
        if args.Number_attack > 0:
            if args.attack_method == "fgsm" or args.attack_method == "pgd" or args.attack_method == "rand_noise" or args.attack_method == "gaussian":
                # Run for a whole episode at a time
                episode_batch = runner.run(test_mode=False, learner=learner, adv_test=adv_test)

                buffer.insert_episode_batch(episode_batch)

                if buffer.can_sample(args.batch_size):
                    episode_sample = buffer.sample(args.batch_size)

                    # Truncate batch to only filled timesteps
                    max_ep_t = episode_sample.max_t_filled()
                    episode_sample = episode_sample[:, :max_ep_t]

                    if episode_sample.device != args.device:
                        # print("******************************")
                        episode_sample.to(args.device)

                    learner.train(episode_sample, runner.t_env, episode)

                # Execute test runs once in a while
                n_test_runs = max(1, args.test_nepisode // runner.batch_size)
                if (runner.t_env - last_test_T) / args.test_interval >= 1.0:

                    logger.console_logger.info("t_env: {} / {}".format(runner.t_env, args.t_max))
                    logger.console_logger.info("Estimated time left: {}. Time passed: {}".format(
                        time_left(last_time, last_test_T, runner.t_env, args.t_max),
                        time_str(time.time() - start_time)))
                    last_time = time.time()

                    last_test_T = runner.t_env
                    for _ in range(n_test_runs):
                        runner.run(test_mode=True, learner=learner, adv_test=False)

                if args.save_model and (
                        runner.t_env - model_save_time >= args.save_model_interval or model_save_time == 0):
                    model_save_time = runner.t_env
                    save_path = os.path.join(args.local_results_path, "models", args.unique_token, str(runner.t_env))
                    # "results/models/{}".format(unique_token)
                    os.makedirs(save_path, exist_ok=True)
                    logger.console_logger.info("Saving models to {}".format(save_path))

                    # learner should handle saving/loading -- delegate actor save/load to mac,
                    # use appropriate filenames to do critics, optimizer states
                    learner.save_models(save_path)

                episode += args.batch_size_run

                if (runner.t_env - last_log_T) >= args.log_interval:
                    logger.log_stat("episode", episode, runner.t_env)
                    logger.print_recent_stats()
                    last_log_T = runner.t_env
            elif args.attack_method == "adv_reg":
                # Run for a whole episode at a time
                episode_batch, reg_episode_batch = runner.run(test_mode=False, learner=learner, adv_test=adv_test)

                buffer.insert_episode_batch(episode_batch)
                reg_buffer.insert_episode_batch(reg_episode_batch)

                if buffer.can_sample(args.batch_size) and reg_buffer.can_sample(args.batch_size):
                    episode_sample,episode_index = buffer.sample_ori(args.batch_size)
                    reg_episode_sample = reg_buffer.sample_reg(args.batch_size,episode_index)
                    # Truncate batch to only filled timesteps
                    max_ep_t = episode_sample.max_t_filled()
                    episode_sample = episode_sample[:, :max_ep_t]
                    reg_max_ep_t = reg_episode_sample.max_t_filled()
                    reg_episode_sample = reg_episode_sample[:, :reg_max_ep_t]

                    if episode_sample.device != args.device:
                        # print("******************************")
                        episode_sample.to(args.device)
                    if reg_episode_sample.device != args.device:
                        # print("******************************")
                        reg_episode_sample.to(args.device)
                    learner.train_reg(episode_sample,reg_episode_sample,runner.t_env, episode)
                    # Execute test runs once in a while
                    n_test_runs = max(1, args.test_nepisode // runner.batch_size)
                    if (runner.t_env - last_test_T) / args.test_interval >= 1.0:

                        logger.console_logger.info("t_env: {} / {}".format(runner.t_env, args.t_max))
                        logger.console_logger.info("Estimated time left: {}. Time passed: {}".format(
                            time_left(last_time, last_test_T, runner.t_env, args.t_max),
                            time_str(time.time() - start_time)))
                        last_time = time.time()

                        last_test_T = runner.t_env
                        for _ in range(n_test_runs):
                            runner.run(test_mode=True, learner=learner, adv_test=False)

                    if args.save_model and (
                            runner.t_env - model_save_time >= args.save_model_interval or model_save_time == 0):
                        model_save_time = runner.t_env
                        save_path = os.path.join(args.local_results_path, "models", args.unique_token,
                                                 str(runner.t_env))
                        adv_save_path = os.path.join(args.adv_local_results_path, "models", args.unique_token,
                                                     str(runner.t_env))
                        # "results/models/{}".format(unique_token)
                        os.makedirs(save_path, exist_ok=True)
                        os.makedirs(adv_save_path, exist_ok=True)
                        logger.console_logger.info("Saving models to {}".format(save_path))
                        logger.console_logger.info("Saving models to {}".format(adv_save_path))

                        # learner should handle saving/loading -- delegate actor save/load to mac,
                        # use appropriate filenames to do critics, optimizer states
                        learner.save_models(save_path)
                        # adv_learner.save_models(adv_save_path)

                    episode += args.batch_size_run

                    if (runner.t_env - last_log_T) >= args.log_interval:
                        logger.log_stat("episode", episode, runner.t_env)
                        logger.print_recent_stats()
                        last_log_T = runner.t_env
            elif args.attack_method == "atla":
                #TODO 若使用mappo训练出扰动状态，需要在run中传入模型，才可以将模型生成的状态传进多智能体网络当中

                # Execute test runs once in a while
                # Run for a whole episode at a time
                # Run for a whole episode at a time
                episode_batch, adv_episode_batch = runner.run(test_mode=False, learner=learner, adv_test=adv_test,adv_learner=adv_learner)
                buffer.insert_episode_batch(episode_batch)
                adv_buffer.insert_episode_batch(adv_episode_batch)

                if buffer.can_sample(args.batch_size):
                    episode_sample = buffer.sample(args.batch_size)

                    # Truncate batch to only filled timesteps
                    max_ep_t = episode_sample.max_t_filled()
                    episode_sample = episode_sample[:, :max_ep_t]

                    if episode_sample.device != args.device:
                        episode_sample.to(args.device)

                    learner.train(episode_sample, runner.t_env, episode)

                if adv_buffer.can_sample(args.adv_batch_size):
                    adv_episode_sample = adv_buffer.sample(args.adv_batch_size)

                    # Truncate batch to only filled timesteps
                    adv_max_ep_t = adv_episode_sample.max_t_filled()
                    adv_episode_sample = adv_episode_sample[:, :adv_max_ep_t]

                    if adv_episode_sample.device != args.device:
                        adv_episode_sample.to(args.device)

                    adv_learner.train(adv_episode_sample, runner.t_env, episode)

                # Execute test runs once in a while
                n_test_runs = max(1, args.test_nepisode // runner.batch_size)
                if (runner.t_env - last_test_T) / args.test_interval >= 1.0:

                    logger.console_logger.info("t_env: {} / {}".format(runner.t_env, args.t_max))
                    logger.console_logger.info("Estimated time left: {}. Time passed: {}".format(
                        time_left(last_time, last_test_T, runner.t_env, args.t_max),
                        time_str(time.time() - start_time)))
                    last_time = time.time()

                    last_test_T = runner.t_env
                    for _ in range(n_test_runs):
                        runner.run(test_mode=True, learner=learner, adv_test=False)

                if args.save_model and (
                        runner.t_env - model_save_time >= args.save_model_interval or model_save_time == 0):
                    model_save_time = runner.t_env
                    save_path = os.path.join(args.local_results_path, "models", args.unique_token, str(runner.t_env))
                    adv_save_path = os.path.join(args.adv_local_results_path, "models", args.unique_token,
                                                 str(runner.t_env))
                    # "results/models/{}".format(unique_token)
                    os.makedirs(save_path, exist_ok=True)
                    os.makedirs(adv_save_path, exist_ok=True)
                    logger.console_logger.info("Saving models to {}".format(save_path))
                    logger.console_logger.info("Saving models to {}".format(adv_save_path))

                    # learner should handle saving/loading -- delegate actor save/load to mac,
                    # use appropriate filenames to do critics, optimizer states
                    learner.save_models(save_path)
                    adv_learner.save_models(adv_save_path)

                episode += args.batch_size_run

                if (runner.t_env - last_log_T) >= args.log_interval:
                    logger.log_stat("episode", episode, runner.t_env)
                    logger.print_recent_stats()
                    last_log_T = runner.t_env

            elif args.attack_method == "mer":
                #TODO 若使用mappo训练出扰动状态，需要在run中传入模型，才可以将模型生成的状态传进多智能体网络当中

                # Execute test runs once in a while
                # Run for a whole episode at a time
                # Run for a whole episode at a time
                episode_batch, adv_episode_batch = runner.run(test_mode=False, learner=learner, adv_test=adv_test,adv_learner=adv_learner)
                # episode_batch, adv_episode_batch = runner.run(test_mode=False, learner=learner, adv_test=adv_test)

                
                buffer.insert_episode_batch(episode_batch)
                # print(adv_episode_batch["perturbations"])
                # aaa
                adv_buffer.insert_episode_batch(adv_episode_batch)

                if buffer.can_sample(args.batch_size):
                    episode_sample = buffer.sample(args.batch_size)

                    # Truncate batch to only filled timesteps
                    max_ep_t = episode_sample.max_t_filled()
                    episode_sample = episode_sample[:, :max_ep_t]

                    if episode_sample.device != args.device:
                        episode_sample.to(args.device)

                    learner.train(episode_sample, runner.t_env, episode)

                if adv_buffer.can_sample(args.adv_batch_size):
                    adv_episode_sample = adv_buffer.sample(args.adv_batch_size)

                    # Truncate batch to only filled timesteps
                    adv_max_ep_t = adv_episode_sample.max_t_filled()
                    adv_episode_sample = adv_episode_sample[:, :adv_max_ep_t]

                    if adv_episode_sample.device != args.device:
                        adv_episode_sample.to(args.device)

                    # print(adv_episode_sample["actions"])
                    # aaaaa
                    adv_learner.train(adv_episode_sample, runner.t_env, episode)

                # Execute test runs once in a while
                n_test_runs = max(1, args.test_nepisode // runner.batch_size)
                if (runner.t_env - last_test_T) / args.test_interval >= 1.0:

                    logger.console_logger.info("t_env: {} / {}".format(runner.t_env, args.t_max))
                    logger.console_logger.info("Estimated time left: {}. Time passed: {}".format(
                        time_left(last_time, last_test_T, runner.t_env, args.t_max),
                        time_str(time.time() - start_time)))
                    last_time = time.time()

                    last_test_T = runner.t_env
                    for _ in range(n_test_runs):
                        runner.run(test_mode=True, learner=learner, adv_test=False)

                if args.save_model and (
                        runner.t_env - model_save_time >= args.save_model_interval or model_save_time == 0):
                    model_save_time = runner.t_env
                    save_path = os.path.join(args.local_results_path, "models", args.unique_token, str(runner.t_env))
                    adv_save_path = os.path.join(args.adv_local_results_path, "models", args.unique_token,
                                                 str(runner.t_env))
                    # "results/models/{}".format(unique_token)
                    os.makedirs(save_path, exist_ok=True)
                    os.makedirs(adv_save_path, exist_ok=True)
                    logger.console_logger.info("Saving models to {}".format(save_path))
                    logger.console_logger.info("Saving models to {}".format(adv_save_path))

                    # learner should handle saving/loading -- delegate actor save/load to mac,
                    # use appropriate filenames to do critics, optimizer states
                    learner.save_models(save_path)
                    adv_learner.save_models(adv_save_path)

                episode += args.batch_size_run

                if (runner.t_env - last_log_T) >= args.log_interval:
                    logger.log_stat("episode", episode, runner.t_env)
                    logger.print_recent_stats()
                    last_log_T = runner.t_env

            elif args.attack_method == "mer_diffusion":

                episode_batch, adv_episode_batch = runner.run(test_mode=False, learner=learner, adv_test=adv_test,adv_learner=adv_learner)
                # episode_batch, adv_episode_batch = runner.run(test_mode=False, learner=learner, adv_test=adv_test)

                
                buffer.insert_episode_batch(episode_batch)
                # print(adv_episode_batch["perturbations"])
                # aaa
                adv_buffer.insert_episode_batch(adv_episode_batch)

                # print(adv_episode_batch["obs"].view(-1, args.obs_shape))
                # print(adv_episode_batch["perturbations"].view(-1, args.obs_shape))
                


                if buffer.can_sample(args.batch_size):
                    episode_sample = buffer.sample(args.batch_size)

                    # Truncate batch to only filled timesteps
                    max_ep_t = episode_sample.max_t_filled()
                    episode_sample = episode_sample[:, :max_ep_t]

                    if episode_sample.device != args.device:
                        episode_sample.to(args.device)

                    learner.train(episode_sample, runner.t_env, episode)

                if adv_buffer.can_sample(args.adv_batch_size):
                    adv_episode_sample = adv_buffer.sample(args.adv_batch_size)

                    # Truncate batch to only filled timesteps
                    adv_max_ep_t = adv_episode_sample.max_t_filled()
                    adv_episode_sample = adv_episode_sample[:, :adv_max_ep_t]

                    if adv_episode_sample.device != args.device:
                        adv_episode_sample.to(args.device)

                    # print(adv_episode_sample["actions"])
                    # aaaaa
                    adv_learner.train(adv_episode_sample, runner.t_env, episode)

                # Execute test runs once in a while
                n_test_runs = max(1, args.test_nepisode // runner.batch_size)
                if (runner.t_env - last_test_T) / args.test_interval >= 1.0:

                    logger.console_logger.info("t_env: {} / {}".format(runner.t_env, args.t_max))
                    logger.console_logger.info("Estimated time left: {}. Time passed: {}".format(
                        time_left(last_time, last_test_T, runner.t_env, args.t_max),
                        time_str(time.time() - start_time)))
                    last_time = time.time()

                    last_test_T = runner.t_env
                    for _ in range(n_test_runs):
                        runner.run(test_mode=True, learner=learner, adv_test=False)

                if args.save_model and (
                        runner.t_env - model_save_time >= args.save_model_interval or model_save_time == 0):
                    model_save_time = runner.t_env
                    save_path = os.path.join(args.local_results_path, "models", args.unique_token, str(runner.t_env))
                    adv_save_path = os.path.join(args.adv_local_results_path, "models", args.unique_token,
                                                 str(runner.t_env))
                    # "results/models/{}".format(unique_token)
                    os.makedirs(save_path, exist_ok=True)
                    os.makedirs(adv_save_path, exist_ok=True)
                    logger.console_logger.info("Saving models to {}".format(save_path))
                    logger.console_logger.info("Saving models to {}".format(adv_save_path))

                    # learner should handle saving/loading -- delegate actor save/load to mac,
                    # use appropriate filenames to do critics, optimizer states
                    learner.save_models(save_path)
                    adv_learner.save_models(adv_save_path)

                episode += args.batch_size_run

                if (runner.t_env - last_log_T) >= args.log_interval:
                    logger.log_stat("episode", episode, runner.t_env)
                    logger.print_recent_stats()
                    last_log_T = runner.t_env
            
            elif args.attack_method == "adv_tar":
                # Run for a whole episode at a time
                episode_batch, adv_episode_batch = runner.run(test_mode=False, learner=learner, adv_test=adv_test)
                buffer.insert_episode_batch(episode_batch)
                adv_buffer.insert_episode_batch(adv_episode_batch)

                if buffer.can_sample(args.batch_size):
                    episode_sample = buffer.sample(args.batch_size)

                    # Truncate batch to only filled timesteps
                    max_ep_t = episode_sample.max_t_filled()
                    episode_sample = episode_sample[:, :max_ep_t]

                    if episode_sample.device != args.device:
                        episode_sample.to(args.device)

                    learner.train(episode_sample, runner.t_env, episode)

                if adv_buffer.can_sample(args.adv_batch_size):
                    adv_episode_sample = adv_buffer.sample(args.adv_batch_size)

                    # Truncate batch to only filled timesteps
                    adv_max_ep_t = adv_episode_sample.max_t_filled()
                    adv_episode_sample = adv_episode_sample[:, :adv_max_ep_t]

                    if adv_episode_sample.device != args.device:
                        adv_episode_sample.to(args.device)

                    adv_learner.train(adv_episode_sample, runner.t_env, episode)

                # Execute test runs once in a while
                n_test_runs = max(1, args.test_nepisode // runner.batch_size)
                if (runner.t_env - last_test_T) / args.test_interval >= 1.0:

                    logger.console_logger.info("t_env: {} / {}".format(runner.t_env, args.t_max))
                    logger.console_logger.info("Estimated time left: {}. Time passed: {}".format(
                        time_left(last_time, last_test_T, runner.t_env, args.t_max),
                        time_str(time.time() - start_time)))
                    last_time = time.time()

                    last_test_T = runner.t_env
                    for _ in range(n_test_runs):
                        runner.run(test_mode=True, learner=learner, adv_test=False)

                if args.save_model and (
                        runner.t_env - model_save_time >= args.save_model_interval or model_save_time == 0):
                    model_save_time = runner.t_env
                    save_path = os.path.join(args.local_results_path, "models", args.unique_token, str(runner.t_env))
                    adv_save_path = os.path.join(args.adv_local_results_path, "models", args.unique_token,
                                                 str(runner.t_env))
                    # "results/models/{}".format(unique_token)
                    os.makedirs(save_path, exist_ok=True)
                    os.makedirs(adv_save_path, exist_ok=True)
                    logger.console_logger.info("Saving models to {}".format(save_path))
                    logger.console_logger.info("Saving models to {}".format(adv_save_path))

                    # learner should handle saving/loading -- delegate actor save/load to mac,
                    # use appropriate filenames to do critics, optimizer states
                    learner.save_models(save_path)
                    adv_learner.save_models(adv_save_path)

                episode += args.batch_size_run

                if (runner.t_env - last_log_T) >= args.log_interval:
                    logger.log_stat("episode", episode, runner.t_env)
                    logger.print_recent_stats()
                    last_log_T = runner.t_env

            elif args.attack_method == "fop_adv_tar":
                # Run for a whole episode at a time
                episode_batch, adv_episode_batch = runner.run(test_mode=False, learner=learner, adv_test=adv_test)
                buffer.insert_episode_batch(episode_batch)
                adv_buffer.insert_episode_batch(adv_episode_batch)

                if buffer.can_sample(args.batch_size):
                    episode_sample, episode_index = buffer.sample_ori(args.batch_size)

                    # Truncate batch to only filled timesteps
                    max_ep_t = episode_sample.max_t_filled()
                    episode_sample = episode_sample[:, :max_ep_t]

                    if episode_sample.device != args.device:
                        episode_sample.to(args.device)

                    learner.train(episode_sample, runner.t_env, episode)

                if adv_buffer.can_sample(args.adv_batch_size):
                    adv_episode_sample = adv_buffer.sample_index(args.adv_batch_size, episode_index)

                    # Truncate batch to only filled timesteps
                    adv_max_ep_t = adv_episode_sample.max_t_filled()
                    adv_episode_sample = adv_episode_sample[:, :adv_max_ep_t]

                    if adv_episode_sample.device != args.device:
                        adv_episode_sample.to(args.device)

                    adv_learner.train_action_loss(adv_episode_sample, episode_sample, runner.t_env, episode)

                # Execute test runs once in a while
                n_test_runs = max(1, args.test_nepisode // runner.batch_size)
                if (runner.t_env - last_test_T) / args.test_interval >= 1.0:

                    logger.console_logger.info("t_env: {} / {}".format(runner.t_env, args.t_max))
                    logger.console_logger.info("Estimated time left: {}. Time passed: {}".format(
                        time_left(last_time, last_test_T, runner.t_env, args.t_max),
                        time_str(time.time() - start_time)))
                    last_time = time.time()

                    last_test_T = runner.t_env
                    for _ in range(n_test_runs):
                        runner.run(test_mode=True, learner=learner, adv_test=False)

                if args.save_model and (
                        runner.t_env - model_save_time >= args.save_model_interval or model_save_time == 0):
                    model_save_time = runner.t_env
                    save_path = os.path.join(args.local_results_path, "models", args.unique_token, str(runner.t_env))
                    adv_save_path = os.path.join(args.adv_local_results_path, "models", args.unique_token,
                                                 str(runner.t_env))
                    # "results/models/{}".format(unique_token)
                    os.makedirs(save_path, exist_ok=True)
                    os.makedirs(adv_save_path, exist_ok=True)
                    logger.console_logger.info("Saving models to {}".format(save_path))
                    logger.console_logger.info("Saving models to {}".format(adv_save_path))

                    # learner should handle saving/loading -- delegate actor save/load to mac,
                    # use appropriate filenames to do critics, optimizer states
                    learner.save_models(save_path)
                    adv_learner.save_models(adv_save_path)

                episode += args.batch_size_run

                if (runner.t_env - last_log_T) >= args.log_interval:
                    logger.log_stat("episode", episode, runner.t_env)
                    logger.print_recent_stats()
                    last_log_T = runner.t_env
        else:
            # Run for a whole episode at a time
            episode_batch = runner.run(test_mode=False, learner=learner, adv_test=adv_test)
            buffer.insert_episode_batch(episode_batch)

            if buffer.can_sample(args.batch_size):
                episode_sample = buffer.sample(args.batch_size)

                # Truncate batch to only filled timesteps
                max_ep_t = episode_sample.max_t_filled()
                episode_sample = episode_sample[:, :max_ep_t]

                if episode_sample.device != args.device:
                    episode_sample.to(args.device)

                learner.train(episode_sample, runner.t_env, episode)

            # Execute test runs once in a while
            n_test_runs = max(1, args.test_nepisode // runner.batch_size)
            if (runner.t_env - last_test_T) / args.test_interval >= 1.0:

                logger.console_logger.info("t_env: {} / {}".format(runner.t_env, args.t_max))
                logger.console_logger.info("Estimated time left: {}. Time passed: {}".format(
                    time_left(last_time, last_test_T, runner.t_env, args.t_max), time_str(time.time() - start_time)))
                last_time = time.time()

                last_test_T = runner.t_env
                for _ in range(n_test_runs):
                    runner.run(test_mode=True, learner=learner, adv_test=False)

            if args.save_model and (runner.t_env - model_save_time >= args.save_model_interval or model_save_time == 0):
                model_save_time = runner.t_env
                save_path = os.path.join(args.local_results_path, "models", args.unique_token, str(runner.t_env))
                # "results/models/{}".format(unique_token)
                os.makedirs(save_path, exist_ok=True)
                logger.console_logger.info("Saving models to {}".format(save_path))

                # learner should handle saving/loading -- delegate actor save/load to mac,
                # use appropriate filenames to do critics, optimizer states
                learner.save_models(save_path)

            episode += args.batch_size_run

            if (runner.t_env - last_log_T) >= args.log_interval:
                logger.log_stat("episode", episode, runner.t_env)
                logger.print_recent_stats()
                last_log_T = runner.t_env

    runner.close_env()
    logger.console_logger.info("Finished Training")


def args_sanity_check(config, _log):
    # set CUDA flags
    # config["use_cuda"] = True # Use cuda whenever possible!
    if config["use_cuda"] and not th.cuda.is_available():
        config["use_cuda"] = False
        _log.warning("CUDA flag use_cuda was switched OFF automatically because no CUDA devices are available!")

    if config["test_nepisode"] < config["batch_size_run"]:
        config["test_nepisode"] = config["batch_size_run"]
    else:
        config["test_nepisode"] = (config["test_nepisode"] // config["batch_size_run"]) * config["batch_size_run"]

    return config
