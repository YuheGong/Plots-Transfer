import pandas as pd
import numpy as np
from scipy.interpolate import make_interp_spline
from tensorboard.backend.event_processing import event_accumulator
from tqdm import tqdm
import matplotlib.pyplot as plt


def plot_function(result, algo):
    plot_mean = result['eval/mean_reward'].reshape(-1)
    plot_samples = np.array(result['step']).reshape(-1)
    var = np.array(result['var']).reshape(-1)


    X = plot_samples/1.e6
    Y = plot_mean
    Z = var

    plt.plot(X, Z, label=algo.upper())
    #plt.fill_between(X, Y - Z, Y + Z, alpha=0.2)


def plot_function_pt3(result, algo):
    plot_mean = result['eval/mean_reward'].reshape(-1)
    plot_samples = np.array(result['step']).reshape(-1)
    var = np.array(result['var']).reshape(-1)

    X = plot_samples/1.e6
    Y = plot_mean
    Z = var

    plt.plot(X, Z, label="Epi.TD3")
    #plt.fill_between(X, Y - Z, Y + Z, alpha=0.2)


def plot_function_promp(result, algo):
    plot_mean = result['eval/mean_reward'].reshape(-1)
    plot_samples = np.array(result['step']).reshape(-1)
    var = np.array(result['var']).reshape(-1)

    X = plot_samples/1.e6 /0.889 * 2
    Y = plot_mean
    Z = var

    plt.plot(X, Z, label="PMP")
    #plt.fill_between(X, Y - Z, Y + Z, alpha=0.2)


def csv_save(folder, name, algo):
    # save csv file
    steps = []
    rewards = []
    result = {}
    for i in range(1, 6):
        path = "./" \
               + folder + "/" + name
        in_path = path + '_' + f'{i}' + '/' + algo + '_1'
        print("path",path)
        ex_path = path + '_' + f'_{i}' + '/' + "eval_reward_mean.csv"
        event_data = event_accumulator.EventAccumulator(in_path)  # a python interface for loading Event data
        event_data.Reload()  # synchronously loads all of the data written so far b
        # print(event_data.Tags())  # print all tags
        event_data.Reload()
        tags = event_data.Tags()

        keys = event_data.scalars.Keys()  # get all tags,save in a list
        for hist in tags['scalars']:
            if hist == 'eval/mean_reward':
                histograms = event_data.scalars.Items("eval/mean_reward")
                rewards.append(np.array(
                    [np.array(h.value) for
                     h in histograms]))
                steps.append(np.array(
                    [np.array(h.step) for
                     h in histograms]))

                # print(steps[-1][-1], steps[-1].shape)
    # assert 1==123
    rewards = np.array(rewards)[:, ::10]
    steps = np.array(steps)[:, ::10]
    var = np.sqrt(np.std(rewards, axis=0))
    rewards = rewards.mean(axis=0)
    steps = steps.mean(axis=0)

    result['eval/mean_reward'] = rewards
    result['step'] = steps
    result['var'] = var
    return result


def csv_save_promp(folder, name, algo):
    # save csv file
    steps = []
    rewards = []
    result = {}
    for i in range(1,6):
        print(i)
        path = "./" \
               + folder + "/" + name
        in_path = path + '_' + f'{i}' + '/' + algo
        ex_path = path + '_' + f'_{i}' + '/' + "eval_reward_mean.csv"
        event_data = event_accumulator.EventAccumulator(in_path)  # a python interface for loading Event data
        event_data.Reload()  # synchronously loads all of the data written so far b
        # print(event_data.Tags())  # print all tags
        event_data.Reload()
        tags = event_data.Tags()

        keys = event_data.scalars.Keys()  # get all tags,save in a list
        for hist in tags['scalars']:
            if hist == 'eval/mean_reward':
                histograms = event_data.Scalars(hist)
                rewards.append(np.array(
                    [np.array(h.value) for
                     h in histograms]))
                steps.append(np.array(
                    [np.array(h.step) for
                     h in histograms]))
                # print(steps[-1][-1], steps[-1].shape)
    # assert 1==123
    rewards = np.array(rewards)[:, ::10]
    steps = np.array(steps)[:, ::10]
    var = np.sqrt(np.std(rewards, axis=0))
    rewards = rewards.mean(axis=0)
    steps = steps.mean(axis=0)

    result['eval/mean_reward'] = rewards
    result['step'] = steps
    result['var'] = var
    return result


folder = "data/MujocoReacher"
value = "eval/mean_reward"

id = "4"
env = "ALRReacherBalanceIP-v" + id
env_promp = "ALRReacherBalanceProMPIP-v" + id

for v in range(1):
    algo = "td3"
    name = algo + "/" + env
    result = csv_save(folder, name, 'TD3')
    plot_function(result, algo)

    algo = "sac"
    name = algo + "/" + env
    result = csv_save(folder, name, "SAC")
    plot_function(result, algo)

    algo = "ppo"
    name = algo + "/" + env
    result = csv_save(folder, name, 'PPO')
    plot_function(result, algo)

    algo = "promp"
    name = algo + "/" + env_promp
    result = csv_save_promp(folder, name, "")
    plot_function_promp(result, algo)

    algo = "episodic_td3"
    name = algo + "/" + env  # + algo + "-v{}".format(v)
    result = csv_save(folder, name, "run")
    plot_function_pt3(result, algo)


# csv_save(folder, name)
# plt.title("ALR Reacher - Line trajectory")
#plt.title("ALR Reacher - Line Trajectory")
#plt.title("Square Root of Standard Deviation")
plt.xlabel("timesteps(1e6)")
plt.ylabel("value")
plt.ylim(ymin=0)
# plt.title("ALRReacher-v3")
# plt.ylim(ymin=-100)
plt.ylim(ymax=5)
plt.legend()
#plt.show()
#plt.savefig("fetchv" + "id" +"_variance.png")

import tikzplotlib

# tikzplotlib.save("latex/alr3.tex")
tikzplotlib.save("./data/MujocoReacher/mujocov" + id +"_best.tex")



