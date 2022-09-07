import pandas as pd
import numpy as np
from scipy.interpolate import make_interp_spline
from tensorboard.backend.event_processing import event_accumulator
from tqdm import tqdm
import tikzplotlib
import matplotlib.pyplot as plt

fill = True
skip = 50
s = 10
env = "HopperXYJumpStep-v0"
#env = "HopperXYJump-v0"
env = "HopperXYJumpMiddle-v0"
#env_promp = "HopperXYJumpStepProMP-v0"
env_promp = "HopperXYJumpMiddleProMP-v0"
value = "mean_reward"
value = "max_height"
value = "min_goal_dist"


if "Step" in env:
    folder = "data/Hopper/Step"
    TIMESTEP = 4e6
    if "mean_reward" in value:
        term = "eval/mean_reward"
        up = 3500
        low = 1000
    elif "max_height" in value:
        term = "eval/max_height"
        up = 1.9
        low = 1.45
    elif "min_goal_dist" in value:
        term = "eval/min_goal_dist"
        up = 1.3
        low = 0

elif "Middle" in env:
    TIMESTEP = 1e7
    folder = "data/Hopper/Middle"
    if "mean_reward" in value:
        term = "eval/mean_reward"
        up = 280
        low = 220
    elif "max_height" in value:
        term = "eval/max_height"
        up = 1.8
        low = 1.45
    elif "min_goal_dist" in value:
        term = "eval/min_goal_dist"
        up = 1.3
        low = 0

def moving_average(interval, windowsize):
    window = np.ones(int(windowsize))/float(windowsize)
    re = np.convolve(interval, window, "same")
    return re
def plot_function(result, algo):
    plot_mean = result['eval/mean_reward'].reshape(-1)
    plot_samples = np.array(result['step']).reshape(-1)
    var = np.array(result['var']).reshape(-1)
    """
    X = np.linspace(plot_samples.min(), plot_samples.max(), 10)
    Y = make_interp_spline(plot_samples, plot_mean)(X)

    Z = make_interp_spline(plot_samples, var)(X)

    plt.plot(X, Y)
    if fill:
        plt.fill_between(X, Y - Z, Y + Z, alpha=0.2)
    """
    #plt.show()


    X = plot_samples/1.e6  / plot_samples[-1] * s
    Y = plot_mean
    #Y = moving_average(Y, 10)
    Z = var #* 0.5
    #Z = moving_average(Z, 10)

    #plt.plot(X, Y, label=algo.upper())
    plt.plot(X, Y)
    if fill:
        plt.fill_between(X, Y - Z, Y + Z, alpha=0.2)


def plot_function_et3(result, algo):
    plot_mean = result['eval/mean_reward'].reshape(-1)
    plot_samples = np.array(result['step']).reshape(-1)
    var = np.array(result['var']).reshape(-1)

    X = plot_samples / 1.e6  / plot_samples[-1] * s
    Y = plot_mean
    #Y = moving_average(Y, 10)
    Z = var  # * 0.5
    #Z = moving_average(Z, 10)

    #plt.plot(X, Y, label="Epi.TD3")
    plt.plot(X, Y)
    if fill:
        plt.fill_between(X, Y - Z, Y + Z, alpha=0.2)


def plot_function_promp(result, algo):
    plot_mean = result['eval/mean_reward'].reshape(-1)
    plot_samples = np.array(result['step']).reshape(-1)
    var = np.array(result['var']).reshape(-1)

    X = plot_samples/1.e6  / plot_samples[-1] * s
    Y = plot_mean
    Z = var

    #plt.plot(X, Y, label="PMP")
    plt.plot(X, Y)
    if fill:
        plt.fill_between(X, Y - Z, Y + Z, alpha=0.1)


def csv_save(folder, name, algo, term):
    # save csv file
    steps = []
    rewards = []
    result = {}
    for i in range(1, NUM):
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
            if hist == term:
                histograms = event_data.scalars.Items(hist)
                rewards.append(np.array(
                    [np.array(h.value) for
                     h in histograms if h.step < TIMESTEP]))
                steps.append(np.array(
                    [np.array(h.step) for
                     h in histograms if h.step < TIMESTEP]))

                # print(steps[-1][-1], steps[-1].shape)
    # assert 1==123
    rewards = np.array(rewards)[:, ::skip]
    steps = np.array(steps)[:, ::skip]
    var = np.std(rewards, axis=0)
    rewards = rewards.mean(axis=0)
    steps = steps.mean(axis=0)

    result['eval/mean_reward'] = rewards
    result['step'] = steps
    result['var'] = var
    return result


def csv_save_promp(folder, name, algo, term):
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
            if hist == term:
                histograms = event_data.Scalars(hist)
                rewards.append(np.array(
                    [np.array(h.value) for
                     h in histograms]))
                steps.append(np.array(
                    [np.array(h.step) for
                     h in histograms]))
                # print(steps[-1][-1], steps[-1].shape)
    # assert 1==123
    rewards = np.array(rewards)[:, ::skip]
    steps = np.array(steps)[:, ::skip]
    var = np.std(rewards, axis=0)
    rewards = rewards.mean(axis=0)
    steps = steps.mean(axis=0)

    result['eval/mean_reward'] = rewards
    result['step'] = steps
    result['var'] = var
    return result







for v in range(1):
    NUM = 6
    algo = "td3"
    name = algo + "/" + env
    result = csv_save(folder, name, 'TD3', term)
    plot_function(result, algo)

    algo = "sac"
    name = algo + "/" + env
    result = csv_save(folder, name, "SAC", term)
    plot_function(result, algo)

    algo = "ppo"
    name = algo + "/" + env
    result = csv_save(folder, name, 'PPO', term)
    plot_function(result, algo)

    algo = "promp"
    name = algo + "/" + env_promp
    result = csv_save_promp(folder, name, "", term)
    plot_function_promp(result, algo)


    algo = "episodic_td3"
    name = algo + "/" + env  # + algo + "-v{}".format(v)
    result = csv_save(folder, name, "run", term)
    plot_function_et3(result, algo)



plt.xlabel("timesteps(1e6)")
if "mean_reward" in value:
    plt.ylabel("rewards")
elif "height" in value:
    plt.ylabel("max height")
elif "dist" in value:
    plt.ylabel("distance from foot to goal")

plt.ylim(ymin=low)
# plt.title("ALRReacher-v3")
# plt.ylim(ymin=-100)
plt.ylim(ymax=up)
plt.yticks()
if "last_success" in value:
    plt.yticks([1/5,2/5,3/5,4/5,5/5], ["1/5","2/5","3/5","4/5","5/5"])
plt.legend()

tikzplotlib.save("./" + folder + '/' + env + "+" + value + ".tex")

plt.show()


