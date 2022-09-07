import pandas as pd
import numpy as np
from scipy.interpolate import make_interp_spline
from tensorboard.backend.event_processing import event_accumulator
from tqdm import tqdm
import tikzplotlib
import matplotlib.pyplot as plt

fill = True
skip = 50
plot_steps = 5.5
#env = "window"
env = "button"
env = "faucet"
Dense = "_dense"
env = env + Dense
value = "mean_reward"
#value = "success_rate"
#value = "last_success"

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


    X = plot_samples/plot_samples[-1] * plot_steps
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

    X = plot_samples / plot_samples[-1] * plot_steps
    Y = plot_mean
    #Y = moving_average(Y, 10)
    Z = var  # * 0.5
    #Z = moving_average(Z, 10)

    #plt.plot(X, Y, label="Epi.TD3")
    plt.plot(X, Y)
    if fill:
        plt.fill_between(X, Y - Z, Y + Z, alpha=0.2)


def csv_save(folder, name, algo, term):
    # save csv file
    steps = []
    rewards = []
    result = {}
    for i in range(num, NUM):
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
                     h in histograms if h.step < plot_steps * 1.e6]))
                steps.append(np.array(
                    [np.array(h.step) for
                     h in histograms if h.step < plot_steps * 1.e6]))

                # print(steps[-1][-1], steps[-1].shape)
    # assert 1==123
    shape = []
    for i in range(num-1, NUM-1):
        shape.append(steps[i].shape)
    index = np.where(np.array(shape)==np.min(np.array(shape)))[0][0]
    new_steps = list(np.zeros((NUM-1,1)))
    new_rewards = list(np.zeros((NUM-1,1)))

    for j in range(num - 1, NUM - 1):
        for i in range(steps[j].shape[0]):
            if steps[j][i] in steps[index]:
                new_steps[j] = np.concatenate([new_steps[j], np.array(steps[j][i]).reshape(1)])
                new_rewards[j] = np.concatenate([new_rewards[j], np.array(rewards[j][i]).reshape(1)])

    rewards = np.array(new_rewards)[:, ::skip]
    steps = np.array(new_steps)[:, ::skip]
    var = np.std(rewards, axis=0)
    rewards = rewards.mean(axis=0)
    steps = steps.mean(axis=0)

    result['eval/mean_reward'] = rewards
    result['step'] = steps
    result['var'] = var
    return result


if "button" in env:
    env = "Meta-context-dense-button-press-v2"
    folder = "data/Context"
    if "mean_reward" in value:
        term = "eval/mean_reward"
        up = 3500
        low = 0
    elif "success_rate" in value:
        term = "eval/success_rate"
        up = 1.1
        low = -0.1
    elif "last_success" in value:
        term = "eval/last_success"
        up = 1.1
        low = -0.1
elif "faucet" in env:
    env = "Meta-context-dense-faucet-open-v2"
    folder = "data/Context"
    if "mean_reward" in value:
        term = "eval/mean_reward"
        up = 3500
        low = 0
    elif "success_rate" in value:
        term = "eval/success_rate"
        up = 1.1
        low = -0.1
    elif "last_success" in value:
        term = "eval/last_success"
        up = 1.1
        low = -0.1



for v in range(1):

    NUM = 6
    num = 1
    algo = "td3"
    name = algo + "/" + env
    result = csv_save(folder, name, 'TD3', term)
    plot_function(result, algo)

    algo = "sac"
    name = algo + "/" + env
    result = csv_save(folder, name, "SAC", term)
    plot_function(result, algo)

    NUM = 6
    num = 1
    algo = "ppo"
    name = algo + "/" + env
    result = csv_save(folder, name, 'PPO', term)
    plot_function(result, algo)

    NUM = 6
    num = 1
    algo = "episodic_td3"
    name = algo + "/" + env  # + algo + "-v{}".format(v)
    result = csv_save(folder, name, "run", term)
    plot_function_et3(result, algo)

plt.xlabel("timesteps(1e6)")
if "mean_reward" in value:
    plt.ylabel("rewards")
elif "last_dist" in value:
    plt.ylabel("distance from object to target")
elif "last_success" in value:
    plt.ylabel("success rate")

plt.ylim(ymin=low)
# plt.title("ALRReacher-v3")
# plt.ylim(ymin=-100)
plt.ylim(ymax=up)
plt.yticks()
if "last_success" in value:
    plt.yticks([1/10,2/10,3/10,4/10,5/10,6/10,7/10,8/10,9/10,10/10], ["1/10","2/10","3/10","4/10","5/10","6/10","7/10","8/10","9/10","10/10"])
plt.legend()

tikzplotlib.save("./" + folder + '/' + env + "+" + value + ".tex")

plt.show()


