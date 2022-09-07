import pandas as pd
import numpy as np
from scipy.interpolate import make_interp_spline
from tensorboard.backend.event_processing import event_accumulator
from tqdm import tqdm
import tikzplotlib
import matplotlib.pyplot as plt

fill = False
env = "window_dense"
env = "coffee"
#env = "soccer_dense"
value = "last_success"


def auto_text(rects):
    for rect in rects:
        plt.text(x=rect.get_x()+0.25, y=rect.get_height()+0.01,
                 s=np.round(rect.get_height(), decimals=3),
                 ha='left', va='bottom')

def plot_function(result, algo):
    n_bins = 1
    fig, ax = plt.subplots()
    a = ax.bar(x=1, height=result['td3']['eval/mean_reward'])
    b = ax.bar(x=2, height=result['sac']['eval/mean_reward'])
    c = ax.bar(x=3, height=result['ppo']['eval/mean_reward'])
    #d = ax.bar(x=4, height=result['promp']['eval/mean_reward'])
    d = ax.bar(x=4, height=result['episodic_td3']['eval/mean_reward'])
    '''
    plt.bar(x=1, height=result['td3'], label="TD3")
    plt.bar(x=2, height=result['sac'], label="SAC")
    plt.bar(x=3, height=result['ppo'], label="PPO")
    plt.bar(x=4, height=result['promp'], label="ProMP")
    plt.bar(x=5, height=result['episodic_td3'], label="Episodic TD3")
    '''
    auto_text(a)
    auto_text(b)
    auto_text(c)
    auto_text(d)
    #auto_text(e)
    #plt.legend()
    plt.xticks(range(1, 5), ['TD3', 'SAC', 'PPO', 'Epi.TD3'])
    #ax.xaxis.tick_top()
    #ax.get_legend().remove()
    #plt.show()
    #plt.hist(, bins=bins)
    #plt.fill_between(X, Y - Z, Y + Z, alpha=0.2)


def csv_save(folder, name, algo, term):
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
            if hist == term:
                histograms = event_data.scalars.Items(hist)
                rewards.append(np.array(
                    [np.array(h.value) for
                     h in histograms]))
                steps.append(np.array(
                    [np.array(h.step) for
                     h in histograms]))

                # print(steps[-1][-1], steps[-1].shape)
    # assert 1==123
    rewards = np.array(rewards)#[:, ::skip]
    steps = np.array(steps)#[:, ::skip]
    ###teps = steps.mean(axis=0)

    result['eval/mean_reward'] = np.sum(rewards==1)/5/rewards.shape[1]
    #result['step'] = steps
    #result['var'] = var
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
    # assert 1==123
    rewards = np.array(rewards)  # [:, ::skip]
    steps = np.array(steps)  # [:, ::skip]
    ###teps = steps.mean(axis=0)

    result['eval/mean_reward'] = np.sum(rewards==1)/5/rewards.shape[1]
    # result['step'] = steps
    # result['var'] = var
    return result


if "window" in env:
    if "dense" in env:
        env = "Meta-dense-window-open-v2"
        env_promp = "Meta-promp-dense-window-open-v2"
    else:
        env = "Meta-window-open-v2"
        env_promp = "Meta-promp-window-open-v2"
    folder = "data/Meta/window_open"
    if "mean_reward" in value:
        term = "eval/mean_reward"
        if "dense" in env:
            up = 1500
            low = 0
        else:
            up = 10.1
            low = 0
    elif "last_dist" in value:
        term = "eval/last_object_to_target"
        up = 0.3
        low = 0
    elif "last_success" in value:
        term = "eval/last_success"
        up = 1.1
        low = -0.1
elif "soccer" in env:
    if "dense" in env:
        env = "Meta-dense-soccer-v2"
        env_promp = "Meta-promp-dense-soccer-v2"
    else:
        env = "Meta-soccer-v2"
        env_promp = "Meta-promp-soccer-v2"
    folder = "data/Meta/soccer"
    if "mean_reward" in value:
        term = "eval/mean_reward"
        if "dense" in env:
            up = 1500
            low = 0
        else:
            up = 10.1
            low = 0
    elif "last_dist" in value:
        term = "eval/last_object_to_target"
        up = 0.8
        low = 0
    elif "last_success" in value:
        term = "eval/last_success"
        up = 1.1
        low = -0.1
elif "coffee" in env:
    if "dense" in env:
        env = "Meta-dense-coffee-push-v2"
        env_promp = "Meta-promp-dense-coffee-push-v2"
    else:
        env = "Meta-coffee-push-v2"
        env_promp = "Meta-promp-coffee-push-v2"
    folder = "data/Meta/coffee_push"
    if "mean_reward" in value:
        term = "eval/mean_reward"
        if "dense" in env:
            up = 1500
            low = 0
        else:
            up = 10.1
            low = 0
    elif "last_dist" in value:
        term = "eval/last_object_to_target"
        up = 0.3
        low = 0
    elif "last_success" in value:
        term = "eval/last_success"
        up = 1.1
        low = -0.1


results = {}

for v in range(1):

    algo = "td3"
    name = algo + "/" + env
    result = csv_save(folder, name, 'TD3', term)
    results[algo] = result
    #plot_function(result, algo)

    algo = "sac"
    name = algo + "/" + env
    result = csv_save(folder, name, "SAC", term)
    results[algo] = result
    #plot_function(result, algo)

    algo = "ppo"
    name = algo + "/" + env
    result = csv_save(folder, name, 'PPO', term)
    results[algo] = result
    #plot_function(result, algo)

    #algo = "promp"
    #name = algo + "/" + env_promp
    #result = csv_save_promp(folder, name, "", term)
    #results[algo] = result
    #plot_function_promp(result, algo)

    algo = "episodic_td3"
    name = algo + "/" + env  # + algo + "-v{}".format(v)
    result = csv_save(folder, name, "run", term)
    results[algo] = result
    #plot_function_pt3(result, algo)

    plot_function(results, algo)



plt.xlabel("algorithm")
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
plt.legend()

tikzplotlib.save("./" + folder + '/' + env + "+" + "success_episode" + ".tex")

plt.show()


