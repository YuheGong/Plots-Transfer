import pandas as pd
import numpy as np
from scipy.interpolate import make_interp_spline
from tensorboard.backend.event_processing import event_accumulator
from tqdm import tqdm
import matplotlib.pyplot as plt


def auto_text(rects):
    for rect in rects:
        plt.text(x=rect.get_x()+0.1, y=rect.get_height()-0.5,
                 s=np.round(rect.get_height(), decimals=3),
                 ha='left', va='bottom')

def plot_function(result, algo):
    n_bins = 1
    fig, ax = plt.subplots()
    a = ax.bar(x=1, height=result['td3'])
    b = ax.bar(x=2, height=result['sac'])
    c = ax.bar(x=3, height=result['ppo'])
    d = ax.bar(x=4, height=result['promp'])
    e = ax.bar(x=5, height=result['episodic_td3'])
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
    auto_text(e)
    #plt.legend()
    plt.xticks(range(1, 6), ['TD3', 'SAC', 'PPO', 'PMP', 'Epi.TD3'])
    #ax.xaxis.tick_top()
    #ax.get_legend().remove()
    #plt.show()
    #plt.hist(, bins=bins)
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
                rewards.append(np.max(np.array(
                    [np.array(h.value) for
                     h in histograms])))
                steps.append(np.array(
                    [np.array(h.step) for
                     h in histograms]))


                # print(steps[-1][-1], steps[-1].shape)
    rewards = np.array(rewards)
    steps = np.array(steps)
    var = np.sqrt(np.std(rewards, axis=0))
    best = np.mean(rewards)
    rewards = rewards.mean(axis=0)
    steps = steps.mean(axis=0)

    result['eval/mean_reward'] = rewards
    result['step'] = steps
    result['var'] = var
    result['best'] = best
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
                rewards.append(np.max(np.array(
                    [np.array(h.value) for
                     h in histograms])))
                steps.append(np.array(
                    [np.array(h.step) for
                     h in histograms]))
                # print(steps[-1][-1], steps[-1].shape)
    # assert 1==123
    rewards = np.array(rewards)
    steps = np.array(steps)
    var = np.sqrt(np.std(rewards, axis=0))
    best = np.mean(rewards)
    rewards = rewards.mean(axis=0)
    steps = steps.mean(axis=0)


    result['eval/mean_reward'] = rewards
    result['step'] = steps
    result['var'] = var
    result['best'] = best
    return result


folder = "data/MujocoReacher"
value = "eval/mean_reward"

id = "4"
env = "ALRReacherBalanceIP-v" + id
env_promp = "ALRReacherBalanceProMPIP-v" + id

for v in range(1):
    results = {}
    algo = "td3"
    name = algo + "/" + env
    results[algo] = csv_save(folder, name, 'TD3')['best']
    #plot_function(result, algo)

    algo = "sac"
    name = algo + "/" + env
    results[algo] = csv_save(folder, name, "SAC")['best']
    #plot_function(result, algo)

    algo = "ppo"
    name = algo + "/" + env
    results[algo] = csv_save(folder, name, 'PPO')['best']
    #plot_function(result, algo)


    algo = "promp"
    name = algo + "/" + env_promp
    results[algo] = csv_save_promp(folder, name, "")['best']
    #plot_function_promp(result, algo)

    algo = "episodic_td3"
    name = algo + "/" + env  # + algo + "-v{}".format(v)
    results[algo] = csv_save(folder, name, "run")['best']

    plot_function(results, algo)



# csv_save(folder, name)
# plt.title("ALR Reacher - Line trajectory")
#plt.title("ALR Reacher - Line Trajectory")
#plt.title("Square Root of Standard Deviation")
#plt.ylim(ymin=1)
#plt.ylim(ymax=2)
#

plt.xlabel("algorithm")
plt.ylabel("best reward")
#plt.savefig("fetchv" + "id" +"_variance.png")
import tikzplotlib

# tikzplotlib.save("latex/alr3.tex")
tikzplotlib.save("./data/MujocoReacher/mujocov" + id +"_best.tex")

plt.show()



