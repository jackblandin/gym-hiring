import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns  # noqa


def play_episode(env, model, eps):
    """
    Plays a single episode. During play, the model is updated, and the total
    reward is accumulated and returned.

    Parameters
    ----------
    env : gym.Env
        Environment.
    model : <TBD>
        Model instance.
    eps : numeric
        Epsilon used in epsilon-greedy.

    Returns
    -------
    numeric
        Total reward accumualted during episode.

    """
    obs = env.reset()
    done = False
    totalreward = 0
    timestep = 0
    while not done:

        # Choose an action based on current observation.
        action = model.select_action(obs, eps)
        prev_obs = obs

        # Take chosen action.
        obs, reward, done, _ = env.step(action)

        totalreward += reward

        # Update the model
        model.add_experience(prev_obs, action, reward, obs, done)
        model.train(timestep)

        timestep += 1

    return totalreward


def running_avg(totalrewards, t, window):
    return totalrewards[max(0, t-window):(t+1)].mean()


def plot_running_avg(totalrewards, window):
    N = len(totalrewards)
    ravg = np.empty(N)
    for t in range(N):
        ravg[t] = running_avg(totalrewards, t, window)
    plt.plot(ravg)
    plt.title('Running Average')
    plt.show()
