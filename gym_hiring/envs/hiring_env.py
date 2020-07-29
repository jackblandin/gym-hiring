# Core modules
import logging.config

# 3rd party modules
import gym
import numpy as np
from gym.spaces import Discrete, Tuple


OBS_GENDER_MAP = {
    0: 'FEMALE',
    1: 'MALE',
}

ACTION_HIRE = 0
ACTION_NOT_HIRE = 1
ACTION_MAP = {
    ACTION_HIRE: 'HIRE',
    ACTION_NOT_HIRE: 'NOT_HIRE',
}


class HiringState:
    """
    Object encapsulating the RL "state" for the hiring environment.

    Parameters
    ----------
    cand_attr_names : array-like, optional
        Names of candidate attributes.
    cand_attr_values : array-like, optional
        Values for each candidate attribute.
    cand_prod_score : float [0, 1], optional
        Candidate's productivity score.
    """

    def __init__(self, cand_attr_names=None, cand_attr_values=None,
                 cand_prod_score=None):
        self.cand_attr_names = cand_attr_names
        self.cand_attr_values = cand_attr_values
        self.cand_prod_score = cand_prod_score


class HiringEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, max_steps_per_episode, attr_probs, attr_names,
                 prod_score_fn):
        """
        OpenAI Gym environment for the Hiring environment.

        Parameters
        ----------
        max_steps_per_episode : int
            Maximum allowed steps per episode. This will define how long an
            episode lasts, since the environment does not end otherwise.
        attr_probs : <array-like<array-like<float>>, [0, 1]
            2-dimensional array. Each outer array represents a candidate
            attribute. Each inner array represents the probability of each
            value for that attribute value. Each inner array must sum to 1.
        attr_names : array-like<str>
            Labels for each attribute. Pairs with attr_probs.
        prod_score_fn : function
            The function that takes as input a candidate's attributes and
            outputs a productivity score.

        Attributes
        ----------
        curr_episode : int
            Current episode as a count.
        action_episode_memory : list<<list<int>>
            History of actions taken in each episode.
        observation_episode_memory : list<list<int>>
            History of observations observed in each episode.
        reward_episode_memory : list<list<int, float>>
            History of rewards observed in each episode.
        curr_step : int
            Current timestep in episode, as a count.
        action_space : gym.spaces.Discrete
            Action space.
        observation_space : gym.spaces.Discrete
            Observation space.
        cur_state : HiringState
            Current state space.
        attr_counts : np.ndarray
            The count of each candidate attribute value that was interviewed.
        attr_hire_counts : np.ndarray
            The count of each candidate attribute value that was hired.

        Examples
        --------
        >>> def prod_score(cand_attr_values):
        >>>     gender = cand_attr_values[0]
        >>>     is_male = gender
        >>>     is_female = gender - 1
        >>>     score = (.5*is_male) + (.5*is_female)
        >>>     return score
        >>>
        >>> args = {'max_steps_per_episode': 100,
        >>>         'attr_probs': [[.5, .5]],
        >>>         'attr_names': ['gender'],
        >>>         'prod_score_fn': prod_score}
        >>>
        >>> env = gym.make('hiring-v0', **args)
        """
        self.max_steps_per_episode = max_steps_per_episode
        self.attr_probs = attr_probs
        self.attr_names = attr_names
        self.prod_score_fn = prod_score_fn
        self.cur_state = HiringState()
        self.attr_counts = np.zeros_like(attr_probs, dtype=int)
        self.attr_hire_counts = np.zeros_like(attr_probs, dtype=int)

        self.__version__ = "0.0.1"
        logging.info("Hiring - Version {}".format(self.__version__))

        self.curr_episode = -1  # Set to -1 b/c reset() adds 1 to episode
        self.action_episode_memory = []
        self.observation_episode_memory = []
        self.reward_episode_memory = []

        self.curr_step = 0

        # Define what the agent can do: HIRE, NOT HIRE
        self.action_space = Discrete(2)

        # Define what agent can observe. E.g.:
        #   - Gender : male or female : Discrete(2)
        #   - Productivity Score : integer from 0 to 9 : Discrete(10)
        self.observation_space = Tuple(
            [Discrete(len(prob)) for prob in attr_probs] + [Discrete(10)])

        self.reset()

    def step(self, action):
        """
        The agent takes a step in the environment.

        Parameters
        ----------
        action : int
            Action to take.

        Returns
        -------
        ob, reward, episode_over, info : tuple
            ob : list
                A list of ones or zeros which together represent the state of
                the environment.
            reward : float
                Amount of reward achieved by the previous action. The scale
                varies between environments, but the goal is always to increase
                your total reward.
            episode_over : bool
                Whether it's time to reset the environment again. Most (but not
                all) tasks are divided up into well-defined episodes, and done
                being True indicates the episode has terminated. (For example,
                perhaps the pole tipped too far, or you lost your last life.)
            info : dict
                Diagnostic information useful for debugging. It can sometimes
                be useful for learning (for example, it might contain the raw
                probabilities behind the environment's last state change).
                However, official evaluations of your agent are not allowed to
                use this for learning.
        """
        done = self.curr_step >= self.max_steps_per_episode
        if done:
            raise RuntimeError("Episode is done")
        self.curr_step += 1
        should_reset = self._take_action(action)
        # Recompute done since action may have modified it
        done = self.curr_step >= self.max_steps_per_episode
        reward = self._get_reward(action)
        # Perform resets that happen after each timestep
        if should_reset:
            self._step_reset()

        ob = self._get_obs()  # Has to come after _step_reset and _take_action

        self.action_episode_memory[self.curr_episode].append(action)
        self.observation_episode_memory[self.curr_episode].append(ob)
        self.reward_episode_memory[self.curr_episode].append(reward)

        return ob, reward, done, {}

    def reset(self):
        """
        Reset the state of the environment and returns an initial observation.

        Returns
        -------
        object
            The initial observation of the space.
        """
        self.curr_step = 0
        self.curr_episode += 1
        self._step_reset()
        self.action_episode_memory.append([])
        self.observation_episode_memory.append([])
        self.reward_episode_memory.append([])
        obs = self._get_obs()
        return obs

    def render(self, mode='human'):
        return

    def close(self):
        pass

    def translate_obs(self, obs):
        """
        Enables interpretation of the observation in plain English. Useful for
        debugging and analyzing learned policies.

        Parameters
        ----------
        obs : list or array-like
            The observation to be translated.

        Returns
        -------
        str
            A representation of the observation in English.
        """
        assert len(obs) == 2

        if obs[0] not in OBS_GENDER_MAP:
            raise ValueError('Invalid observation: '.format(obs))
        else:
            gender = OBS_GENDER_MAP[obs[0]]
            prod_score = obs[1]
            return 'Gender: {}, Prod. Score: {:.2f}'.format(gender, prod_score)

    def translate_action(self, action):
        """
        Enables interpretation of the action in plain English. Useful for
        debugging and analyzing learned policies.

        Parameters
        ----------
        action : int
            The action to be translated.

        Returns
        -------
        str
            A representation of the action in English.
        """
        return ACTION_MAP[action]

    def _take_action(self, action):
        """
        How to change the environment when taking an action.

        Parameters
        ----------
        action : int
            Action.

        Returns
        -------
        bool
            Whether or not to reset the state.
        """
        should_reset = True

        attr_vals = self.cur_state.cand_attr_values

        # Keep track of how many of each type of attr have been interviewed
        for i, attr in enumerate(attr_vals):
            self.attr_counts[i][attr] += 1

        if action == 1:
            # Keep track of how many of each type of attr have been hired
            for i, attr in enumerate(attr_vals):
                self.attr_hire_counts[i][attr] += 1
        elif action == 0:
            pass
        else:
            raise ValueError('Invalid action ', action)

        return should_reset

    def _get_reward(self, action):
        """
        Returns the reward based on the current state and action.

        Parameters
        ----------
        action : int
            Action to take.

        Returns
        -------
        float
            Reward.
        """
        HIRE_COST = 5  # TODO 7/29/20 - Determine significance of this value
        r = action*(self.cur_state.cand_prod_score - HIRE_COST)
        return r

    def _get_obs(self):
        """
        Obtain the observation for the current state of the environment. This
        is a fully observable environment, so we can return the state directly.

        Returns
        -------
        list
            <object>
        """
        return (self.cur_state.cand_attr_values.tolist()
                + [[self.cur_state.cand_prod_score]])

    def _step_reset(self):
        """
        Performs resets that happen after each timestep.

        Returns
        -------
        None
        """
        # Reset candidate
        new_cand_attr_vals = np.zeros(len(self.attr_probs), dtype=int)

        for i, attr_prob in enumerate(self.attr_probs):
            new_cand_attr_vals[i] = np.random.choice(len(attr_prob),
                                                     p=attr_prob)

        self.cur_state.cand_attr_values = new_cand_attr_vals
        self.cur_state.cand_attr_names = self.attr_names
        self.cur_state.cand_prod_score = self.prod_score_fn(new_cand_attr_vals)
