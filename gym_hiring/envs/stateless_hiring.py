# Core modules
import logging.config

# 3rd party modules
import gym
import numpy as np
from gym.spaces import Discrete, Tuple


class StatelessHiring(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, max_steps_per_episode):
        """
        A stateless hiring environment.

        Parameters
        ----------
        max_steps_per_episode : int
            Maximum allowed steps per episode. This will define how long an
            episode lasts, since the environment does not end otherwise.

        Attributes
        ----------
        cur_episode : int
            Current episode as a count.
        action_episode_memory : list<<list<int>>
            History of actions taken in each episode.
        observation_episode_memory : list<list<int>>
            History of observations observed in each episode.
        reward_episode_memory : list<list<int, float>>
            History of rewards observed in each episode.
        cur_step : int
            Current timestep in episode, as a count.
        action_space : gym.spaces.Discrete
            Action space.
        observation_space : gym.spaces.Discrete
            Observation space.
        n_states : int
            Number of possible states.
        n_actions : int
            Number of possible actions.
        observation_function : list<list>
            Index represents state, value is observations for that state.
        transition_probabilities : np.ndarray (n_states, n_actions, n_states)
            3-dim matrix where index [s][a][s'] represents the probability of
            transitioning from state s to state s' after taking action a.
        cur_state : int
            Current state.

        Examples
        --------
        >>> args = {'max_steps_per_episode': 100}
        >>> env = gym.make('StatelessHiring-v0', **args)
        """
        self.__version__ = "0.0.1"
        logging.info("StatelessHiring Version {}".format(self.__version__))

        self.max_steps_per_episode = max_steps_per_episode

        self.cur_episode = -1  # Set to -1 b/c reset() adds 1 to episode
        self.action_episode_memory = []
        self.observation_episode_memory = []
        self.reward_episode_memory = []
        self.cur_step = 0

        # Define what the agent can do: HIRE, REJECT
        self.action_space = Discrete(2)

        # Define what agent can observe.
        self.observation_space = Tuple((
            Discrete(2),  # Gender : Male or Female
            Discrete(5),  # Productivity Score : integer from 0 to 4
        ))

        self.n_states = np.prod([space.n for space in self.observation_space])
        self.n_actions = self.action_space.n

        # Define how observations are generated based on the state.
        self.observation_function = [
            [0, 0],  # state=0 => gender=F, prod_score=0,
            [0, 1],  # state=1 => gender=F, prod_score=1,
            [0, 2],  # state=2 => gender=F, prod_score=2,
            [0, 3],  # state=3 => gender=F, prod_score=3,
            [0, 4],  # state=4 => gender=F, prod_score=4,
            [1, 0],  # state=5 => gender=M, prod_score=0,
            [1, 1],  # state=6 => gender=M, prod_score=1,
            [1, 2],  # state=7 => gender=M, prod_score=2,
            [1, 3],  # state=8 => gender=M, prod_score=3,
            [1, 4],  # state=9 => gender=M, prod_score=4,
        ]

        # Define how new state is determined from previous state and action
        self.transition_probabilities = np.array([
            [[self._transition_probability(i, j, k)
              for k in range(self.n_states)]
             for j in range(self.n_actions)]
            for i in range(self.n_states)])

        # Reset
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
        done = self.cur_step >= self.max_steps_per_episode

        if done:
            raise RuntimeError("Episode is done")

        self.cur_step += 1

        # Compute new state based on previous state and action
        new_state = self._take_action(action)

        # Compute reward value based on previous state and action
        reward = self._get_reward(action)

        # Update current state to new state
        self.cur_state = new_state

        # Compute observation from current state
        ob = self._get_obs()  # Has to come after new state update

        # Update action, observation and reward histories
        self.action_episode_memory[self.cur_episode].append(action)
        self.observation_episode_memory[self.cur_episode].append(ob)
        self.reward_episode_memory[self.cur_episode].append(reward)

        # Recompute done since action may have modified it
        done = self.cur_step >= self.max_steps_per_episode

        return ob, reward, done, {}

    def reset(self):
        """
        Reset the state of the environment and returns an initial observation.

        Returns
        -------
        object
            The initial observation of the space.
        """
        self.cur_step = 0
        self.cur_episode += 1
        self.cur_state = 0  # Always start with fixed state
        self.action_episode_memory.append([])
        self.observation_episode_memory.append([])
        self.reward_episode_memory.append([])
        obs = self._get_obs()
        return obs

    def render(self, mode='human'):
        return

    def close(self):
        pass

    def _take_action(self, action):
        """
        How to change the environment when taking an action.

        Parameters
        ----------
        action : int
            Action.

        Returns
        -------
        int
            New state after taking action.
        """
        # Get transition probabilities for all potential next state values
        trans_probs = self.transition_probabilities[self.cur_state, action]

        # Generate an array of next state options to choose from
        next_state_options = np.linspace(0, self.n_states-1, self.n_states,
                                         dtype=int)

        # Sample from new state options based on the transition probabilities
        new_state = np.random.choice(next_state_options, p=trans_probs)

        return new_state

    def _transition_probability(self, s, a, s1):
        """
        Returns the probability of transitioning into state s1 after taking
        action a in state s.

        Parameters
        ----------
        s : int
            Initial state index.
        a : int
            Action index.
        s1 : int
            New state index.

        Returns
        -------
        float, range [0, 1]
            Transition probability.
        """
        unreachable_states = [4,  # F with prod_score == 4
                              5]  # M with prod_score == 0

        if s1 in unreachable_states:
            return 0
        else:
            return 1 / (self.n_states - len(unreachable_states))

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
        HIRE_COST = 1  # TODO 7/29/20 - Determine significance of this value

        # Lookup the state representation using the cur_state index. Then we
        # can get the candidate productivity score.
        obs = self.observation_function[self.cur_state]
        prod_score = obs[1]
        r = action*(prod_score - HIRE_COST)
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
        return self.observation_function[self.cur_state]
