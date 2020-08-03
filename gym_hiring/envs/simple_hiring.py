
# Core modules
import logging.config

# 3rd party modules
import gym
import numpy as np
from gym.spaces import Discrete, Tuple


class SimpleHiring(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, max_steps_per_episode, verbose=False):
        """
        A simple hiring environment.

        Parameters
        ----------
        max_steps_per_episode : int
            Maximum allowed steps per episode. This will define how long an
            episode lasts, since the environment does not end otherwise.
        verbose : bool, default False
            If True, log current state on each timestep.

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
        features_by_state_index : np.ndarray, shape(n_states, 6)
            Index represents state, value is observations for that state.
        state_index_by_features : list<list>
            Feature values lookup by state index.
        features_by_state_index : np.ndarray, shape(2, 3, 3, 4, 3, 4)
            State index by features lookup.
        transition_probabilities : np.ndarray (n_states, n_actions, n_states)
            3-dim matrix where index [s][a][s'] represents the probability of
            transitioning from state s to state s' after taking action a.
        cur_state : int
            Current state.

        Examples
        --------
        >>> args = {'max_steps_per_episode': 100}
        >>> env = gym.make('SimpleHiring-v0', **args)
        """
        self.__version__ = "0.0.1"
        logging.info("SimpleHiring - Version {}".format(self.__version__))

        self.max_steps_per_episode = max_steps_per_episode
        self.verbose = verbose

        self.cur_episode = -1  # Set to -1 b/c reset() adds 1 to episode
        self.action_episode_memory = []
        self.observation_episode_memory = []
        self.reward_episode_memory = []
        self.cur_step = 0

        # Define what the agent can do: HIRE, REJECT
        self.action_space = Discrete(2)

        self.n_cand_genders = 2
        self.n_cand_prod_scores = 3
        self.n_emp_genders = 3
        self.n_emp_prod_scores = 4
        self.n_employees = 2

        # Define what agent can observe.
        self.observation_space = Tuple((
            Tuple((
                Discrete(self.n_cand_genders),  # Candidate Gender
                Discrete(self.n_cand_prod_scores),  # Candidate Prod. Score
            )),
            Tuple((
                Discrete(self.n_emp_genders),  # Employee 1 Gender
                Discrete(self.n_emp_prod_scores),  # Employee 1 Prod. Score
            )),
            Tuple((
                Discrete(self.n_emp_genders),  # Employee 2 Gender
                Discrete(self.n_emp_prod_scores),  # Employee 2 Prod. Score
            )),
        ))

        self.n_states = (self.n_cand_genders * self.n_cand_prod_scores *
                         ((self.n_emp_genders * self.n_emp_prod_scores)
                          ** self.n_employees))
        self.n_actions = self.action_space.n

        # Define how observations are generated based on the state.
        self.features_by_state_index, self.state_index_by_features = (
            self._compute_state_index_lookups())

        # Define how new state is determined from previous state and action
        self.transition_probabilities = np.array([
            [[self._transition_probability(i, j, k)
              for k in range(self.n_states)]
             for j in range(self.n_actions)]
            for i in range(self.n_states)])
        # Normalize values
        self.transition_probabilities /= (
            self.transition_probabilities.sum(axis=2, keepdims=True))

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

        # Compute reward value based on new state
        reward = self._get_reward(new_state)

        # Update current state to new state
        self.cur_state = new_state

        # Compute observation from current state
        ob = self._get_obs(new_state)

        # Update action, observation and reward histories
        self.action_episode_memory[self.cur_episode].append(action)
        self.observation_episode_memory[self.cur_episode].append(ob)
        self.reward_episode_memory[self.cur_episode].append(reward)

        # Recompute done since action may have modified it
        done = self.cur_step >= self.max_steps_per_episode

        if self.verbose:
            logging.info('\t' + self.render_state(self.cur_state))

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
        # Always start with female candidate with score=1, no existing emps.
        self.cur_state = 287
        self.action_episode_memory.append([])
        self.observation_episode_memory.append([])
        self.reward_episode_memory.append([])
        obs = self._get_obs(self.cur_state)

        if self.verbose:
            logging.info(f'Episode {self.cur_episode}')

        if self.verbose:
            logging.info('\t' + self.render_state(self.cur_state))

        return obs

    def render(self, mode='human'):
        return

    def close(self):
        pass

    def render_state(self, s):
        """
        String representation of a particular state.

        Parameters
        ----------
        s : int
            State index.

        Returns
        -------
        str
            String representation in human readable format.
        """
        s_ = self._state_feature_values(s)

        TRANSLATE = {
            'cand_gender': ['F', 'M'],
            'cand_prod_score': ['0', '1', '2'],
            'emp1_gender': ['F', 'M', '-'],
            'emp1_prod_score': ['0', '1', '2', '-'],
            'emp2_gender': ['F', 'M', '-'],
            'emp2_prod_score': ['0', '1', '2', '-'],
        }

        reps = [TRANSLATE[f][s_[f]] for f in TRANSLATE.keys()]
        rep = ', '.join(reps)

        return rep

    def _compute_state_index_lookups(self):
        """
        Assigns an integer to each possible state. The
        `features_by_state_index` attribute will provide the mapping of
        state index to state values. Similarly, the `state_index_by_features`
        attribute will provide the reverse lookup.

        Returns
        -------
        tuple (2)
            np.ndarray, shape(n_states, 6)
                Feature values lookup by state index.
            np.ndarray, shape(2, 3, 3, 4, 3, 4)
                State index by features lookup.

        """
        n_cg = self.n_cand_genders
        n_cps = self.n_cand_prod_scores
        n_eg = self.n_emp_genders
        n_eps = self.n_emp_prod_scores

        features_by_state_index = np.zeros((self.n_states, 6), dtype=int)
        state_index_by_features = np.zeros((n_cg, n_cps, n_eg, n_eps, n_eg,
                                            n_eps), dtype=int)

        self.n_cand_genders = 2
        self.n_cand_prod_scores = 3
        self.n_emp_genders = 3
        self.n_emp_prod_scores = 4
        self.n_employees = 2

        # Iterate over candidate genders
        for cg_idx, cg in enumerate(range(0, n_cg)):

            # Iterate over candidate productivity scores
            for cps_idx, cps in enumerate(range(0, n_cps)):

                # Iterate over emp1 genders
                for e1g_idx, e1g in enumerate(range(0, n_eg)):

                    # Iterate over emp1 productivity scores
                    for e1ps_idx, e1ps in enumerate(range(0, n_eps)):

                        # Iterate over emp2 genders
                        for e2g_idx, e2g in enumerate(range(0, n_eg)):

                            # Iterate over emp2 productivity scores
                            for e2ps_idx, e2ps in enumerate(range(0, n_eps)):

                                idx = ((n_cps*n_eg*n_eps*n_eg*n_eps*cg_idx) +
                                       (n_eg*n_eps*n_eg*n_eps*cps_idx) +
                                       (n_eps*n_eg*n_eps*e1g_idx) +
                                       (n_eg*n_eps*e1ps_idx) +
                                       (n_eps*e2g_idx) +
                                       e2ps_idx)

                                features_by_state_index[idx][0] = cg
                                features_by_state_index[idx][1] = cps
                                features_by_state_index[idx][2] = e1g
                                features_by_state_index[idx][3] = e1ps
                                features_by_state_index[idx][4] = e2g
                                features_by_state_index[idx][5] = e2ps

                                state_index_by_features[
                                    cg_idx][cps_idx][e1g_idx][e1ps_idx][
                                        e2g_idx][e2ps_idx] = idx

        return features_by_state_index, state_index_by_features

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
        Returns the UNNORMALIZED probability of transitioning into state s1
        after taking action a in state s.

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
        s_ = self._state_feature_values(s)
        s1_ = self._state_feature_values(s1)

        # New candidate generation is independent from the rest of the
        # environment. Candidates have equal probability of Gender=M and G=F.
        p_cg = 1 / self.n_cand_genders
        p_cps = 1 / self.n_cand_prod_scores

        # Assume emp1 and emp2 remain constant unless otherwise specified.
        p_e1g = int(s_['emp1_gender'] == s1_['emp1_gender'])
        p_e1ps = int(s_['emp1_prod_score'] == s1_['emp1_prod_score'])
        p_e2g = int(s_['emp2_gender'] == s1_['emp2_gender'])
        p_e2ps = int(s_['emp2_prod_score'] == s1_['emp2_prod_score'])

        # Female candidates cannot have prod score == 2
        if s1_['cand_gender'] == 0 and s1_['cand_prod_score'] == 2:
            return 0  # p_cps = 0

        # Male candidates cannot have prod score == 0
        if s1_['cand_gender'] == 1 and s1_['cand_prod_score'] == 0:
            return 0  # p_cps = 0

        # Emp1 and Emp2 quits with fixed probability of 10%
        emp1_quit = np.random.rand() < .1
        emp2_quit = np.random.rand() < .1

        if emp1_quit:
            p_e1g = int(s1_['emp1_gender'] == 2)  # emp.gender=2 => N/A
            p_e1ps = int(s1_['emp1_prod_score'] == 3)  # emp.ps=3 => N/A

            if p_e1g == 0 or p_e1ps == 0:
                return 0

        if emp2_quit:
            p_e2g = int(s1_['emp2_gender'] == 2)  # emp.gender=2 => N/A
            p_e2ps = int(s1_['emp2_prod_score'] == 3)  # emp.ps=3 => N/A

            if p_e2g == 0 or p_e2ps == 0:
                return 0

        # If hire=1
        if a == 1:
            # If emp1 is empty then emp1 -> candidate
            if s_['emp1_gender'] == 2:  # emp gender=2 represents NA
                p_e1g = int(s_['cand_gender'] == s1_['emp1_gender'])
                p_e1ps = int(s_['cand_prod_score'] == s1_['emp1_prod_score'])
            # Else If emp2 is empty then emp2 -> candidate
            elif s_['emp2_gender'] == 2:  # emp gender=2 represents NA
                p_e2g = int(s_['cand_gender'] == s1_['emp2_gender'])
                p_e2ps = int(s_['cand_prod_score'] == s1_['emp2_prod_score'])

        return p_cg * p_cps * p_e1g * p_e1ps * p_e2g * p_e2ps

    def _state_feature_values(self, s):
        """
        Maps state index to its feature values.

        Parameters
        ----------
        s : int
            State index.

        Returns
        ----------
        Dict<str, int>
            Keys are state feature names, values are state feature values.
        """
        if self.features_by_state_index is None:
            raise Exception('features_by_state_index not yet computed.')

        return {
            'cand_gender': self.features_by_state_index[s][0],
            'cand_prod_score': self.features_by_state_index[s][1],
            'emp1_gender': self.features_by_state_index[s][2],
            'emp1_prod_score': self.features_by_state_index[s][3],
            'emp2_gender': self.features_by_state_index[s][4],
            'emp2_prod_score': self.features_by_state_index[s][5],
        }

    def _get_reward(self, s):
        """
        Returns the reward based on state.

        Parameters
        ----------
        s : int
            State index.

        Returns
        -------
        float
            Reward.
        """
        # emp1 or emp2 prod_score=3 represents NA
        s_ = self._state_feature_values(s)
        emp1_ps = s_['emp1_prod_score']
        emp2_ps = s_['emp2_prod_score']

        if emp1_ps == 3:
            emp1_ps = 0

        if emp2_ps == 3:
            emp2_ps = 0

        r = emp1_ps + emp2_ps

        return r

    def _get_obs(self, s):
        """
        Obtain the observation for the current state of the environment. This
        is a fully observable environment, so we can return the state directly.

        Parameters
        ----------
        s : int
            State index.

        Returns
        -------
        list
            <object>
        """
        return self.features_by_state_index[s]
