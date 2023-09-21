from dataclasses import dataclass
from typing import List, Tuple, Union

import numpy as np
from gym import spaces
from nlp_gym.data_pools.question_answering_pool import Sample, ConvSearchSample
from nlp_gym.envs.common.base_env import BaseEnv
from nlp_gym.envs.common.action_space import ActionSpace
from nlp_gym.envs.common.reward import RewardFunction
from nlp_gym.envs.question_answering.featurizer import InformedFeaturizer
from nlp_gym.envs.question_answering.observation import (Observation,
                                                         ObservationFeaturizer,
                                                         ConvSearchObservation)
from nlp_gym.envs.question_answering.reward import BinaryRewardFunction
from rich import print


@dataclass(init=True)
class ObsTriplet:
    question: str
    facts: List[str]
    choice_id: str
    choice_text: str


class QAEnv(BaseEnv):
    """
    An environment for question answering with multiple choices and supporting facts

    Observation consists of triple of Question, Facts, Choice
    Actions are binary. Either to ANSWER OR CONTINUE

    ANSWER corresponds to answering with the choice in the observation
    CONTINUE correponds to agent asking for the next choice

    """
    def __init__(self, observation_featurizer: ObservationFeaturizer = None, reward_function: RewardFunction = None,
                 return_obs_as_vector: bool = True):
        # set action spaces
        self.action_space = ActionSpace(actions=["ANSWER", "CONTINUE"])

        # set observation spaces
        if return_obs_as_vector:
            observation_featurizer = InformedFeaturizer() if observation_featurizer is None else observation_featurizer
        else:
            observation_featurizer = None

        reward_function = BinaryRewardFunction() if reward_function is None else reward_function
        super().__init__(None, reward_function, observation_featurizer, return_obs_as_vector)

        # set the counter
        self.time_step = None

        # observation time line
        self.__observation_sequence = None
        self.__current_target = None
        self.__current_observation = None

        # hold samples
        self.__samples: List[Sample] = []

    def _is_terminal(self, action: str, time_step: int):
        termimal = (action == "ANSWER") or (time_step == len(self.__observation_sequence) - 1)
        return termimal

    def step(self, action: int) -> Tuple[Union[Observation, np.array], int, bool, dict]:

        # current action
        action_str = self.action_space.ix_to_action(action)

        # get reward
        reward = self.reward_function(self.__current_observation, action_str, self.__current_target)

        # terminal or not
        done = self._is_terminal(action_str, self.time_step)

        # if not done
        if not done:
            self.time_step += 1
            info = {}
        else:
            # populate the info field
            info = {"selected_choice": self.__observation_sequence[self.time_step].choice_id}

        # current observation
        observation_at_t = self.__observation_sequence[self.time_step]
        observation = Observation.build(observation_at_t.question, observation_at_t.facts,
                                        observation_at_t.choice_text, observation_at_t.choice_id,
                                        self.time_step, len(self.__observation_sequence),
                                        self.observation_featurizer,
                                        self.return_obs_as_vector)
        self.__current_observation = observation
        observation_to_return = observation.get_vector().numpy() if self.return_obs_as_vector else observation
        return observation_to_return, reward, done, info

    def reset(self, sample: Sample = None) -> Union[Observation, np.array]:
        # get a QA sample
        if sample is None:
            sample = np.random.choice(self.__samples)

        # init on reset
        if self.observation_featurizer is not None:
            self.observation_featurizer.init_on_reset(sample.question, sample.facts)

        # create the observation sequence
        self.__observation_sequence = QAEnv._create_sequence(sample)

        # set the current target
        self.__current_target = sample.answer

        # time step
        self.time_step = 0

        # current observation
        observation_at_t = self.__observation_sequence[self.time_step]
        observation = Observation.build(observation_at_t.question, observation_at_t.facts,
                                        observation_at_t.choice_text, observation_at_t.choice_id,
                                        self.time_step, len(self.__observation_sequence),
                                        self.observation_featurizer, 
                                        self.return_obs_as_vector)
        self.__current_observation = observation
        observation_to_return = observation.get_vector().numpy() if self.return_obs_as_vector else observation
        return observation_to_return

    @staticmethod
    def _create_sequence(sample: Sample) -> List[ObsTriplet]:
        sequences = []
        for choice_id, choice in sample.choices.items():
            triplet = ObsTriplet(sample.question, sample.facts, choice_id, choice)
            sequences.append(triplet)
        return sequences

    def render(self):
        print(f"[italic yellow]Step {self.time_step}[/italic yellow]")
        print(f"[italic red]Question[/italic red]: {self.__observation_sequence[0].question}")
        for obs in self.__observation_sequence[:self.time_step+1]:
            for fact in obs.facts:
                print(f"[italic red]Fact[/italic red]: {fact}")
            print(f"[italic red]Choice[/italic red] {obs.choice_id}: {obs.choice_text}")

    def close(self):
        pass

    # Methods for online learning and sampling

    def add_sample(self, sample: Sample, weight: int = 1.0):
        self.__samples.append(sample)

    def get_samples(self) -> List[Sample]:
        return self.__samples


class ConvSearchObsTriplet:
    query: str
    results: List[str]
    questions: List[str]
    results_scores: List[float]
    questions_scores: List[float]
    results_mrr: float
    relevant_question_rank: int


class ConvSearchEnv(BaseEnv):
    """
    An environment for question answering with multiple choices and supporting facts

    Observation consists of triple of Question, Facts, Choice
    Actions are binary. Either to ANSWER OR CONTINUE

    ANSWER corresponds to answering with the choice in the observation
    CONTINUE correponds to agent asking for the next choice

    """
    def __init__(self, observation_featurizer: ObservationFeaturizer = None, reward_function: RewardFunction = None,
                 return_obs_as_vector: bool = True):
        # set action spaces
        self.action_space = ActionSpace(actions=["ANSWER", "ASK"])

        # set observation spaces
        if return_obs_as_vector:
            observation_featurizer = InformedFeaturizer() if observation_featurizer is None else observation_featurizer
        else:
            observation_featurizer = None

        reward_function = BinaryRewardFunction() if reward_function is None else reward_function
        super().__init__(None, reward_function, observation_featurizer, return_obs_as_vector)

        # set the counter
        self.time_step = None

        # observation time line
        self.__observation_sequence = None
        self.__current_observation = None

        self.max_question_rank = 99

        # hold samples
        self.__samples: List[ConvSearchSample] = []

    def _is_terminal(self, action: str):
        termimal = (action == "ANSWER") or (action == "ASK" and self.__current_observation.get_relevant_question_rank() > self.max_question_rank)
        return termimal

    def step(self, action: int) -> Tuple[Union[Observation, np.array], int, bool, dict]:

        # current action
        action_str = self.action_space.ix_to_action(action)

        # get reward
        reward = self.reward_function(self.__current_observation, action_str)

        # terminal or not
        done = self._is_terminal(action_str)

        # if not done
        if not done:
            self.time_step += 1
            info = {}
        else:
            # populate the info field
            info = {"results_mrr": self.__observation_sequence[self.time_step].results_mrr, 
                    "relevant_question_rank": self.__observation_sequence[self.time_step].relevant_question_rank}

        # current observation
        observation_at_t = self.__observation_sequence[self.time_step]
        observation = ConvSearchObservation.build(observation_at_t.query,
                                        observation_at_t.results,
                                        observation_at_t.questions,
                                        observation_at_t.results_scores,
                                        observation_at_t.questions_scores,
                                        observation_at_t.results_mrr, 
                                        observation_at_t.relevant_question_rank,
                                        self.time_step, len(self.__observation_sequence),
                                        self.observation_featurizer,
                                        self.return_obs_as_vector)
        self.__current_observation = observation
        observation_to_return = observation.get_vector().numpy() if self.return_obs_as_vector else observation
        return observation_to_return, reward, done, info

    def reset(self, sample: ConvSearchSample = None) -> Union[Observation, np.array]:
        # get a convsearch sample
        if sample is None:
            sample = np.random.choice(self.__samples)

        # init on reset
        if self.observation_featurizer is not None:
            self.observation_featurizer.init_on_reset(sample.results, sample.questions, sample.results_scores, sample.questions_scores)

        # create the observation sequence
        self.__observation_sequence = ConvSearchEnv._create_sequence(sample)

        # time step
        self.time_step = 0

        # current observation
        observation_at_t = self.__observation_sequence[self.time_step]
        observation = ConvSearchObservation.build(observation_at_t.query,
                                        observation_at_t.results,
                                        observation_at_t.questions,
                                        observation_at_t.results_scores,
                                        observation_at_t.questions_scores,
                                        observation_at_t.results_mrr, 
                                        observation_at_t.relevant_question_rank,
                                        self.time_step, len(self.__observation_sequence),
                                        self.observation_featurizer, 
                                        self.return_obs_as_vector)
        self.__current_observation = observation
        observation_to_return = observation.get_vector().numpy() if self.return_obs_as_vector else observation
        return observation_to_return
    


    @staticmethod
    def _create_sequence(sample: Sample) -> List[ConvSearchObsTriplet]:
        '''
        sample format:
        each sample should be a tuple with a query field and a turns field which is a list of turn.
        each turn should be a tuple with a results field, a questions field, a results_mrr field, and a relevant_question_rank field.
        results and questions are lists of result(s) and question(s), respectively
        results_mrr should be one float number that equals the reciprocal of relevant result
        relevant_question_rank should be a int number that is the rank of relevant clarifying question
        '''
        sequences = []
        for turn in sample['turns']:
            # each turn should be a tuple
            triplet = ConvSearchObsTriplet(sample['query'], turn['answers'], turn['questions'], turn['answers_scores'], turn['questions_scores'], turn['results_mrr'], turn['relevant_question_rank'])
            sequences.append(triplet)
        return sequences

    def render(self):
        print(f"[italic yellow]Step {self.time_step}[/italic yellow]")
        print(f"[italic red]Query[/italic red]: {self.__observation_sequence[0].query}")
        for obs in self.__observation_sequence[:self.time_step+1]:
            for result in obs.results:
                print(f"[italic red]Result[/italic red]: {result}")
            for question in obs.questions:
                print(f"[italic red]Question[/italic red]: {question}")
            print(f"[italic red]Results_MRR[/italic red] {obs.results_mrr}")
            print(f"[italic red]Relevant_question_rank[/italic red] {obs.relevant_question_rank}")

    def close(self):
        pass

    # Methods for online learning and sampling

    def add_sample(self, sample: Sample, weight: int = 1.0):
        self.__samples.append(sample)

    def get_samples(self) -> List[Sample]:
        return self.__samples
    
    def generate_expert_traj(self, save_path, alpha, n_episodes):
        '''
        '''
        def find_best_traj(answer_rewards, question_ranks, alpha):
            '''
            Find the best conversation trajectory given all the answer rank and question rank.
            answer_traj is a list of ((state, action), reward) tuples.
            question_traj is a list of ((state, action), correct question rank) tuples.
            '''
            best_ecrr, best_step = 0, 0
            cumulative_ecrr = 1
            assert len(answer_rewards) == len(question_ranks)
            for step in range(len(answer_rewards)):
                if cumulative_ecrr * answer_rewards[step] > best_ecrr:
                    best_ecrr = cumulative_ecrr * answer_rewards[step]
                    best_step = step
                cumulative_ecrr *= alpha ** question_ranks[step]
            return best_step, best_ecrr
        

        actions = []
        obs = []
        rewards = []
        returns = []
        episode_starts = []

        for _ in n_episodes:
            sample = np.random.choice(self.__samples)

            answer_rewards = [turn['results_mrr'] for turn in sample['turns']]
            question_ranks = [turn['relevant_question_rank'] for turn in sample['turns']]
            best_result_turn, best_ecrr = find_best_traj(answer_rewards, question_ranks, alpha)

            actions += ["ASK"] * best_result_turn + ["ANSWER"]
            obs += [
                ConvSearchObservation.build(sample['turns'][step].query,
                                            sample['turns'][step].results,
                                            sample['turns'][step].questions,
                                            sample['turns'][step].results_scores,
                                            sample['turns'][step].questions_scores,
                                            sample['turns'][step].results_mrr, 
                                            sample['turns'][step].relevant_question_rank,
                                            step, len(sample['turns']),
                                            self.observation_featurizer, 
                                            self.return_obs_as_vector).get_vector().numpy() if self.return_obs_as_vector else observation
                for step in range(best_result_turn + 1)
            ]

            returns += [
                (step + 1)/(best_result_turn + 1) * best_ecrr
                for step in range(best_result_turn + 1)
            ]

            rewards += [ 
                returns[step] - returns[step-1] if step > 0 else returns[step]
                for step in range(best_result_turn + 1)
            ]

            episode_starts += [True] + [False] * best_result_turn
        
        assert len(actions) == len(obs)
        assert len(obs) == len(rewards)
        assert len(rewards) == len(returns)
        assert len(returns) == len(episode_starts)

        numpy_dict = {
            'actions': actions,
            'obs': obs,
            'rewards': rewards,
            'episode_returns': returns,
            'episode_starts': episode_starts
        }

        if save_path is not None:
            np.savez(save_path, **numpy_dict)

        return numpy_dict
