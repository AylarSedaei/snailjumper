import copy
import random
import numpy as np
from matplotlib import pyplot as plt

from player import Player


class Evolution:
    def __init__(self):
        self.game_mode = "Neuroevolution"
        self.scores = {'bests': [], 'means': [], 'worsts': []}

    def next_population_selection(self, players, num_players):
        """
        Gets list of previous and current players (μ + λ) and returns num_players number of players based on their
        fitness value.

        :param players: list of players in the previous generation
        :param num_players: number of players that we return
        """
        # TODO Implement top-k algorithm here
        players.sort(key=lambda p: p.fitness, reverse=True)
        population = players[:num_players]

        # TODO (Additional: Implement roulette wheel here)
        # population = self.roulette_wheel(players, num_players)
        # TODO (Additional: Implement SUS here)
        # population = self.roulette_wheel(players, num_players)
        # TODO (Additional: Learning curve)
        all_fitness = [p.fitness for p in players]
        self.scores['bests'].append(max(all_fitness))
        self.scores['means'].append(np.mean(all_fitness))
        self.scores['worsts'].append(min(all_fitness))

        return population

    def generate_new_population(self, num_players, prev_players=None):
        """
        Gets survivors and returns a list containing num_players number of children.

        :param num_players: Length of returning list
        :param prev_players: List of survivors
        :return: A list of children
        """
        first_generation = prev_players is None
        if first_generation:
            return [Player(self.game_mode) for _ in range(num_players)]
        else:
            # TODO ( Parent selection and child generation )

            parents = self.q_tournament(prev_players, num_players)
            children = self.generate_children(parents, num_players)
            new_players = [self.mutate(child) for child in children]

            return new_players

    def clone_player(self, player):
        """
        Gets a player as an input and produces a clone of that player.
        """
        new_player = Player(self.game_mode)
        new_player.nn = copy.deepcopy(player.nn)
        new_player.fitness = player.fitness
        return new_player

    def q_tournament(self, players, num_players, q=10):
        selected = []
        for _ in range(num_players):
            batch = []
            for _ in range(q):
                batch.append(random.choice(players))

            batch.sort(key=lambda p: p.fitness, reverse=True)
            selected.append(self.clone_player(batch[0]))

        return selected

    def roulette_wheel(self, players, num_players):
        population_fitness = sum([player.fitness for player in players])
        probabilities = [player.fitness / population_fitness for player in players]
        selected = np.random.choice(players, size=num_players, p=probabilities, replace=False)
        return list(selected)

    def generate_children(self, parents, num_players):
        children = []
        r = num_players % 2
        for i in range(0, num_players - r, 2):
            p = random.uniform(0, 1)
            father = self.clone_player(parents[i])
            mother = self.clone_player(parents[i + 1])

            if p > 0.85:
                children.append(father)
                children.append(mother)

            else:
                first_child, second_child = self.cross_over(father, mother)
                children.append(first_child)
                children.append(second_child)

        if r == 1:
            children.append(parents[0])

        return children

    def cross_over(self, father: Player, mother: Player):

        def cross_matrix(m1, m2):
            c = int(m1.shape[0] / 2)
            return np.concatenate([m1[:c], m2[c:]])

        first_child = Player(self.game_mode)
        second_child = Player(self.game_mode)

        layers_num = father.nn.L
        for l in range(1, layers_num):
            fW = father.nn.parameters['W' + str(l)]
            fb = father.nn.parameters['b' + str(l)]
            mW = mother.nn.parameters['W' + str(l)]
            mb = mother.nn.parameters['b' + str(l)]

            first_child.nn.parameters['W' + str(l)] = cross_matrix(fW, mW)
            first_child.nn.parameters['b' + str(l)] = cross_matrix(fb, mb)
            second_child.nn.parameters['W' + str(l)] = cross_matrix(mW, fW)
            second_child.nn.parameters['b' + str(l)] = cross_matrix(mb, fb)

        return first_child, second_child

    def mutate(self, parent: Player):
        child = self.clone_player(parent)
        layers_num = child.nn.L

        for l in range(1, layers_num):
            W = child.nn.parameters['W' + str(l)]
            b = child.nn.parameters['b' + str(l)]

            p = random.uniform(0, 1)
            if p < 0.1:
                child.nn.parameters['W' + str(l)] += np.random.normal(0., 0.4, W.shape)
                child.nn.parameters['b' + str(l)] += np.random.normal(0., 0.4, b.shape)

            return child

    def plot_learning_curve(self):
        bests = self.scores['bests']
        means = self.scores['means']
        worsts = self.scores['worsts']
        generations = [i for i in range(len(bests))]

        plt.plot(generations, bests, color='green')
        plt.plot(generations, means, color='orange')
        plt.plot(generations, worsts, color='brown')

        plt.xlabel('Generations')
        plt.ylabel('Scores')
        plt.legend(['bests', 'means', 'worsts'])
        plt.title('Learning Curve')

        plt.show()
