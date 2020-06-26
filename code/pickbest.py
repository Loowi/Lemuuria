# Input dict
import numpy as np
opts = {'a2a3': 0.0003333, 'a2a4': 0.000117, 'b1a3': 0.0003617, 'b1c3': 0.05182,
        'b2b3': 0.004093,  'b2b4': 0.0005, 'c2c3': 0.002106,
        'c2c4': 0.09686, 'd2d3': 0.0005646, 'd2d4': 0.1523, 'e2e3': 0.02411, 'e2e4': 0.1313,
        'f2f3': 0.000964, 'f2f4': 0.01193, 'g1f3': 0.4478}

options = []
sum(opts.values())


class Move:
    # Hold infor required to pick the best option
    def __init__(self, move, prior_prob):
        self.move = move  # chess.Board(fen)
        self.prior_prob = prior_prob
        self.num_visits = 0
        self.total_value = 0
        self.mean_value = prior_prob
        self.adjust_value = self.mean_value + (1.5 * self.prior_prob *
                                               np.sqrt(self.num_visits + 1) / (1 + self.num_visits))

    def __repr__(self):
        s1 = str(self.prior_prob)
        s2 = str(self.num_visits)
        s3 = str(self.adjust_value)
        return 'move: '+self.move+', prior_prob: '+s1+', num_visits: '+s2+', adjust_value: '+s3+' | '

    def calcValue(self, new_value):
        self.num_visits += 1
        self.total_value = np.nansum(new_value + self.total_value)
        self.mean_value = self.total_value / self.num_visits

        add_on = (1.5 * self.prior_prob *
                  np.sqrt(self.num_visits + 1) / (1 + self.num_visits))
        self.adjust_value = self.mean_value + add_on


# Create nodes
for opt in opts.keys():
    prior_prob = opts[opt]
    node = Move(opt, prior_prob)
    options.append(node)

best_value = -np.inf

# Monte Carlo
for i in range(100):
    for option in options:
        if option.adjust_value > best_value:  # should be probability based
            best_move = option

    option.calcValue(0.3)


# Compare actual visits to prior probabilities
pass
