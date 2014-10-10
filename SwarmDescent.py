'''
Swarm Descent functionality

Dan Morris, 10/9/14 -

The intuition behind swarm descent is to provide an alternative to gradient
  descent, particularly in complex parameter spaces with potentially many local
  minima. Given a training set X, y and a cost function J, we randomly
  initialize many 'bees', each of which has one primary function: To take a
  step in the direction of the bee with lowest current cost (the 'queen').
  Additionally, each bee has a small percentage chance of 'straying' in a
  random direction at each step, thus providing additional robustness to
  local minima. The swarm will continue to take steps until some stopping
  criteria is met; A fixed total number of steps or a specified number of steps
  for which the queen remains the same.

TODOs:
- Test on simple datasets/cost functions
-   Evaluate timing/efficacy with various knob settings
- Implement 'dynasty' stopping condition: same bee is best bee for n
  consecutive steps
- figure out that algorithm to build the 'spread' formation. check out that
  cool algorithm visualization page I shared to piazza.
- Stopping condition: some number of bees converge into an epsilon-small
  neighborhood around the best bee? Might not be better than what we have
  already.
'''
import numpy as np


class Swarm(object):
    '''
    cost_function (callable): any function which takes (X, y, params) as its
        inputs. We seek the parameters which minimize its value.
    param_shape (tuple: int x int): the shape of the parameter array
    size (int): the number of bees in the swarm
    formation (str "random" or "spread"): the initial formation of the bees
    spread_dist (float): If formation == "random", the factor by which the
        random initial paramters are multiplied. If formation == "spread", the
        minimum distance between each bee.
    learning_rate (float in (0, 1)): the fraction of the distance between a bee
        and the best bee which it will travel in one step
    stray_rate (float in (0, 1)): the percentage of steps taken in a random
        direction rather than towards the best bee
    stop_criteria (str "iters", "dynasty", or "fast"): the stopping criteria
    stop_iters (int): the number of steps to take before stopping
    n_workers (int): number of 'workers' that the queen spawns each step to
        check her immediate neighborhood for a better cost
    worker_eps (float): the distance her workers travel from the queen
    reign_stop (int): stop iterating if the queen doesn't change for this many
        iterations
    '''
    def __init__(self, cost_function, param_shape, size=100,
                 formation="random", spread_dist=20.0, learning_rate=.2,
                 stray_rate=.2, stop_criteria="fast", stop_iters=1000,
                 n_workers=3, worker_eps=.01, reign_stop=20):
        self.J = cost_function
        self.param_shape = param_shape
        self.spread_dist = spread_dist
        self.size = size
        self.formation = formation
        self.learning_rate = learning_rate
        self.stray_rate = stray_rate
        self.stop_criteria = stop_criteria
        self.stop_iters = stop_iters
        self.n_workers = n_workers
        self.worker_eps = worker_eps
        self.reign_stop = reign_stop
        self.queen_reign = 0
        self.bees = self.spawn()
        self.queen = self.bees[0]

    def spawn(self):
        '''
        Initializes the swarm based on the input parameters.
        '''
        s = []
        if self.formation == 'random':
            a = self.param_shape[0]
            b = self.param_shape[1]
            for i in range(self.size):
                s.append(Bee((np.random.rand(a, b) - .5) *
                             self.spread_dist, self))
        elif self.formation == 'spread':
            # TODO
            pass
        else:
            # future formations?
            print 'bad formation string'
            pass
        return s

    def fit(self, X, y, verbose=1):
        '''
        Uses the swarm to optimize parameters given training set X, y
        verbose 0: no print reporting
        verbose 1: periodic updates on iterations and current best cost
        verbose 2: frequent updates on iterations and current best cost
        verbose 3: constant updates on iterations and current best cost
        '''
        start_cost = self.find_queen(X, y)
        if verbose > 0:
            print 'Initial best cost: ', start_cost
        if self.stop_criteria == "iters":
            for i in range(self.stop_iters):
                cur_best_cost = self.swarm_step(X, y)
                # report status
                if verbose > 0 and (i+1) % (1000 / (10 ** verbose)) == 0:
                    self.report_status(X, y, i, cur_best_cost)

        elif self.stop_criteria == "dynasty":
            i = 0
            while self.queen_reign < self.reign_stop:
                cur_best_cost = self.swarm_step(X, y)
                if verbose > 0 and (i+1) % (1000 / (10 ** verbose)) == 0:
                    self.report_status(X, y, i, cur_best_cost)
                i += 1

        elif self.stop_criteria == "fast":
            # Combination of iters and dynasty, stops when either is met.
            for i in range(self.stop_iters):
                cur_best_cost = self.swarm_step(X, y)
                if self.queen_reign > self.reign_stop:
                    break
                if verbose > 0 and (i+1) % (1000 / (10 ** verbose)) == 0:
                    self.report_status(X, y, i, cur_best_cost)

        else:
            # future stop_criteria?
            print 'bad stop_criteria string'
            pass

        # whatever fit-ending verbosity
        if verbose > 0:
            print 'Final cost: ', self.J(X, y, self.queen.params)
            print 'Returning best parameters...'
        return self.queen.params

    def swarm_step(self, X, y):
        '''
        One swarm step, regardless of stopping condition
        '''
        # queen epsilon check
        self.queen_epsilon_check(X, y)
        # step all bees
        for b in self.bees:
            b.step(self.queen)
        # find + set queen
        return self.find_queen(X, y)

    def find_queen(self, X, y):
        '''
        Finds the best bee, sets it to self.queen, and returns its current cost
        '''
        best_cost = self.J(X, y, self.queen.params)
        for b in self.bees:
            j = self.J(X, y, b.params)
            if j < best_cost:
                self.queen = b
                self.queen_reign = 0
                best_cost = j
        self.queen_reign += 1
        return best_cost

    def mean_swarm_cost(self, X, y):
        '''
        Returns the average current cost of all bees in the swarm
        '''
        c = np.zeros(self.size)
        for i, b in enumerate(self.bees):
            c[i] = self.J(X, y, b.params)
        return np.mean(c)

    def report_status(self, X, y, i, cur_best_cost):
        print 'Iteration ' + str(i+1) + '. Current best cost: ' + \
              str(cur_best_cost)
        print '  Mean Swarm cost: ', self.mean_swarm_cost(X, y)

    def queen_epsilon_check(self, X, y):
        '''
        Spawn some 'workers' in a small epsilon range around the queen. If any
          have a lower cost than the queen, replace the queen!
        '''
        a = self.param_shape[0]
        b = self.param_shape[1]
        queencost = self.J(X, y, self.queen.params)
        for w in range(self.n_workers):
            wparams = self.queen.params + ((np.random.rand(a, b) - .5) *
                                           self.worker_eps)
            wcost = self.J(X, y, wparams)
            if wcost < queencost:
                queencost = wcost
                self.queen.params = wparams
                self.queen_reign = 0


class Bee(object):
    def __init__(self, params, swarm):
        self.swarm = swarm
        self.params = params
        self.best = False
        self.near_best = False

    def __repr__(self):
        return str(self.params)

    def step(self, queen):
        '''
        Take one step, likely towards the queen but perhaps on a random vector
        Length of random steps are proportional to distance from queen.
        '''
        if self == queen:
            return
        d = dist(self.params, queen.params)
        if np.random.rand < self.swarm.stray_rate:
            # take a random step of the right size
            self.params += self.generate_stray_vector(d)
        else:
            self.params += self.swarm.learning_rate * \
                (queen.params - self.params)

    def generate_stray_vector(self, d):
        '''
        Determines a random direction to stray in, normalizes that distance
          vector, and multiplies it by distance d.
        '''
        rv = np.random.rand(self.params.shape[0], self.params.shape[1]) - .5
        return rv / norm(rv) * d


def dist(v1, v2):
    '''
    Returns the euclidean distance between vectors v1 and v2
    '''
    a = v1.ravel()
    b = v2.ravel()
    return np.sqrt(np.sum([(a[i] - b[i]) ** 2 for i in range(len(a))]))


def test_cost_function(X, y, params):
    '''
    For testing purposes: simply returns the euclidean distance from the params
      vector to the origin.
    '''
    return dist(params, np.zeros(params.shape))


def RMSE(X, y, params):
    params = params.reshape(len(params), 1)
    y = y.reshape(len(y), 1)
    RMSE = np.sqrt(((np.dot(X, params)-y)**2).mean())
    return RMSE
