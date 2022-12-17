import numpy as np
import sys
import matplotlib.pylab as plt
import itertools
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Rectangle, Polygon
from IPython.display import clear_output
from tabulate import tabulate
import pickle

OPEN = 110
NARROW = 115
WALL = 111
START = 112
REWARD = 113
OUT_OF_BOUNDS = 114
LAVA = 118

TILES = {OPEN, WALL, START, REWARD, LAVA}

STR_MAP = {
    'O': OPEN,
    'N': NARROW,
    '#': WALL,
    'S': START,
    'R': REWARD,
    'L': LAVA
}

RENDER_DICT = {v:k for k, v in STR_MAP.items()}
RENDER_DICT[OPEN] = ' '
RENDER_DICT[NARROW] = ' '
RENDER_DICT[START] = ' '


def spec_from_string(s, valmap=STR_MAP):
    if s.endswith('\\'):
        s = s[:-1]
    rows = s.split('\\')
    rowlens = np.array([len(row) for row in rows])
    assert np.all(rowlens == rowlens[0])
    w, h = len(rows[0]), len(rows)

    gs = GridSpec(w, h)
    for i in range(h):
        for j in range(w):
            gs[j,i] = valmap[rows[i][j]]
    return gs


def spec_from_sparse_locations(w, h, tile_to_locs):
    """

    Example usage:
    >> spec_from_sparse_locations(10, 10, {START: [(0,0)], REWARD: [(7,8), (8,8)]})

    """
    gs = GridSpec(w, h)
    for tile_type in tile_to_locs:
        locs = np.array(tile_to_locs[tile_type])
        for i in range(locs.shape[0]):
            gs[tuple(locs[i])] = tile_type
    return gs


def local_spec(map, xpnt):
    """
    >>> local_spec("yOy\\\\Oxy", xpnt=(5,5))
    array([[4, 4],
           [6, 4],
           [6, 5]])
    """
    Y = 0; X=1; O=2
    valmap={
        'y': Y,
        'x': X,
        'O': O
    }
    gs = spec_from_string(map, valmap=valmap)
    ys = gs.find(Y)
    x = gs.find(X)
    result = ys-x + np.array(xpnt)
    return result


class GridSpec(object):
    def __init__(self, w, h):
        self.__data = np.zeros((w, h), dtype=np.int32)
        self.__w = w
        self.__h = h

    def __setitem__(self, key, val):
        self.__data[key] = val

    def __getitem__(self, key):
        if self.out_of_bounds(key):
            raise NotImplementedError("Out of bounds:"+str(key))
        return self.__data[tuple(key)]

    def out_of_bounds(self, wh):
        """ Return true if x, y is out of bounds """
        w, h = wh
        if w<0 or w>=self.__w:
            return True
        if h < 0 or h >= self.__h:
            return True
        return False

    def get_neighbors(self, k, xy=False):
        """ Return values of up, down, left, and right tiles """
        if not xy:
            k = self.idx_to_xy(k)
        offsets = [np.array([0,-1]), np.array([0,1]),
                   np.array([-1,0]), np.array([1,0])]
        neighbors = \
            [self[k+offset] if (not self.out_of_bounds(k+offset)) else OUT_OF_BOUNDS for offset in offsets ]
        return neighbors

    def get_value(self, k, xy=False):
        """ Return values of up, down, left, and right tiles """
        if not xy:
            k = self.idx_to_xy(k)
        return self[k]

    def find(self, value):
        return np.array(np.where(self.spec == value)).T
    
    def find_non(self, values):
        idxs = np.arange(self.spec.size)
        if isinstance(values, list):
            for value in values:
                idxs = np.delete(idxs, self.xy_to_idx(self.find(value)))
        else:
            idxs = np.delete(idxs, self.xy_to_idx(self.find(values)))
        return idxs

    @property
    def spec(self):
        return self.__data

    @property
    def width(self):
        return self.__w

    def __len__(self):
        return self.__w*self.__h

    @property
    def height(self):
        return self.__h

    def idx_to_xy(self, idx):
        if hasattr(idx, '__len__'):  # array
            x = idx % self.__w
            y = np.floor(idx/self.__w).astype(np.int32)
            xy = np.c_[x,y]
            return xy
        else:
            return np.array([ idx % self.__w, int(np.floor(idx/self.__w))])

    def xy_to_idx(self, key):
        shape = np.array(key).shape
        if len(shape) == 1:
            return key[0] + key[1]*self.__w
        elif len(shape) == 2:
            return key[:,0] + key[:,1]*self.__w
        else:
            raise NotImplementedError()

    def __hash__(self):
        data = (self.__w, self.__h) + tuple(self.__data.reshape([-1]).tolist())
        return hash(data)
 

ACT_NOOP = 0
ACT_UP = 1
ACT_DOWN = 2
ACT_LEFT = 3
ACT_RIGHT = 4
ACT_DICT = {
    ACT_NOOP: [0,0],
    ACT_UP: [0, -1],
    ACT_LEFT: [-1, 0],
    ACT_RIGHT: [+1, 0],
    ACT_DOWN: [0, +1]
}
ACT_TO_STR = {
    ACT_NOOP: 'NOOP',
    ACT_UP: 'UP',
    ACT_LEFT: 'LEFT',
    ACT_RIGHT: 'RIGHT',
    ACT_DOWN: 'DOWN'
}

class TransitionModel(object):
    def __init__(self, gridspec, eps=0.2):
        self.gs = gridspec
        self.eps = eps

    def get_aprobs(self, s, a):
        # TODO: could probably output a matrix over all states...
        legal_moves = self.__get_legal_moves(s)
        p = np.zeros(len(ACT_DICT))
        p[legal_moves] = self.eps / (len(legal_moves))
        if a in legal_moves:
            p[a] += 1.0-self.eps
        else:
            #p = np.array([1.0,0,0,0,0])  # NOOP
            p[ACT_NOOP] += 1.0-self.eps
        return p

    def __get_legal_moves(self, s):
        xy = np.array(self.gs.idx_to_xy(s))
        moves = [move for move in ACT_DICT if not self.gs.out_of_bounds(xy+ACT_DICT[move])
                                             and self.gs[xy+ACT_DICT[move]] != WALL]
        return moves

      
OBS_ONEHOT = 'onehot'
OBS_RANDOM = 'random'
class GridEnv(object):
    def __init__(self, gridspec, 
                 teps=0.0,
                 observation_type=OBS_ONEHOT,
                 dim_obs=8):
        super(GridEnv, self).__init__()
        self.num_states = len(gridspec)
        self.num_actions = len(ACT_DICT)
        self.obs_type = observation_type
        self.gs = gridspec
        self.model = TransitionModel(gridspec, eps=teps)
        self._transition_matrix = None
       
        if self.obs_type == OBS_RANDOM:
          self.dim_obs = dim_obs
          self.obs_matrix = np.random.randn(self.num_states, self.dim_obs)
        else:
          self.dim_obs = self.gs.width+self.gs.height

        
    def observation(self, s):
        if self.obs_type == OBS_ONEHOT:
          xy_vec = np.zeros(self.gs.width+self.gs.height)
          xy = self.gs.idx_to_xy(s)
          xy_vec[xy[0]] = 1.0
          xy_vec[xy[1]+self.gs.width] = 1.0
          return xy_vec
        elif self.obs_type == OBS_RANDOM:
          return self.obs_matrix[s]
        else:
          raise ValueError("Invalid obs type %s" % self.obs_type)
        
    def reward(self, s, a, ns):
        """ 
        Returns the reward (float)
        """
        tile_type = self.gs[self.gs.idx_to_xy(s)]
        if tile_type == REWARD:
          return 1
        elif tile_type == LAVA:
          return -1
        else:
          return 0

    def transitions(self, s, a):
        """
        Returns a dictionary of next_state (int) -> prob (float)
        """
        tile_type = self.gs[self.gs.idx_to_xy(s)]
        #if tile_type == LAVA: # Lava gets you stuck
        #    return {s: 1.0}
        if tile_type == WALL:
          return {s: 1.0}

        aprobs = self.model.get_aprobs(s, a)
        t_dict = {}
        for sa in range(len(ACT_DICT)):
            if aprobs[sa] > 0:
                next_s = self.gs.idx_to_xy(s) + ACT_DICT[sa]
                next_s_idx = self.gs.xy_to_idx(next_s)
                t_dict[next_s_idx] = t_dict.get(next_s_idx, 0.0) + aprobs[sa]
        return t_dict
      
    def initial_state_distribution(self):
        start_idxs = np.array(np.where(self.gs.spec == START)).T
        num_starts = start_idxs.shape[0]
        initial_distribution = {}
        for i in range(num_starts):
          initial_distribution[self.gs.xy_to_idx(start_idxs[i])] = 1.0/num_starts 
        return initial_distribution

    def step_stateless(self, s, a, verbose=False):
        probs = self.transitions(s, a).items()
        ns_idx = np.random.choice(range(len(probs)), p=[p[1] for p in probs])
        ns = list(probs)[ns_idx][0]
        rew = self.reward(s, a, ns)
        return ns, rew

    def step(self, a, verbose=False):
        ns, r = self.step_stateless(self.__state, a, verbose=verbose)
        done = self.gs[self.gs.idx_to_xy(self.__state)] == REWARD
        self.__state = ns
        return ns, r, done, {}

    def reset(self):
        init_distr = self.initial_state_distribution().items()
        start_idx= np.random.choice(len(init_distr), p=[p[1] for p in init_distr])
        self.__state = list(init_distr)[start_idx][0]
        self._timestep = 0
        return self.__state

    def render(self, close=False, ostream=sys.stdout):
        if close:
            return

        state = self.__state
        ostream.write('-'*(self.gs.width+2)+'\n')
        for h in range(self.gs.height):
            ostream.write('|')
            for w in range(self.gs.width):
                if self.gs.xy_to_idx((w,h)) == state:
                    ostream.write('*')
                else:
                    val = self.gs[w, h]
                    ostream.write(RENDER_DICT[val])
            ostream.write('|\n')
        ostream.write('-' * (self.gs.width + 2)+'\n')
    
    def render_narrow_open(self, narrow, open, ostream=sys.stdout):
        ostream.write('-'*(self.gs.width+2)+'\n')
        for h in range(self.gs.height):
            ostream.write('|')
            for w in range(self.gs.width):
                if [h,w] in narrow:
                    ostream.write('+')
                elif [h,w] in open:
                    ostream.write(' ')
                else:
                    ostream.write("#")
            ostream.write('|\n')
        ostream.write('-' * (self.gs.width + 2)+'\n')
        
    def transition_matrix(self):
        if self._transition_matrix is None:
          transition_matrix = np.zeros((self.num_states, self.num_actions, self.num_states))
          for s in range(self.num_states):
            for a in range(self.num_actions):
              for ns, prob in self.transitions(s, a).items():
                transition_matrix[s,a,ns] = prob
          self._transition_matrix = transition_matrix
        return self._transition_matrix


# Tabular Q-iteration
def q_backup_sparse(env, q_values, discount=0.99):
    dS = env.num_states
    dA = env.num_actions
        
    new_q_values = np.zeros_like(q_values)
    value = np.max(q_values, axis=1)
    for s in range(dS):
        for a in range(dA):
            new_q_value = 0
            for ns, prob in env.transitions(s, a).items():
                new_q_value += prob * (env.reward(s,a,ns) + discount*value[ns])
            new_q_values[s,a] = new_q_value
    return new_q_values


def q_iteration(env, num_itrs=100, render=False, **kwargs):
  """
  Run tabular Q-iteration
  
  Args:
    env: A GridEnv object
    num_itrs (int): Number of FQI iterations to run
    render (bool): If True, will plot q-values after each iteration
  """
  q_values = np.zeros((env.num_states, env.num_actions))
  for i in range(num_itrs):
    q_values = q_backup_sparse(env, q_values, **kwargs)
    if render:
      plot_sa_values(env, q_values, update=True, title='Q-values')
  return q_values



#@title Plotting Code (double-click to expand)

PLT_NOOP = np.array([[-0.1,0.1], [-0.1,-0.1], [0.1,-0.1], [0.1,0.1]])
PLT_UP = np.array([[0,0], [0.5,0.5], [-0.5,0.5]])
PLT_LEFT = np.array([[0,0], [-0.5,0.5], [-0.5,-0.5]])
PLT_RIGHT = np.array([[0,0], [0.5,0.5], [0.5,-0.5]])
PLT_DOWN = np.array([[0,0], [0.5,-0.5], [-0.5,-0.5]])

TXT_OFFSET_VAL = 0.3
TXT_CENTERING = np.array([-0.08, -0.05])
TXT_NOOP = np.array([0.0,0])+TXT_CENTERING
TXT_UP = np.array([0,TXT_OFFSET_VAL])+TXT_CENTERING
TXT_LEFT = np.array([-TXT_OFFSET_VAL,0])+TXT_CENTERING
TXT_RIGHT = np.array([TXT_OFFSET_VAL,0])+TXT_CENTERING
TXT_DOWN = np.array([0,-TXT_OFFSET_VAL])+TXT_CENTERING

ACT_OFFSETS = [
    [PLT_NOOP, TXT_NOOP],
    [PLT_UP, TXT_UP],
    [PLT_DOWN, TXT_DOWN],
    [PLT_LEFT, TXT_LEFT],
    [PLT_RIGHT, TXT_RIGHT]
]

PLOT_CMAP = cm.RdYlBu

def plot_sa_values(env, q_values, text_values=True, 
                   invert_y=True, update=False,
                   title=None):
  w = env.gs.width
  h = env.gs.height
  
  if update:
    clear_output(wait=True)
  plt.figure()
  ax = plt.gca()
  normalized_values = q_values
  normalized_values = normalized_values - np.min(normalized_values)
  normalized_values = normalized_values/np.max(normalized_values)
  for x, y in itertools.product(range(w), range(h)):
      state_idx = env.gs.xy_to_idx((x, y))
      if invert_y:
          y = h-y-1
      xy = np.array([x, y])
      xy3 = np.expand_dims(xy, axis=0)

      for a in range(4, -1, -1):
          val = normalized_values[state_idx,a]
          og_val = q_values[state_idx,a]
          patch_offset, txt_offset = ACT_OFFSETS[a]
          if text_values:
              xy_text = xy+txt_offset
              ax.text(xy_text[0], xy_text[1], '%.1f'%og_val, size='small')
          color = PLOT_CMAP(val)
          ax.add_patch(Polygon(xy3+patch_offset, True,
                                     color=color))
  ax.set_xticks(np.arange(-1, w+1, 1))
  ax.set_yticks(np.arange(-1, h+1, 1))
  plt.grid()
  if title:
    plt.title(title)
  plt.show()

def plot_s_values(env, v_values, text_values=True, 
                  invert_y=True, update=False,
                  title=None):
  w = env.gs.width
  h = env.gs.height
  if update:
    clear_output(wait=True)
  plt.figure()
  ax = plt.gca()
  normalized_values = v_values
  normalized_values = normalized_values - np.min(normalized_values)
  normalized_values = normalized_values/np.max(normalized_values)
  for x, y in itertools.product(range(w), range(h)):
      state_idx = env.gs.xy_to_idx((x, y))
      if invert_y:
          y = h-y-1
      xy = np.array([x, y])

      val = normalized_values[state_idx]
      og_val = v_values[state_idx]
      if text_values:
          xy_text = xy
          ax.text(xy_text[0], xy_text[1], '%.1f'%og_val, size='small')
      color = PLOT_CMAP(val)
      ax.add_patch(Rectangle(xy-0.5, 1, 1, color=color))
  ax.set_xticks(np.arange(-1, w+1, 1))
  ax.set_yticks(np.arange(-1, h+1, 1))
  plt.grid()  
  if title:
    plt.title(title)
  plt.show()
  
def compute_policy_deterministic(q_values, eps_greedy=0.0):
  policy_probs = np.zeros_like(q_values)
  policy_probs[np.arange(policy_probs.shape[0]), np.argmax(q_values, axis=1)] = 1.0 - eps_greedy
  policy_probs += eps_greedy / (policy_probs.shape[1])
  return policy_probs

def compute_visitation(env, policy, discount=1.0, T=50):
    dS = env.num_states
    dA = env.num_actions
    state_visitation = np.zeros((dS, 1))
    for (state, prob) in env.initial_state_distribution().items():
        state_visitation[state] = prob
    t_matrix = env.transition_matrix()  # S x A x S
    sa_visit_t = np.zeros((dS, dA, T))
    s_visit_t = np.zeros((dS, T))

    norm_factor = 0.0
    for i in range(T):
        sa_visit = state_visitation * policy
        cur_discount = (discount ** i)
        sa_visit_t[:, :, i] = cur_discount * sa_visit
        norm_factor += cur_discount
        # sum-out (SA)S
        new_state_visitation = np.einsum('ij,ijk->k', sa_visit, t_matrix)
        state_visitation = np.expand_dims(new_state_visitation, axis=1)
        s_visit_t[:, i] = state_visitation.reshape(dS)
    sa_visitations = np.sum(sa_visit_t, axis=2) / norm_factor
    return sa_visitations, state_visitation

class State():
    def __init__(self, s, a, r, ns):
        self.s = s
        self.a = a
        self.r = r
        self.ns = ns
        self.V = None
        self.parent = None
        self.child = None
    
    def __repr__(self):
        return f"s={self.s}, a={self.a}, r={self.r}, ns={self.ns} V={self.V}"

class Trajectory():
    def __init__(self):
        self.start_state = None
        self.curr_state = None
        self.len = 0
    
    def add_state(self, state: State):
        if self.start_state == None:
            self.start_state = state
            self.curr_state = state
            self.len += 1
            return
        self.curr_state.child = state
        state.parent = self.curr_state

        self.curr_state = state
        self.len += 1

    def print_all(self,):
        s = self.start_state
        while s is not None:
            print(s)
            s = s.child
    
    def backprop(self, gamma):
        s = self.curr_state
        s.V = s.r

        while s.parent is not None:
            s = s.parent
            s.V = s.r + gamma*s.child.V

def compute_goal_greedy_policy(env):
    optimal_qvalues = q_iteration(env, num_itrs=180, discount=0.8, render=False)
    # Compute and plot the value function
    v_values = np.max(optimal_qvalues, axis=1)
    # plot_s_values(env, v_values, title='Values')
    print(tabulate(v_values.reshape(env.gs.height,env.gs.width).round(5)))
    # plot_sa_values(env, policy, title='Policy')

    goal_greedy_policy = compute_policy_deterministic(optimal_qvalues, eps_greedy=0)
    with open("goal_greedy_policy.pkl", "wb") as f:
        pickle.dump(goal_greedy_policy ,f)
    return goal_greedy_policy

def compute_behavior_policy(env, goal_greedy_policy):
    def flip_direction(logits):
        if logits[1] == 1.0:
            return [0., 0., 1., 0., 0.]
        elif logits[2] == 1.0:
            return [0., 1., 0., 0., 0.]
        elif logits[3] == 1.0:
            return [0., 0., 0., 0., 1.]
        elif logits[4] == 1.0:
            return [0., 0., 0., 1., 0.]
        else:
            ValueError("can't swap, no deterministic direction found")

    dS = env.num_states
    dA = env.num_actions
    behavior_policy = np.zeros((dS, dA))

    for s in range(dS):
        tile_type = maze[maze.idx_to_xy(s)]
        if tile_type == NARROW or tile_type == START or tile_type == REWARD:
            behavior_policy[s] = goal_greedy_policy[s]
        if maze[maze.idx_to_xy(s)] == OPEN:
            flipped_direction_policy = flip_direction(goal_greedy_policy[s])
            new_behavior_policy = np.ones(dA)
            new_behavior_policy[np.argmax(flipped_direction_policy)] = (dA-1) * 4
            behavior_policy[s] = new_behavior_policy/sum(new_behavior_policy)
    with open("behavior_policy.pkl", "wb") as f:
        pickle.dump(behavior_policy ,f)

if __name__ == "__main__":
    maze = spec_from_string(
        "#SNNN#NNNNNNNN##\\"+
        "###N####N####N##\\"+
        "##NN###NN###NN##\\"+
        "##N####N####N###\\"+
        "#OOOO#OOOO#OOOO#\\"+
        "#OOOO#OOOO#OOOO#\\"+
        "#OOOO#OOOO#OOOO#\\"+
        "#OOOO#OOOO#OOOO#\\"+
        "#OOO##OOO##OOO##\\"+
        "###N####N####N##\\"+
        "##NN###NN###NN##\\"+
        "##N####N####N###\\"+
        "#OOOO#OOOO#OOOO#\\"+
        "#OOOO#OOOO#OOOO#\\"+
        "#OOOO#OOOO#OOOO#\\"+
        "#OOOO#OOOO#OOOO#\\"+
        "#OOO##OOO##OOO##\\"+
        "###N####N####N##\\"+
        "##NN###NN###NN##\\"+
        "##N####N####N###\\"+
        "#OOOOOOOOO#OOOO#\\"+
        "#OOOOOOOOO#OOOO#\\"+
        "#OOOOOOOOO#OOOO#\\"+
        "#OOOOOOOOO#OOOR#\\"
    )
    # maze = spec_from_string("SNNO\\"+
    #                     "N##N\\"+
    #                     "OOOO\\"+
    #                     "N#RO\\"
    #                    )
    env = GridEnv(maze, observation_type=OBS_RANDOM, dim_obs=8)

    goal_greedy_policy = compute_goal_greedy_policy(env)
    # behavior_poliicy = compute_behavior_policy(env, goal_greedy_policy)


    with open("behavior_policy.pkl", "rb") as f:
        behavior_policy = pickle.load(f)
    # with open("goal_greedy_policy.pkl", "rb") as f:
    #     goal_greedy_policy = pickle.load(f)
    # sa_visitations, s_visitations = compute_visitation(env, behavior_policy, T=400)
    # sa_visitationsg, s_visitations = compute_visitation(env, goal_greedy_policy, T=400)
    # print(tabulate(s_visitations.reshape(24,16)))
    dS = env.num_states
    dA = env.num_actions
    gamma = 0.9

    critic = np.ones((dS, dA))*1/(1-gamma) * 0.5
    actor = np.ones((dS,dA))/dA
    rho = np.ones((dS, dA))/dA

    breakpoint()

    # done = False
    # s = env.reset()
    # traj = Trajectory()
    # while not done:
    #     a = np.random.choice(dA, p=behavior_policy[s])
    #     ns, r, done, info = env.step(a)
    #     # print(f"{s=}, {a=}, {r=}, {ns=}")
    #     traj.add_state(State(s, a, r, ns))
    #     s = ns
    #     env.render()
    # traj.backprop(gamma=0.99)
    # traj.print_all()

