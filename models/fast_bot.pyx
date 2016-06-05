import cython
import cPickle
import tarfile

import numpy as np
cimport numpy as np

import interface as bb
cimport interface as bb

from libc.math cimport tanh

ctypedef np.float64_t DTYPE_t
ctypedef np.int64_t DTYPE_t_int

sq_f = [1,2]
intr_f = [1,4,11,12,13]

sq_f2 = [1, 2, 14, 16]
intr_f2 = []
    
sq_f3 = [0,1,2, 9, 10]
intr_f3 = [4, 11, 12, 13, 15, 18, 20]

cdef:
    int n_features = 36
    int n_actions = 4
    
    int n_hidden = 100
    float w1[36][100]
    float w2[100][4]
    float b1[100]
    float b2[4]
    
    int n_hidden_ = 16
    float w_1[36][16]
    float w_2[16][4]
    float b_1[16]
    float b_2[4]

    float min_state[36]
    float dif_state[36]

def prepare_bbox(lvl = 'test', verbose=0):
    bb.load_level('../levels/'+lvl+'_level.data', verbose)
        
def solve_lsq(X, y, lmd = 0):
    #regularization
    if lmd >0:
        Xsq = X.T.dot(X)
        I = np.diag([1]*Xsq.shape[0])
        I[-1,-1] = 0
        return np.linalg.inv(Xsq + lmd*I).dot(X.T.dot(y))
    else:
        return np.linalg.inv(X.T.dot(X)).dot(X.T.dot(y))

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef np.ndarray[DTYPE_t, ndim=1] fast_q_reg(float* state,
                                            np.ndarray[DTYPE_t, ndim=2] weights,
                                            np.ndarray[DTYPE_t_int, ndim=1] squares,
                                            np.ndarray[DTYPE_t_int, ndim=1] inter):
    cdef:
        int i, best_act = -1, a
        float best_val = -1e9, r
        int n_sq = squares.shape[0]
        int n_int = inter.shape[0]
        int n_features = weights.shape[1]
        np.ndarray[DTYPE_t, ndim = 1] qvals = np.zeros(4)

    for a in xrange(4):
        r = 0
        for i in xrange(36):
            r += state[i] * weights[a,i]

        for i in xrange(n_sq):
            r += state[squares[i]]**2 * weights[a,36+i]

        for i in xrange(n_int):
            r += state[inter[i]]*state[35] * weights[a,36+n_sq+i]

        r += weights[a, n_features-1]
        qvals[a] = r
    
    return qvals


cdef np.ndarray[DTYPE_t, ndim=1] fast_q_nn1(float* state):
    cdef:
        int i, best_act = -1
        float best_val = -1e9
        float s[36]
        float h[100]
        np.ndarray[DTYPE_t, ndim=1] y = np.zeros(4)

    
    for i in xrange(n_features):
        s[i] = (state[i] - min_state[i]) / dif_state[i]
        
    for i in xrange(n_hidden):
        h[i] = b1[i]
        for j in xrange(n_features):
            h[i] = h[i] + s[j] * w1[j][i]
        h[i] = tanh(h[i])
        
    for i in xrange(4):
        y[i] = b2[i]
        for j in xrange(n_hidden):
            y[i] = y[i] + h[j] * w2[j][i]
        if y[i] > best_val:
            best_val = y[i]
            best_act = i

    return y


cdef np.ndarray[DTYPE_t, ndim=1] fast_q_nn2(float* state):
    cdef:
        int i, best_act = -1
        float best_val = -1e9
        float s[36]
        float h[16]
        np.ndarray[DTYPE_t, ndim=1] y = np.zeros(4)

    
    for i in xrange(n_features):
        s[i] = (state[i] - min_state[i]) / dif_state[i]
        
    for i in xrange(n_hidden_):
        h[i] = b_1[i]
        for j in xrange(n_features):
            h[i] = h[i] + s[j] * w_1[j][i]
        h[i] = tanh(h[i])
        
    for i in xrange(4):
        y[i] = b_2[i]
        for j in xrange(n_hidden_):
            y[i] = y[i] + h[j] * w_2[j][i]
        if y[i] > best_val:
            best_val = y[i]
            best_act = i

    return y


with open('regr_data16.pkl', 'rb') as f:
    regr_d = cPickle.load(f)

X_train = regr_d['trainX'].astype(np.float64)
Y_train = regr_d['trainY'].astype(np.float64)
X_test = regr_d['testX'].astype(np.float64)
Y_test = regr_d['testY'].astype(np.float64)


f35_change_tr = np.zeros(X_train.shape[0]).astype('bool')
for i in range(1, len(X_train)):
    if X_train[i-1][35] != X_train[i][35]:
        f35_change_tr[i-1] = 1

f35_change_test = np.zeros(X_test.shape[0]).astype('bool')
for i in range(1, len(X_test)):
    if X_test[i-1][35] != X_test[i][35]:
        f35_change_test[i-1] = 1




def train1(X, S, V, A, M, sim_steps, lmb, sample_prob = 1):
    X_ = np.array(X[:-sim_steps])
    A_ = np.array(A[:-sim_steps])
    V_ = np.array(V[sim_steps:])
    R_ = np.array([S[i] - S[i-sim_steps] for i in range(len(S)) if i >= sim_steps])
    Y_ = R_ + V_
    
    M = np.array(M[:-sim_steps]).astype('bool')
    X = X_[M]
    A = A_[M]
    Y = Y_[M]

    f = np.zeros(X.shape[0]).astype('bool')
    for i in range(1, len(f)):
        if round(X[i-1][35]*10) != round(X[i][35]*10):
            f[i-1] = 0
        else:
            f[i-1] = 1
    
    
    f_a0 = A==0
    f_a1 = A==1
    f_a2 = A==2
    f_a3 = A==3 
    f_action = [f_a0, f_a1, f_a2, f_a3]
    
    weights = {}
   
    X_tr_sq = X[:, sq_f]**2
    X_tr_intr = X[:, intr_f]*X[:, 35].reshape(-1,1)
    bias = np.ones((X.shape[0], 1))           
    X_tr = np.concatenate([X, X_tr_sq, X_tr_intr, bias], axis =1)
    
    X_tr_sq2 = X_train[:, sq_f]**2
    X_tr_intr2 = X_train[:, intr_f]*X_train[:, 35].reshape(-1,1)
    bias2 = np.ones((X_train.shape[0], 1))           
    X_tr2 = np.concatenate([X_train, X_tr_sq2, X_tr_intr2, bias2], axis =1)

    X_tr_sq3 = X_test[:, sq_f]**2
    X_tr_intr3 = X_test[:, intr_f]*X_test[:, 35].reshape(-1,1)
    bias3 = np.ones((X_test.shape[0], 1))           
    X_tr3 = np.concatenate([X_test, X_tr_sq3, X_tr_intr3, bias3], axis =1)
    
    weights_tmp = []
    for i in range(4):
        m1 =  f
        X_train1 = X_tr[f_action[i] & m1]
        Y_train1 = Y[f_action[i] & m1]
            
        X_train2 = X_tr2
        Y_train2 = Y_train[i]
        m = np.random.rand(Y_train2.shape[0]) <= sample_prob
        X_train2 = X_train2[m | f35_change_tr]
        Y_train2 = Y_train2[m | f35_change_tr]

        X_train3 = X_tr3
        Y_train3 = Y_test[i]
        m3 = np.random.rand(Y_train3.shape[0]) <= 0.
        X_train3 = X_train3[m3 | f35_change_test]
        Y_train3 = Y_train3[m3 | f35_change_test]
        X_all = np.concatenate([X_train1, X_train2, X_train3], axis = 0)
        Y_all = np.concatenate([Y_train1, Y_train2, Y_train3], axis = 0)


        weights_tmp.append(solve_lsq(X_all, Y_all, lmb))
    for s in (-1, 0, 1):
        weights[s] = np.array(weights_tmp)
    
    return weights
    
def policy_improvment(sim_steps = 50, lvl = 'test', lmb = 10):
    cdef:
        int action
        float* state
        np.ndarray[DTYPE_t_int, ndim=1] f_sq1, f_intr1, f_sq2, f_intr2, f_sq3, f_intr3
        np.ndarray[DTYPE_t, ndim=1] q1, q2, q3, q4, q5

    f_sq1 = np.array([1,2], dtype= np.int64)
    f_intr1 = np.array([1,4,11,12,13], dtype= np.int64)
    
    f_sq2 = np.array([1, 2, 14, 16 ], dtype= np.int64)
    f_intr2 = np.array([], dtype= np.int64)
    
    f_sq3 = np.array([0,1,2, 9, 10], dtype= np.int64)
    f_intr3 = np.array([4,11,12, 13, 15, 18,20], dtype= np.int64)

    
    with open('weights_reg1.pkl', 'rb') as f:
        weights1 = cPickle.load(f)
        for k, v in weights1.iteritems():
            weights1[k] = v.astype(np.float64).T
    
    with open('weights_reg2.pkl', 'rb') as f:
        weights2 = cPickle.load(f)
        for k, v in weights2.iteritems():
            weights2[k] = v.astype(np.float64).T
    
    with open('weights_reg3.pkl', 'rb') as f:
        weights3 = cPickle.load(f)
        for k, v in weights3.iteritems():
            weights3[k] = v.astype(np.float64).T

    with open('state36.pkl', 'r') as file:
        state36 = cPickle.load(file)
        state_min = state36['state_min']
        state_dif = state36['state_dif']
        for i in xrange(n_features):
            min_state[i] = state_min[i]
            dif_state[i] = state_dif[i]
            
    with open('mean_weights_hid100.pkl', 'r') as file:
        weights = cPickle.load(file)
        for i in xrange(n_hidden):
            for j in xrange(n_features):
                w1[j][i] = weights[0][j,i]
            b1[i] = weights[1][i]
        for i in xrange(4):
            for j in xrange(n_hidden):
                w2[j][i] = weights[2][j,i]
            b2[i] = weights[3][i]

    with open('best_weights_hid16.pkl', 'r') as file:
        weights = cPickle.load(file)
        for i in xrange(n_hidden_):
            for j in xrange(n_features):
                w_1[j][i] = weights[0][j,i]
            b_1[i] = weights[1][i]
        for i in xrange(4):
            for j in xrange(n_hidden_):
                w_2[j][i] = weights[2][j,i]
            b_2[i] = weights[3][i]
    
    X = []
    S = []
    V1 = []
    A = []
    M1 = []
    
    s_prob = 0
    r1 = 1.
    r2 = 1.
    r3 = 1.
    r4 = 1.
    r5 = 1.
    
    train_steps = 200000
    h_len = 1
    s_ind =  -1*h_len*train_steps
    
    prepare_bbox(lvl)
    while True:
        X.append(bb.get_state().copy())
        S.append(bb.c_get_score())
                
        state = bb.c_get_state()
        if round(state[35]*10) == 1:
            f_35_s2 = 1
        elif round(state[35]*10) == -1:
            f_35_s2 = -1
        elif round(state[35]*10) >= 2:
            f_35_s2 = 2
        elif round(state[35]*10) <= -2:
            f_35_s2 = -2
        else:
            f_35_s2 = 0

        f_35_s1 = 0
        if f_35_s2 > 0:
            f_35_s1 = 1
        elif f_35_s2 < 0:
            f_35_s1 = -1

        q1 = fast_q_reg(state, weights1[f_35_s1], f_sq1, f_intr1)
        q2 = fast_q_reg(state, weights2[f_35_s2], f_sq2, f_intr2)
        q3 = fast_q_reg(state, weights3[f_35_s1], f_sq3, f_intr3)
        q4 = fast_q_nn1(state)
        q5 = fast_q_nn2(state)
        
        action = (r1*q1+r2*q2+r3*q3+r4*q4+r5*q5).argmax()
        if state[35] < -.5:
            action =  1
        if state[35] > .5:
            action =  2

        action1 = q1.argmax()
        if action == action1:
            M1.append(1)
        else:
            M1.append(0)

        A.append(action)
        V1.append(q1.max())
       
        
        if bb.c_get_time() % 100000 ==0:
            print 'steps {}, score {:.0f}'.format(bb.c_get_time(), bb.c_get_score())
        if bb.c_get_time() >= 100000 and bb.c_get_time() % train_steps ==0:
            new_weights1 = train1(X[s_ind:], S[s_ind:], V1[s_ind:], A[s_ind:], M1[s_ind:], sim_steps, lmb, s_prob)
            r1 += 0.1
            for k, v in weights1.iteritems():
                weights1[k] = 0.02*new_weights1[k] + 0.98*weights1[k]

        if bb.c_do_action(action) == 0:
            bb.finish(verbose=1)
            break
