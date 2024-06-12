import numpy as np
import itertools
from pathSum2 import *
from scipy.optimize import minimize
from scipy.stats import norm
from typing import Callable
from sympy import *


def vdCB_UCB(K, T, theta, tau, sigma, d, mu_func: Callable, verbose=False, single=False):
    """
    UCB algorithm with variable lookahead
    
    Args:
    K: number of arms
    T: time horizon
    theta: dict parameter for mean reward
    tau: memory size
    sigma: standard deviation
    d: lookahead steps

    """
    if K < 1:
        raise ValueError("K must be at least 1")

    if K != len(theta):
        raise ValueError("K must equal the length of theta")

    if T < 1:
        raise ValueError("T must be at least 1")
    
    if tau < 0:
        raise ValueError("tau must be at least 0")
    
    if sigma < 0:
        raise ValueError("sigma must be at least 0")
    
    if d < 1:
        raise ValueError("m must be at least 1")
    
    if isinstance(theta, dict) == False:
        raise ValueError("theta must be a dictionary")
    
    if d > K and single:
        raise ValueError("m must be less than or equal to K")
    
    ntilde, history = {}, {} # initialise ntilde, history
    arm_history, reward_list, reward_optimal, regret = [], [], [], [] # initialise lists
    reward_history, empirical_mean, UCB, n, mu = np.zeros((K, tau+1)), np.zeros((K, tau+1)), np.zeros((K, tau+1)), np.zeros((K, tau+1)), np.zeros((K, tau+1)) # initialise matrices
 
    for i in range(K):
        ntilde[i+1] = 0
        for j in range(tau+1):
            mu[i, j] = mu_func(theta[i+1], j)

    if single:
        allPaths = list(itertools.permutations(list(range(1, K+1)), d))
    else:
        allPaths = list(itertools.product(list(range(1, K+1)), repeat=d))

    # initial exploration

    time = 1
    arm_order = np.random.permutation(K)

    for t, nt in itertools.product(arm_order, range(tau+1)):

        # collect rewards for each arm
        reward = np.random.normal(mu_func(theta[t+1], ntilde[t+1]), sigma)

        reward_list.append(reward)
        history[time] = (t+1, ntilde[t+1], reward)

        reward_history[t, nt] = reward
        n[t, nt] += 1
        empirical_mean[t, nt] = reward
        UCB[t, nt] = empirical_mean[t, nt] + np.sqrt(2 * sigma**2 * np.log(T) / n[t, nt])

        if verbose:
            print(f"Time: {time}, Arm: {t+1}, Reward: {reward}")
            print(f"Ntilde: {ntilde}")
            print(f"UCB: {UCB}")
            print(f"Mu: {mu}")

        arm_history.append(t+1)
        for key in ntilde:
            ntilde[key] = len([i for i in arm_history[-tau:] if i == key])

        
       
        time += 1

    # main loop
    for t in range((K*(tau+1))+1, T+1):
        
        ntilde_temp = ntilde.copy()

        k0 = [arm_history[-1]]
        prior_path = arm_history[-(tau+1):-1]

        path, pathAvg = mPath(k0, list(range(1, K+1)), d, ntilde_temp, tau, prior_path, UCB, allPaths) # find path with highest cumulative UCB 

        k = len(path[1:])

        if verbose: 
            print(f"Time: {time}, Path: {path[1:]}", f"PathAvg: {pathAvg}")


    
        # loop through each arm in the path
        for idx, arm in enumerate(path[1:]):

            reward = np.random.normal(mu_func(theta[arm], ntilde[arm]), sigma)

            if verbose:
                print(f"Time: {time}, Arm: {arm}, Reward: {reward}")
                print(f"Ntilde: {ntilde}")
                print(f"UCB: {UCB}")
                print(f"Mu: {mu}")

            # update information
            reward_list.append(reward)
            history[time] = (arm, ntilde[arm], reward)
        
            reward_history[arm-1, ntilde[arm]] += reward 
            n[arm-1, ntilde[arm]] += 1
            empirical_mean[arm-1, ntilde[arm]] = reward_history[arm-1, ntilde[arm]] / n[arm-1, ntilde[arm]] 
            UCB[arm-1, ntilde[arm]] = empirical_mean[arm-1, ntilde[arm]] + np.sqrt(2 * sigma**2 * np.log(T) / n[arm-1, ntilde[arm]]) 
            arm_history.append(arm) 
            for key in ntilde:
                ntilde[key] = len([i for i in arm_history[-tau:] if i == key])

            time += 1

            if time > T and idx == k-1:
                break

        if time > T:
            break
    
    return reward_list[:T], regret[:T]


def vdCB_TS(K, T, theta, tau, sigma, d, mu_func: Callable, verbose=False, single=False):
    """
    Thompson Sampling algorithm with variable lookahead

    Args:
    K: number of arms
    T: time horizon
    theta: dict parameter for mean reward
    tau: memory size
    sigma: standard deviation
    d: lookahead steps
    """
    if K < 1:
        raise ValueError("K must be at least 1")

    if K != len(theta):
        raise ValueError("K must equal the length of theta")

    if T < 1:
        raise ValueError("T must be at least 1")
    
    if tau < 0:
        raise ValueError("tau must be at least 0")
    
    if sigma < 0:
        raise ValueError("sigma must be at least 0")
    
    if d < 1:
        raise ValueError("m must be at least 1")
    
    if isinstance(theta, dict) == False:
        raise ValueError("theta must be a dictionary")

    if d > K and single:
        raise ValueError("m must be less than or equal to K")
    
    arms, ntilde, history, prior = {}, {}, {}, {}
    arm_history, reward_list, reward_optimal, regret = [], [], [], []

    reward_history, n, true_reward, mu, sigma0, theta_posterior = np.zeros((K, tau+1)), np.zeros((K, tau+1)), np.zeros(K), np.zeros((K, tau+1)), np.ones((K, tau+1)), np.zeros((K, tau+1))

    mu0 = np.array([np.linspace(1, 0, tau+1)] * K)

    # initialise arms, ntilde, prior
    for j in range(K):
        ntilde[j+1] = 0
        for nt in range(tau+1):
            mu[j, nt] = mu_func(theta[j+1], nt)
            prior[(j+1, nt)] = lambda j=j, nt=nt: np.random.normal(mu0[j, nt], sigma0[j, nt])
    
    if single:
        allPaths = list(itertools.permutations(list(range(1, K+1)), d))
    else:
        allPaths = list(itertools.product(list(range(1, K+1)), repeat=d))
    
    arm_order = np.random.permutation(K)
    time = 1

    for t, nt in itertools.product(arm_order, range(tau+1)):

        reward = np.random.normal(mu_func(theta[t+1], ntilde[t+1]), sigma)

        reward_list.append(reward)
        history[time] = (t+1, ntilde[t+1], reward)

        reward_history[t, nt] = reward
        n[t, nt] += 1
        mu0_prev = mu0[t, nt]
        sigma0_prev = sigma0[t, nt]
        mu0[t, nt] = ((1/sigma0_prev**2 + n[t, nt]/sigma**2) **-1) * (mu0_prev/sigma0_prev**2 + reward_history[t, nt]/sigma**2)
        sigma0[t, nt] = np.sqrt((1/sigma0_prev**2 + n[t, nt]/sigma**2) **-1)

        if verbose:
            print(f"Time: {time}, Arm: {t+1}, Reward: {reward}")
            print(f"Ntilde: {ntilde}")
            print(f"Mu0: {mu0}, Sigma0: {sigma0}")
            print(f"Mu: {mu}")

        arm_history.append(t+1)
        for key in ntilde:
            ntilde[key] = len([i for i in arm_history[-tau:] if i == key])

        time += 1


    for t in range((K*(tau+1))+1, T+1):

        for j in range(K):
            for nt in range(tau + 1):
                theta_posterior[j, nt] = prior[(j+1, nt)]()

        ntilde_temp = ntilde.copy()
        k0 = [arm_history[-1]]
        prior_path = arm_history[-(tau+1):-1]

        path, pathSum = mPath(k0, list(range(1, K+1)), d, ntilde_temp, tau, prior_path, theta_posterior, allPaths)

        k = len(path[1:])

        if verbose:
            print(f"Time: {time}, Path: {path[1:]}, PathAvg: {pathSum}")
        
   
        dStep_reward = 0
        for idx, arm in enumerate(path[1:]):
           
            reward = np.random.normal(mu_func(theta[arm], ntilde[arm]), sigma)
            dStep_reward += mu_func(theta[arm], ntilde[arm])


            if verbose:
                print(f"Time: {time}, Arm: {arm}, Reward: {reward}")
                print(f"Ntilde: {ntilde}")
                print(f"Posterior: {theta_posterior}")
                print(f"Mu0: {mu0}")
                print(f"Mu: {mu}")

            reward_list.append(reward)
            history[time] = (arm, ntilde[arm], reward)

            reward_history[arm-1, ntilde[arm]] += reward
            n[arm-1, ntilde[arm]] += 1
            mu0_prev = mu0[arm-1, ntilde[arm]]
            sigma0_prev = sigma0[arm-1, ntilde[arm]]
            mu0[arm-1, ntilde[arm]] = ((1/sigma0_prev**2 + n[arm-1, ntilde[arm]]/sigma**2) **-1) * (mu0_prev/sigma0_prev**2 + reward_history[arm-1, ntilde[arm]]/sigma**2)
            sigma0[arm-1, ntilde[arm]] = np.sqrt((1/sigma0_prev**2 + n[arm-1, ntilde[arm]]/sigma**2) **-1)

            arm_history.append(arm)
            for key in ntilde:
                ntilde[key] = len([i for i in arm_history[-tau:] if i == key])

            time += 1

            if time > T and idx == k-1:
                break


        if time > T:
            break
            
    return reward_list[:T], regret[:T] 


def vdPCB_MLE(K, T, theta, tau, sigma, d, mu_func: Callable, distribution, verbose=False, single=False):
    """
    UCB tuned type algorithm with d-step lookahead
    """
    if K < 1:
        raise ValueError("K must be at least 1")

    if K != len(theta):
        raise ValueError("K must equal the length of theta")

    if T < 1:
        raise ValueError("T must be at least 1")
    
    if tau < 0:
        raise ValueError("tau must be at least 0")
    
    if sigma < 0:
        raise ValueError("sigma must be at least 0")
    
    if d < 1:
        raise ValueError("m must be at least 1")
    
    if isinstance(theta, dict) == False:
        raise ValueError("theta must be a dictionary")
    
    if d > K and single:
        raise ValueError("m must be less than or equal to K")

    theta_sym, n_sym = symbols('theta n')

    mu_function = mu_func(theta_sym, n_sym)
    muprime = mu_function.diff(theta_sym)
    muprime_func = lambdify((theta_sym, n_sym), muprime, 'numpy')

    
    if distribution == "Normal":
        def log_likelihood(theta, r, N, sigma):
            likelihood = 0
            for i in range(len(r)):
                likelihood += 0.5*np.log(2*np.pi*sigma**2) + 1/(2*sigma**2) * (r[i] - mu_func(theta, N[i]))**2 
            return likelihood
        def ieq(theta, thetaHat, r, N, sigma, time):
            T = len(r)
            D, V = 0, 0
            eta = -np.inf
            for i in range(len(r)):
                D += (sigma**2 + (mu_func(theta,N[i]) - mu_func(thetaHat, N[i])))/(2*sigma**2) - 0.5
                V += muprime_func(theta, N[i]) * (mu_func(theta, N[i]) - mu_func(thetaHat, N[i]))
                if (sigma**2 + (mu_func(theta,N[i]) - mu_func(thetaHat, N[i])))/(2*sigma**2) - 0.5 > eta:
                    eta = (sigma**2 + (mu_func(theta,N[i]) - mu_func(thetaHat, N[i])))/(2*sigma**2) - 0.5
            return -1/T * D + np.sqrt(min(eta/4, 1/(sigma**2 * T**3) * V**2) * np.log(time) * 1/T)
    else:
        raise ValueError("Distribution not supported")
    
    def objective(theta, n):
        return -1 * mu_func(theta, n)
    
    def MLE(r, N, sigma):
        result = minimize(log_likelihood, np.random.rand(), args=(r, N, sigma))
        return result.x

    def ucbMLE(thetaHat, r, N, sigma, time):
        constraints = {'type': 'ineq', 'fun': ieq, 'args': (thetaHat, r, N, sigma, time)}
        result = minimize(objective, thetaHat, args=(N[-1]), constraints=constraints)
        return result.x, result.fun

    thetaMLE, ntilde, ntilde_arm, reward_arm, history = {}, {}, {}, {}, {}
    reward_list, arm_history, reward_optimal, regret = [], [], [], []
    UCB, mu = np.zeros((K, tau + 1)), np.zeros((K, tau + 1))

    for i in range(K):
        thetaMLE[i+1] = -np.inf
        ntilde_arm[i+1] = []
        reward_arm[i+1] = []
        ntilde[i+1] = 0
        for j in range(tau + 1):
            mu[i, j] = mu_func(theta[i+1], j)

    if single:
        allPaths = list(itertools.permutations(list(range(1, K+1)), d))
    else:
        allPaths = list(itertools.product(list(range(1, K+1)), repeat=d))



    time = 1 
    arm_order = np.random.permutation(K)
    
    for t in arm_order:


        reward = np.random.normal(mu_func(theta[t+1], ntilde[t+1]), sigma)


        reward_list.append(reward)
        history[time] = (t+1, ntilde[t+1], reward)
        ntilde_arm[t+1].append(ntilde[t+1])
        reward_arm[t+1].append(reward)

        thetaMLE[t+1] = MLE(reward_arm[t+1], ntilde_arm[t+1], sigma)

        for nt in range(tau+1):
            ntilde_arm_temp = ntilde_arm[t+1].copy()
            ntilde_arm_temp[-1] = nt
            UCB[t, nt] = ucbMLE(thetaMLE[t+1], reward_arm[t+1], ntilde_arm_temp, sigma, time)[1] * -1

        if verbose:
            print(f"Time: {time}, Arm: {t+1}, Reward: {reward}")
            print(f"Ntilde: {ntilde}")
            print(f"ThetaMLE: {thetaMLE}")
            print(f"UCB: {UCB}")
            print(f"Mu: {mu}")

        arm_history.append(t+1)
        for key in ntilde:
            ntilde[key] = len([i for i in arm_history[-tau:] if i == key])

    
        time += 1

    if verbose:
        print("Initialisation complete")


    for t in range(K+1, T+1):

        ntilde_temp = ntilde.copy() 

        k0 = [arm_history[-1]] 
        
        prior_path = arm_history[-(tau+1):-1]
        if len (prior_path) < tau:
            prior_path = [0 for i in range(tau - len(prior_path))] + prior_path

     
        path, pathSum = mPath(k0, list(range(1, K+1)), d, ntilde_temp, tau, prior_path, UCB, allPaths) 
        k = len(path[1:])

        if verbose:
            print(f"Time: {time}, Path: {path[1:]}, PathAvg: {pathSum}")




        dStep_reward = 0
        for idx, arm in enumerate(path[1:]):

            reward = np.random.normal(mu_func(theta[arm], ntilde[arm]), sigma)
            dStep_reward += mu_func(theta[arm], ntilde[arm]) 

            if verbose:
                print(f"Time: {time}, Arm: {arm}, Reward: {reward}")
                print(f"Ntilde: {ntilde}")
                print(f"ThetaMLE: {thetaMLE}")
                print(f"UCB: {UCB}")
                print(f"Mu: {mu}")
            

            reward_list.append(reward)
            history[time] = (arm, ntilde[arm], reward)

            ntilde_arm[arm].append(ntilde[arm])
            reward_arm[arm].append(reward)
            thetaMLE[arm] = MLE(reward_arm[arm], ntilde_arm[arm], sigma)

            # update UCB of arm pulled and all ntilde values (row i and all columns)
            for nt in range(tau+1):
                ntilde_arm_temp = ntilde_arm[arm].copy()
                ntilde_arm_temp[-1] = nt
                UCB[arm-1, nt] = ucbMLE(thetaMLE[arm], reward_arm[arm], ntilde_arm_temp, sigma, time)[1] * -1


            arm_history.append(arm)
            for key in ntilde:
                ntilde[key] = len([i for i in arm_history[-tau:] if i == key])

            time += 1

            if time > T and idx == d-1:
                break
        
    
        if time > T:
            break

    return reward_list[:T]