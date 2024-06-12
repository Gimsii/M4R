import numpy as np
import itertools

# Function to calculate the average reward of a path

def tauAvg(path, ntilde, tau, prior_path, reward_array):

    # add prior path to path to calculate ntilde properly
    combined_path = list(prior_path) + list(path)

    # initialise reward
    reward = []

    # loop through each arm in the path
    for idx, k in enumerate(path):

        # update ntilde
        if idx < tau:
            for key in ntilde:
                ntilde[key] = len([i for i in combined_path[idx:idx+tau] if i == key])
        else:
            for key in ntilde:
                ntilde[key] = len([i for i in path[idx-tau:idx] if i == key])

        #update reward
        if idx == 0:
            pass
        else:
            reward.append(reward_array[k-1, ntilde[k]])
    
    return np.cumsum(reward)/np.arange(1, len(reward)+1)

def mPath(k0, arms, m, ntilde, tau, prior_path, reward_array, allPaths):

    # initialise maximum cumulated reward and path
    maxSum = -np.inf
    maxPath = []

    # loop through all paths and find the one with the maximum cumulative reward
    for path in allPaths:
        
        # add k0 to the beginning of the path
        path = k0 + list(path)
        # calculate the cumulative reward of the path
        # it is average over path not cumulative sum
        allPathSum = tauAvg(path, ntilde, tau, prior_path, reward_array)

        # return idx of the maximum cumulative reward in allPathSum
        idx = np.argmax(allPathSum)

        pathSum = allPathSum[idx]

        # update maxSum and maxPath if necessary
        if pathSum > maxSum:
            maxSum = pathSum
            maxPath = path[:idx+2]

    return maxPath, maxSum






























# def tau_sum_ts(path ,ntilde, tau, prior_path, prior):

#     # add prior path to path to calculate ntilde properly
#     combined_path = list(prior_path) + list(path)

#     # initialise reward
#     reward = 0

#     # loop through each arm in the path
#     for idx, k in enumerate(path):


#         # update ntilde
#         if idx < tau: 
#             for key in ntilde:
#                 ntilde[key] = len([i for i in combined_path[idx:idx+tau] if i == key])

#         else:
#             for key in ntilde:
#                 ntilde[key] = len([i for i in path[idx-tau:idx] if i == key])
        
#         #update reward
#         reward += prior[(k, ntilde[k])](k-1, ntilde[k])

#     return reward

# def mPathTS(k0, arms, m, ntilde, tau, prior_path, prior, allPaths):

#     maxSum = -np.inf
#     maxPath = []

#     for path in allPaths:
        
#         path = k0 + list(path)
#         pathSum = tau_sum_ts(path, ntilde, tau, prior_path, prior)

#         if pathSum > maxSum:
#             maxSum = pathSum
#             maxPath = path

#     return maxPath, maxSum
