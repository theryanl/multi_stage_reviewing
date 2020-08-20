import numpy as np
import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt

##OVERALL IDEA:
"""We do the standard TPMS objective assignment but with a little twist: 
At the start, we take out a fraction of the reviewers and do the initial 
assignment without them. Then, after the assignment, we take a portion of 
the papers (which may have been already assigned to other reviewers), and 
match them with the split-off reviewers. We record the combined similarity
of the two steps.

We run this over many combinations of reviewer/paper portions, and for 
each combination we run a fixed number of trials. After, we retrieve the
min, max, and mean of the combined similarity scores, and store the 
reviewer splits that caused the min and max. We also graph a comparison 
of the mean similarity scores for each combination."""

def solve_LP(S, M, A, n, d, step, split_revs, paps, k, l):
    #this function was created with reference to the Gurobi quickstart guide for mac on their website.
    try:
        model = gp.Model("my_model") 
        obj = 0 #objective is the total sum similarity
        
        for i in range(n):
            for j in range(d):
                
                if (M[i][j] == 1):
                    v = model.addVar(lb = 0, ub = 0, name = f"{i} {j}")
                else:
                    v = model.addVar(lb = 0, ub = 1, name = f"{i} {j}")
                
                A[i][j] = v
                obj += v * S[i][j]
        
        model.setObjective(obj, GRB.MAXIMIZE) #telling Gurobi to maximize obj
        
        for i in range(n):
            papers = 0
            for j in range(d):
                papers += A[i][j]
            if (step == 1):
                if (i in split_revs):
                    model.addConstr(papers == 0) #reviewer not in this step
                else:
                    model.addConstr(papers <= k) #each reviewer has k or less papers to review
            else:
                if (i in split_revs):
                    model.addConstr(papers <= k) #each reviewer has k or less papers to review
                else:
                    model.addConstr(papers == 0) #reviewer not in this step
        
        for j in range(d):
            reviewers = 0
            for i in range(n):
                reviewers += A[i][j]
            if (step == 1) or (j in paps):
                model.addConstr(reviewers == l) #each paper gets exactly l reviews
            else:
                model.addConstr(reviewers == 0)
        
        model.optimize()
        return model.objVal
    
    except gp.GurobiError as e:
        print("Error code " + str(e.errno) + ": " + str(e))
    except AttributeError:
        print("Attribute error")

def get_TPMS_similarity(rev_split, pap_split, k, l):
    #this function does set up work and calls solve_LP to return the TPMS similarity
    #the randomness of the code is completely within this function.
    #part of this is referenced from github.com/xycforgithub/StrategyProof_Conference_Review
    
    scores = np.load("iclr2018.npz")
    S = scores["similarity_matrix"]
    M = scores["mask_matrix"] #each entry is 0 or 1, with 1 representing a conflict.
    
    n = len(S) #number of reviewers
    d = len(S[0]) #number of papers

    A = [([0] * d) for i in range(n)] #A represents first step assignment
    AA = [([0] * d) for i in range(n)] #AA represents second step assignment
    
    split_rev_count = int(np.round(n * rev_split/100))
    split_revs = np.random.choice(n, split_rev_count, replace = False)
    split_pap_count = int(np.round(d * pap_split/100))
    split_paps = np.random.choice(d, split_pap_count, replace = False)
    
    score = solve_LP(S, M, A, n, d, 1, split_revs, [], k, l) + solve_LP(S, M, AA, n, d, 2, split_revs, split_paps, k, l)
    info = dict()
    info["split_rev_count"] = split_rev_count
    info["split_revs"] = split_revs
    info["split_pap_count"] = split_pap_count
    info["split_paps"] = split_paps
    return (score, info)

def save_info(info, score):
    file = open("saved_info.txt", "w")
    split_rev_count = info["split_rev_count"]
    split_revs = info["split_revs"]
    split_pap_count = info["split_pap_count"]
    split_paps = info["split_paps"]
    file.write(f"score: {score}\n\n")
    file.write(f"split_rev_count: {split_rev_count}\n")
    file.write(f"split_revs: \n{split_revs}\n\n")
    file.write(f"split_pap_count: {split_pap_count}\n")
    file.write(f"split_paps: \n{split_paps}")
    file.close()

def save_matrix(M, dims):
    file = open("saved_matrix.txt", "w")
    for i in range(dims[0]):
        for j in range(dims[1]):
            index = i * dims[1] + j
            item = M[index]
            file.write(f"{M[index]}, ")
        file.write("\n")
    file.close()

def main():
    rev_split_lo = 10 #lowest percent of reviewers that are split off in the trials
    rev_split_hi = 50 #highest percent ...
    rev_split_step = 10
    
    pap_split_lo = 10 #same but for papers
    pap_split_hi = 50
    pap_split_step = 10
    
    num_rev_splits = int(np.round((rev_split_hi - rev_split_lo)/rev_split_step + 1))
    num_pap_splits = int(np.round((pap_split_hi - pap_split_lo)/pap_split_step + 1))
    
    num_trials = 10 #number of times each rev_split value is run
    total_calls = num_trials * num_rev_splits * num_pap_splits
    
    k = 6
    l = 3
    
    scores = np.array([])
    max_score = -1
    max_split_info = None
    min_score = np.inf
    min_split_info = None
    current_call = 0
    
    for rev_split in range(rev_split_lo, rev_split_hi + rev_split_step, rev_split_step):
        for pap_split in range(pap_split_lo, pap_split_hi + pap_split_step, pap_split_step):
            
            scores_for_this_combo = []
            for i in range(num_trials):
                
                current_call += 1
                print(f"{current_call}/{total_calls}\n")
                
                results = get_TPMS_similarity(rev_split, pap_split, k, l)
                score = results[0]
                scores_for_this_combo.append(score)
                if score > max_score:
                    max_score = score
                    max_split_info = results[1]
                if score < min_score:
                    min_score = score
                    min_split_info = results[1]
            
            scores = np.append(scores, np.mean(scores_for_this_combo))
    
    if (max_split_info is not None):
        save_info(max_split_info, max_score)
    if (min_split_info is not None):
        save_info(min_split_info, min_score)
    
    # np.reshape(scores, (num_rev_splits, num_pap_splits))
    print(scores)
    save_matrix(scores, (num_rev_splits, num_pap_splits))
    
    control_trials = []
    for i in range(num_trials):
        control_trials.append(get_TPMS_similarity(0, 0, k, l)[0])
    control_score = np.mean(control_trials)
    print(control_score)
main()
