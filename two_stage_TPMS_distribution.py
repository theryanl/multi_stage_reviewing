import numpy as np
import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt
import time

##OVERALL IDEA:
"""
The overall goal of this program is to answer the question:
    'is there obvious evidence that just picking one random choice of 
    reviewers is bad?'

To answer this question, we want to find the distribution of total TPMS 
scores across the two stages, and examine what fraction lays below a 
certain threshold.

The two-stage assignment is the standard TPMS objective assignment, but 
with a little twist: 
We pick a subset of papers to go through an additional review. We split 
off a fixed fraction of the reviewers at the start to review just these 
papers. Then, we run the TPMS LP two times; one for the majority of papers 
with the remaining reviewers, and another for the extra-review papers and 
the selected reviewers.

To find a simple distribution of total TPMS scores, we fix the extra-review 
papers and the number of reviewers we want to split off, and then randomly
sample reviewer splits for a number of trials. For each trial, we calculate 
the total TPMS score. We store this information in order to find the 
distribution.

As an optional additional step, we also retrieve the min, max, and mean of 
the combined similarity scores, and store the reviewer and paper splits that 
caused the min and max. 
As a part of data collection, we also store the paper split portion, reviewer 
split portion, papers selected, reviewers split off, and TPMS score. We can 
use this to re-generate the assignment if needed.
"""

##IMPLEMENTATION

"""
solve_LP creates a gurobi model, inputs variables (one for each 
reviewer-paper pair), adds constraints on the individual variables, adds 
constraints on the sum of papers for each reviewer, the sum of reviewers for 
each paper, and maximizes the TPMS score (similarity matrix * assignment).

For step 1, split_revs are limited such that they have 0 papers.
For step 2, non-split_revs are limited such that they have 0 papers, and 
non-selected papers are limited such that they have 0 reviewers.

The TPMS score is of the step is returned.
"""
def solve_LP(A, step, split_revs, paps, k, l):
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
        # return (model.objVal, A)
        return model.objVal
    
    except gp.GurobiError as e:
        print("Error code " + str(e.errno) + ": " + str(e))
    except AttributeError:
        print("Attribute error")


def get_TPMS_similarity(rev_split, split_pap_count, split_paps, k, l1, l2):

    A = [([0] * d) for i in range(n)] #A represents first step assignment
    AA = [([0] * d) for i in range(n)] #AA represents second step assignment
    
    split_rev_count = int(np.round(n * rev_split/100))
    split_revs = np.random.choice(n, split_rev_count, replace = False)
    
    # (score1, assignment1) = solve_LP(A, 1, split_revs, [], k, l1)
    # (score2, assignment2) = solve_LP(AA, 2, split_revs, split_paps, k, l2)
    # score = score1 + score2
    # return (score, split_revs, (assignment1, assignment2))
    score = solve_LP(A, 1, split_revs, [], k, l1) + solve_LP(AA, 2, split_revs, split_paps, k, l2)
    return (score, split_revs)
    

def main():
    num_rev_splits = int(np.round((rev_split_hi - rev_split_lo)/rev_split_step + 1))
    num_pap_splits = int(np.round((pap_split_hi - pap_split_lo)/pap_split_step + 1))
    
    total_calls = num_rev_splits * num_pap_splits * pap_samples * rev_samples #progress bar for testing
    
    scores = [] #TODO: scores is currently only a 1d list, would function better as a 2d list.
    rev_split_record = [[]] #TODO: ... 2d list, ... 3d list.
    pap_split_record = []
    # assignment_record = []
    
    max_score = -1
    max_split_revs = None
    min_score = np.inf
    min_split_revs = None
    current_call = 0
    
    for pap_split in range(pap_split_lo, pap_split_hi + pap_split_step, pap_split_step):
        
        for rev_split in range(rev_split_lo, rev_split_hi + rev_split_step, rev_split_step):
            
            for pap_iter in range(pap_samples):
                
                split_pap_count = int(np.round(d * pap_split/100))
                split_paps = np.random.choice(d, split_pap_count, replace = False)
                pap_split_record.append(split_paps)
                
                scores_for_this_pap_selection = []
                for rev_iter in range(rev_samples):
                    
                    result = get_TPMS_similarity(rev_split, split_pap_count, split_paps, k, l1, l2)
                    score = result[0]
                    scores_for_this_pap_selection.append(score)
                    
                    if score > max_score:
                        max_score = score
                        max_split_revs = result[1]
                    if score < min_score:
                        min_score = score
                        min_split_revs = result[1]
                    
                    rev_split_record.append(result[1])
                    # assignment_record.append(result[2])
                    
                    current_call += 1
                    print(f"{current_call}/{total_calls} trials done\n")
                    
                scores = np.append(scores, scores_for_this_pap_selection)
    
    # np.reshape(scores, (num_rev_splits, num_pap_splits))
    print("\nstarting control calculations\n")
    
    control_trials = []
    for i in range(num_control_trials):
        control_trials.append(get_TPMS_similarity(0, 0, np.array([]), k, l1, l1)[0])
    control_score = np.mean(control_trials)
    
    mean_score = np.mean(scores)
    plt.hist(scores)
    plt.savefig("score_distribution")
    plt.show()
    
    end_time = time.time()
    time_taken = end_time - start_time
    print(f"\nfinished, time taken = {time_taken} s, saving...\n")


##SAVING INFORMATION
    title = f"rev{rev_split_lo}-{rev_split_hi} {rev_samples} samples, pap{pap_split_lo}-{pap_split_hi} {pap_samples} samples iclr"
    
    np.savez(title, \
    rev_split_lo = rev_split_lo, \
    rev_split_hi = rev_split_hi, \
    rev_split_step = rev_split_step, \
    pap_split_lo = pap_split_lo, \
    pap_split_hi = pap_split_hi, \
    pap_split_step = pap_split_step, \
    k = k, \
    l1 = l1, \
    l2 = l2, \
    pap_samples = pap_samples, \
    rev_samples = rev_samples, \
    num_control_trials = num_control_trials, \
    dataset = dataset, \
    min_score = min_score, \
    min_split_revs = min_split_revs, \
    max_score = max_score, \
    max_split_revs = max_split_revs, \
    mean_score = mean_score, \
    control_score = control_score, \
    scores = scores, \
    rev_split_record = rev_split_record, \
    pap_split_record = pap_split_record, \
    # assignment_record = assignment_record, \
    time_taken = time_taken)
    
    print("saving complete!")


##CODE START
start_time = time.time()


"""SET THE FOLLOWING VARIABLES:"""
rev_split_lo = 30 #lowest percent of reviewers that are split off in the trials
rev_split_hi = 30 #highest percent ...
rev_split_step = 10

pap_split_lo = 30 #same but for papers
pap_split_hi = 30
pap_split_step = 10

k = 6
l1 = 3 #step 1's l value
l2 = 1 #step 2's l value

pap_samples = 1
rev_samples = 100
num_control_trials = 10
dataset = "iclr2018.npz"
""""""


#the following block is referenced from github.com/xycforgithub/StrategyProof_Conference_Review
scores = np.load(dataset)
S = scores["similarity_matrix"]
M = scores["mask_matrix"] #each entry is 0 or 1, with 1 representing a conflict.
n = len(S) #number of reviewers
d = len(S[0]) #number of papers

main()
