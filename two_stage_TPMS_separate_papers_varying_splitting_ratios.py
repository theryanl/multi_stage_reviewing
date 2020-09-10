import numpy as np
import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt
import time
import math

##OVERALL IDEA:
"""
The overall goal of this program is to answer the question:
    'Is our hypothesized curve for disjoint paper reviewer assignments valid?"

To answer this question, we want to find the average match quality sum of the 
two disjoint rounds whilst varying the reviewer split and paper split ratios.

To present the results, we want to show a line graph with error bars depicting 
the average match quality (y) vs. the reviewer and paper split (x). For 
feasibility's sake, we keep the reviewer split and paper split ratios identical.

This disjoint two-stage assignment differs from previous two-stage assignments 
in that the split-off papers are also not in the first round. 

We split the papers into two sets to review based on the paper split ratio. We 
also split the reviewers into two sets based on the reviewer split ratio to 
review these papers. We run the TPMS LP two times and find the sum, and then we 
divide the sum by the number of matched reviewer-paper pairs to get the avg 
match quality.

Through multiple trials, we vary the paper split/reviewer split ratio, and 
within those trials, we repeatedly create paper splits and reviewer splits via 
random selection, running a few reviewer splits per paper split. 
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
def solve_LP(step, split_revs, split_paps):
    #this function was created with reference to the Gurobi quickstart guide for mac on their website.
    try:
        model = gp.Model("my_model") 
        obj = 0 #objective is the total sum similarity
        A = [([0] * d) for i in range(n)] #A represents assignment
        
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
                    model.addConstr(papers <= k)
                else:
                    model.addConstr(papers == 0)
        
        for j in range(d):
            reviewers = 0
            for i in range(n):
                reviewers += A[i][j]
            if (step == 1):
                if (j in split_paps):
                    model.addConstr(reviewers == 0) #paper not in this step
                else:
                    model.addConstr(reviewers == l) #each paper gets exactly l reviews
            else:
                if (j in split_paps):
                    model.addConstr(reviewers == l)
                else:
                    model.addConstr(reviewers == 0)
        
        model.optimize()
        return model.objVal
    
    except gp.GurobiError as e:
        print("Error code " + str(e.errno) + ": " + str(e))
    except AttributeError:
        print("Attribute error")    

def main():
    scores = []
    mean_scores = []
    yerrors = []
    min_scores = []
    min_split_revss = []
    min_split_papss = []
    max_scores = []
    max_split_revss = []
    max_split_papss = []
    
    total_matches = d * l #num_papers * reviews_per_paper
    splits = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    samples_per_split = pap_samples_per_split * rev_samples_per_split
    sqrt_samples_per_split = math.sqrt(samples_per_split)
    total_calls = samples_per_split * len(splits) #progress bar for testing
    current_call = 0
    
    for split in splits:
        split_pap_count = int(np.round(d * split/100))
        split_rev_count = int(np.round(n * split/100))
        
        max_score = -1
        max_split_revs = None
        max_split_paps = None
        min_score = np.inf
        min_split_revs = None
        min_split_paps = None
        
        scores_for_this_split = []
        
        for pap_iter in range(pap_samples_per_split):
            
            split_paps = np.random.choice(d, split_pap_count, replace = False)
            
            for rev_iter in range(rev_samples_per_split):
                
                split_revs = np.random.choice(n, split_rev_count, replace = False)
                
                result = solve_LP(1, split_revs, split_paps) + \
                solve_LP(2, split_revs, split_paps)
                
                score = result/total_matches
                scores_for_this_split.append(score)
                
                if score > max_score:
                    max_score = score
                    max_split_revs = split_revs
                    max_split_paps = split_paps
                if score < min_score:
                    min_score = score
                    min_split_revs = split_revs
                    min_split_paps = split_paps
                
                current_call += 1
                print(f"{current_call}/{total_calls} trials done\n")
                
        scores.append(scores_for_this_split)
        mean_scores.append(np.mean(scores_for_this_split))
        yerrors.append(np.std(scores_for_this_split)/sqrt_samples_per_split)
        min_scores.append(min_score)
        min_split_revss.append(min_split_revs)
        min_split_papss.append(min_split_paps)
        max_scores.append(max_score)
        max_split_revss.append(max_split_revs)
        max_split_papss.append(max_split_paps)
        
        ##PLOTTING HISTOGRAMS (one per outer loop iteration)
    plt.errorbar(splits, mean_scores, yerr = yerrors)
    plt.savefig(f"two separate stage mean scores 0-100 {pap_samples_per_split} * {rev_samples_per_split} errorbar")
    plt.clf()
    
    end_time = time.time()
    time_taken = end_time - start_time
    print(f"\nfinished, time taken = {int(time_taken/60)} min, saving...\n")


##SAVING INFORMATION
    title = f"data for TPMS sum across rev splits {pap_samples_per_split} pap * {rev_samples_per_split} rev iclr"
    
    np.savez(title, \
    splits = splits, \
    k = k, \
    l = l, \
    pap_samples_per_split = pap_samples_per_split, \
    rev_samples_per_split = rev_samples_per_split, \
    dataset = dataset, \
    scores = scores, \
    mean_scores = mean_scores, \
    yerrors = yerrors, \
    min_scores = min_scores, \
    min_split_revss = min_split_revss, \
    min_split_papss = min_split_papss, \
    max_scores = max_scores, \
    max_split_revss = max_split_revss, \
    max_split_papss = max_split_papss, \
    time_taken = time_taken)
    
    print("saving complete!")


##CODE START
start_time = time.time()

"""SET THE FOLLOWING VARIABLES:"""
k = 6
l = 3

pap_samples_per_split = 3
rev_samples_per_split = 10
dataset = "iclr2018.npz"

#the following block is referenced from github.com/xycforgithub/StrategyProof_Conference_Review
dataset_scores = np.load(dataset)
S = dataset_scores["similarity_matrix"]
M = dataset_scores["mask_matrix"] #each entry is 0 or 1, with 1 representing a conflict.
n = len(S) #number of reviewers
d = len(S[0]) #number of papers

main()
