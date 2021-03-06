import numpy as np
import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt
import time

##OVERALL IDEA:
"""
The overall goal of this program is to answer the question:
    'Is the two-stage review process able to be manipulated significantly?'

To answer this question, we want to find the distribution of total TPMS 
scores across the two stages, and examine how that compares to an assignment 
LP run with the same total paper capacities with just one round.

The one-round TPMS score will always be optimal. Additionally, the two-round 
score may never be able to reach the one-round score if most reviewers are 
needed across both rounds to achieve the ideal.

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

"""the ideal split test runs once per selection of extra-papers. It creates a 
one-stage TPMS scenario where papers are reviewed equal to the total number of
times it would be reviewed in the two stages. This is deterministic and 
equivalent or better to having the best reviewer split possible"""
""""""

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


"""input_paper_constraints_LP is very much like solve_LP without a lot of fluff.
This was made separate since the differences between what was needed between 
the two-stage and the ideal-one-round was too large and complicated."""
def input_paper_constraints_LP(k, paper_constraints):
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
            model.addConstr(papers <= k) #each reviewer reviewers <= k papers

        for j in range(d):
            reviewers = 0
            for i in range(n):
                reviewers += A[i][j]
            model.addConstr(reviewers == paper_constraints[j]) #each paper gets specific number of reviews
        
        model.optimize()
        return model.objVal
    
    except gp.GurobiError as e:
        print("Error code " + str(e.errno) + ": " + str(e))
    except AttributeError:
        print("Attribute error")


def split_reviewers_start_TPMS(rev_split, split_pap_count, split_paps, k, l1, l2):

    A = [([0] * d) for i in range(n)] #A represents first step assignment
    AA = [([0] * d) for i in range(n)] #AA represents second step assignment
    
    split_rev_count = int(np.round(n * rev_split/100))
    split_revs = np.random.choice(n, split_rev_count, replace = False)
    
    """this comment block is code for storing the assignments - since 
    storing assignments takes a LOT of space, this code is currently dormant."""
    # (score1, assignment1) = solve_LP(A, 1, split_revs, [], k, l1)
    # (score2, assignment2) = solve_LP(AA, 2, split_revs, split_paps, k, l2)
    # score = score1 + score2
    # return (score, split_revs, (assignment1, assignment2))
    
    score = solve_LP(A, 1, split_revs, [], k, l1) + solve_LP(AA, 2, split_revs, split_paps, k, l2)
    return (score, split_revs)
    

def main():
    total_calls = pap_samples * rev_samples #progress bar for testing
    current_call = 0
    
    scores = []
    ideal_match_qualities = []
    min_scores = []
    min_split_revss = []
    max_scores = []
    max_split_revss = []
    
    split_pap_count = int(np.round(d * pap_split/100))
    pap_split_record = []
    rev_split_record = []
    
    total_matches = l1 * d + l2 * split_pap_count
    
    for pap_iter in range(pap_samples):
        max_score = -1
        max_split_revs = None
        min_score = np.inf
        min_split_revs = None
        
        rev_splits_for_this_sample = []
        scores_for_this_sample = []
        
        split_paps = np.random.choice(d, split_pap_count, replace = False)
        pap_split_record.append(split_paps)
        
        ##IDEAL CALCULATION PORTION (control)
        #construct matrix of modified reviews per paper
        ideal_split_test_capacities = [l1] * d
        
        #adding the stage 2 capacities
        for paper in split_paps:
            ideal_split_test_capacities[paper] += l2
        
        ideal_match_quality = input_paper_constraints_LP(k, ideal_split_test_capacities)/total_matches
        ideal_match_qualities.append(ideal_match_quality)
        
        ##TWO-STAGE PORTION (experimental)
        for rev_iter in range(rev_samples):
            
            result = split_reviewers_start_TPMS(rev_split, split_pap_count, split_paps, k, l1, l2)
            score = result[0]/total_matches
            scores_for_this_sample.append(score)
            
            if score > max_score:
                max_score = score
                max_split_revs = result[1]
            if score < min_score:
                min_score = score
                min_split_revs = result[1]
            
            rev_splits_for_this_sample.append(result[1])
            
            current_call += 1
            print(f"{current_call}/{total_calls} trials done\n")
        
        rev_split_record.append(rev_splits_for_this_sample)
        scores = np.append(scores, scores_for_this_sample)
        min_scores.append(min_score)
        min_split_revss.append(min_split_revs)
        max_scores.append(max_score)
        max_split_revss.append(max_split_revs)
        
        ##PLOTTING HISTOGRAMS (one per outer loop iteration)
        plt.hist([ideal_match_quality])
        
        plt.hist(scores_for_this_sample)
        plt.savefig(f"score_distribution_{pap_iter}")
        plt.clf()
    
    end_time = time.time()
    time_taken = end_time - start_time
    print(f"\nfinished, time taken = {int(time_taken/60)} min, saving...\n")


##SAVING INFORMATION
    title = f"rev{rev_split} {rev_samples} samples, pap{pap_split} {pap_samples} samples iclr "
    
    np.savez(title, \
    rev_split = rev_split, \
    pap_split = pap_split, \
    k = k, \
    l1 = l1, \
    l2 = l2, \
    pap_samples = pap_samples, \
    rev_samples = rev_samples, \
    dataset = dataset, \
    min_scores = min_scores, \
    min_split_revss = min_split_revss, \
    max_scores = max_scores, \
    max_split_revss = max_split_revss, \
    scores = scores, \
    ideal_match_qualities = ideal_match_qualities, \
    rev_split_record = rev_split_record, \
    pap_split_record = pap_split_record, \
    time_taken = time_taken)
    
    print("saving complete!")


##CODE START
start_time = time.time()


"""SET THE FOLLOWING VARIABLES:"""
rev_split = 10
pap_split = 10

k = 6
l1 = 3 #step 1's l value
l2 = 1 #step 2's l value

pap_samples = 4
rev_samples = 30
dataset = "iclr2018.npz"

#the following block is referenced from github.com/xycforgithub/StrategyProof_Conference_Review
scores = np.load(dataset)
S = scores["similarity_matrix"]
M = scores["mask_matrix"] #each entry is 0 or 1, with 1 representing a conflict.
n = len(S) #number of reviewers
d = len(S[0]) #number of papers

main()
