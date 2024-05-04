#!/usr/bin/env python
# coding: utf-8

# In[40]:


import numpy as np
import pandas as pd
import sys, copy, os, shutil, time
from dateutil.relativedelta import relativedelta

# glmnet implementation - for the LASSO + CV.
import glmnet_python
from glmnet import glmnet; from glmnetPlot import glmnetPlot
from glmnetPrint import glmnetPrint; from glmnetCoef import glmnetCoef; from glmnetPredict import glmnetPredict
from cvglmnet import cvglmnet; from cvglmnetCoef import cvglmnetCoef
from cvglmnetPlot import cvglmnetPlot; from cvglmnetPredict import cvglmnetPredict

# for logging + dynamic assessment of results
from IPython.display import clear_output

# are we gonna log results or nah?
verbose = False


# In[41]:


# create our possible list of settings
settings = []

# iterate through all possible settings, starting with to log-transform or not
for log_trans in [True, False]:
    
    # (protected local lags, protected global lags)
    for reg_scheme in [(0, 0), (1, 0), (1, 1), (2, 0), (2, 1), (2, 2)]:
    
        # how many weeks are we predicting into the future?
        for fh in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]:

            # how many historical lags are we considering when making predictions?
            for num_lags in [1, 2, 3, 4, 6, 12, 24]:

                # political (0), geographical (1), fully-connected (2), and no connections (3)
                for cluster_mech in [0, 1, 2, 3]:
                    
                    # add to our list of settings
                    settings.append([log_trans, fh, num_lags, cluster_mech, reg_scheme])


# In[49]:


# which index are we starting at? Governed by the command line argument! 144 jobs! 346 minutes apiece.
start_idx = int(sys.argv[1])

# let's do 28x variants per job
for i in range(start_idx*28, (start_idx*28)+28):
    
    # which setting are we working with?
    log_trans, fh, num_lags, cluster_mech, reg_scheme = settings[i]

    # what's the file name for this run?
    fname = f"fh={fh}_log-trans={log_trans}_num-lags={num_lags}_cluster-mech={cluster_mech}_reg-scheme={reg_scheme[0]}+{reg_scheme[1]}.csv"

    # let's load in our data
    cases = pd.read_csv("processed/weekly_cases.csv")
    cases.date = pd.to_datetime(cases.date)
    cases.set_index("date", inplace=True)

    # how many locations do we have?
    num_locations = len(cases.columns)

    # political state-based clustering
    if cluster_mech == 0:
        adj_matrix = np.loadtxt("processed/political_clustering/adj_matrix1.txt")

    # for distance-based clustering
    elif cluster_mech == 1:
        adj_matrix = np.loadtxt("processed/distance_clustering/adj_matrix2.txt")

    # for fully-connected clustering
    elif cluster_mech == 2:
        adj_matrix = np.loadtxt("processed/single_clustering/adj_matrix3.txt")

    # special option: just a diagonal matrix because treating each location separately!
    elif cluster_mech == 3:
        adj_matrix = np.eye(num_locations)

    # else, throw an error
    else:
        raise Exception("The specified cluster_mech is not available.")

    # what day is it today? and what is the end of our training set?
    today, train_end = pd.Timestamp(2020, 9, 27), pd.Timestamp(2020, 9, 27)

    # what is the first point ever in our training set, always?
    train_start = cases.iloc[:num_lags + fh].index[-1]

    # specify the end of our test set
    test_end = pd.Timestamp(2022, 5, 15)

    # create a dataframe to store our predictions
    predictions = pd.DataFrame(data=None, columns=["date"] + list(cases.columns))

    # set a seed re any stochasticity / randomness
    np.random.seed(858)

    # keep track of our R2s over time
    R2s = [np.nan]

    # initialize only for verbose logging
    if verbose:
        weeks_remaining = np.nan

    # iterate through each week in our test set until we are done.
    while True:

        # what is our prediction date based on the forecast horizon?
        pred_date = today + relativedelta(days=7*fh)

        # check if we've already made all the predictions we need
        if pred_date > test_end:
            break

        # let's do a quick sanity check to see if this is actually a feasible time to start training + making predictions ...
        if train_start <= today:

            # create a list to store our predictions for each location AT THIS TIMESTEP!
            preds = []

            # the move is to train + predict for each location individually
            for loc_idx in range(num_locations):

                ######## TRAINING OUR MODEL (FOR THIS LOCATION ONLY) ###########

                # create a data structure to encompass the X_train later
                X_train = []

                # build x_train and penalty_factor vectors for each training date
                for train_date in cases.loc[train_start : today].index:

                    # figure out which lags are we not regularizing for each of local vs. neighbor lags
                    local_reg, neighbor_reg = reg_scheme

                    # start our x_train point & our regularization vector
                    x_train = np.array([])
                    penalty_factor = np.array([])

                    # get all possible cases during the num_lags corresponding to this training point
                    avail_lags = cases.loc[train_date - relativedelta(days=7*(num_lags + fh - 1)) : \
                                           train_date - relativedelta(days=7*fh)]

                    # add in the self-loop first corresponding to local lags, also the penalty_factor
                    x_train = np.concatenate([x_train, copy.deepcopy(avail_lags[cases.columns[loc_idx]].values)])

                    # add in the local lags regularizer
                    penalty_factor_loc = np.ones(num_lags)
                    if (local_reg > 0) and (local_reg < num_lags):
                        penalty_factor_loc[-np.minimum(local_reg, num_lags):] = 0.0
                    penalty_factor = np.concatenate([penalty_factor, copy.deepcopy(penalty_factor_loc)])

                    # which locations are connected to this loc_idx?
                    relevant_neighbor_idxs = np.argwhere(adj_matrix[loc_idx] == 1.0).flatten()
                    relevant_neighbor_idxs = relevant_neighbor_idxs[relevant_neighbor_idxs != loc_idx]

                    # add in neighbor lags + the corresponding penalty_factor_neighbor
                    for neighbor_idx in relevant_neighbor_idxs:

                        # add this neighboring location's lags in
                        x_train = np.concatenate([x_train, copy.deepcopy(avail_lags[cases.columns[neighbor_idx]].values)])

                        # create the neighbor logs regularizer
                        penalty_factor_neighbor = np.ones(num_lags)
                        if (neighbor_reg > 0) and (neighbor_reg < num_lags):
                            penalty_factor_neighbor[-np.minimum(neighbor_reg, num_lags):] = 0.0
                        penalty_factor = np.concatenate([penalty_factor, copy.deepcopy(penalty_factor_neighbor)])

                    # add to our X_train structure
                    X_train.append(x_train)

                # assemble our X_train completely + compute our y_train FOR THIS LOCATION ONLY
                X_train = np.array(X_train)
                y_train = cases.loc[train_start : today][[cases.columns[loc_idx]]].values

                # check if we need to log-transform!
                if log_trans == True:
                    X_train = np.log1p(X_train)
                    y_train = np.log1p(y_train)

                # do a try catch because there is occasionally some errors ...
                try:

                    # train our model using cross-validated lasso
                    cvfit = cvglmnet(x = X_train, y = y_train, alpha = 1, nlambda = 10000, 
                                     penalty_factor = penalty_factor,
                                     ptype = 'mse', nfolds = 5)

                    ######## PREDICTING WITH OUR MODEL (FOR THIS LOCATION ONLY) ###########

                    # assemble x_test features for making our prediction AS A ROW VECTOR!
                    x_test = np.array([])

                    # get all possible cases during the num_lags corresponding to this PREDICTION POINT
                    avail_lags = cases.loc[pred_date - relativedelta(days=7*(num_lags + fh - 1)) : \
                                           pred_date - relativedelta(days=7*fh)]

                    # add in the self-loop first corresponding to local lags, also the penalty_factor
                    x_test = np.concatenate([x_test, copy.deepcopy(avail_lags[cases.columns[loc_idx]].values)])

                    # add in neighbor lags
                    for neighbor_idx in relevant_neighbor_idxs:

                        # add this neighboring location's lags in
                        x_test = np.concatenate([x_test, copy.deepcopy(avail_lags[cases.columns[neighbor_idx]].values)])

                    # cast to a ROW VECTOR + potentially log-transforming
                    x_test = x_test.reshape(1, -1)
                    if log_trans == True:
                        x_test = np.log1p(x_test)

                    # make our prediction
                    pred = cvglmnetPredict(cvfit, newx = x_test, s='lambda_1se')

                    # check if we need to reverse the log-transform
                    if log_trans == True:
                        pred = np.expm1(pred)

                    # add to our list
                    preds.append(pred[0, 0])

                # if there is an error due to glmnet + numpy instablity
                except:

                    # just do the naive persistence prediction
                    preds.append(cases.loc[today][cases.columns[loc_idx]])

            # add to our dataframe
            predictions.loc[len(predictions.index)] = [pred_date] + preds

            # compute R^2 to see how good we're doing ...
            SS_tot = ((cases.loc[pred_date] - cases.loc[pred_date].values.mean()) ** 2).sum()
            SS_res = ((np.array(preds) - cases.loc[pred_date].values) ** 2).sum()
            R2s.append( 1 - (SS_res / SS_tot) )

            # logging
            if verbose == True:

                # how many weeks remaining?
                weeks_remaining = ((test_end - pred_date) / 7).days

                # check our status
                clear_output(wait=True)
                print(f"{pred_date} | {weeks_remaining} weeks remaining. Most recent R^2: {R2s[-1]}")

        else:

            # just create a flag just says we didn't have enough days
            print("Not enough observations to start training yet!")

        # increment what today is
        today += relativedelta(days=7)

    # save our logs at the very end
    predictions.to_csv(f"linear_results/{fname}", index=False)

