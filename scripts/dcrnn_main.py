import numpy as np
import pandas as pd
import sys, copy, os, shutil, time
from dateutil.relativedelta import relativedelta
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric_temporal.nn.recurrent import DCRNN
from torch_geometric_temporal.signal.static_graph_temporal_signal import StaticGraphTemporalSignal
from IPython.display import clear_output

# for notebooks only
verbose = False

# a global input setting
N_EPOCHS = 1000

# create a list of settings
settings = []

# iterate through all possible settings
for shuffle in [True, False]:
    for log_trans in [True, False]:
        for fh in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]:
            for num_lags in [1, 2, 3, 4, 6, 12, 24]:
                for cluster_mech in [0, 1, 2]:
                    for num_nodes in [8, 16, 32, 64, 128]:

                        # create a tuple + add to our settings
                        settings.append((shuffle, log_trans, fh, num_lags, cluster_mech, num_nodes))

# which index are we starting at? Governed by the command line argument!
start_idx = int(sys.argv[1])

# which settings are we working on? AS OF 4/16/2024: we have 864 JOBS!
for i in range(start_idx*15, (start_idx*15)+15):

    # get the setting that we're running
    shuffle, log_trans, fh, num_lags, cluster_mech, num_nodes = settings[i]

    # build our model dynamically according to the command-line arguments
    class Model(torch.nn.Module):
    
        # our initialization function
        def __init__(self):
    
            # call the parent constructor
            super(Model, self).__init__()
                
            # K is the filter size, which the docs recommended using for 1 in epidemiological setting.
            # https://pytorch-geometric-temporal.readthedocs.io/en/latest/notes/introduction.html#applications
            self.gcn = DCRNN(in_channels=num_lags, out_channels=num_nodes, K=1)
    
            # create our linear layer
            self.linear = torch.nn.Linear(in_features=num_nodes, out_features=1)
    
        # define our forward pass function
        def forward(self, x, edge_index, edge_weight):
    
            # standard graph module + activation + linear layer
            h = self.gcn(x, edge_index, edge_weight)
            h = F.relu(h)
            h = self.linear(h)
            return h


    # what's the file name for this run?
    fname = f"fh={fh}_log-trans={log_trans}_num-nodes={num_nodes}_cluster-mech={cluster_mech}_num-lags={num_lags}_shuffle={shuffle}.csv"
    
    # let's load in our data
    cases = pd.read_csv("processed/weekly_cases.csv")
    cases.date = pd.to_datetime(cases.date)
    cases.set_index("date", inplace=True)
    
    # how many locations do we have?
    num_locations = len(cases.columns)
    
    # what day is it today? and what is the end of our training set?
    today, train_end = pd.Timestamp(2020, 9, 27), pd.Timestamp(2020, 9, 27)
    
    # what is the first point ever in our training set, always?
    train_start = cases.iloc[:num_lags + fh].index[-1]
    
    # specify the end of our test set
    test_end = pd.Timestamp(2022, 5, 15)
    
    # load in our edge_index data, which we will treat as static
    if cluster_mech == 0:
        edge_index = torch.tensor(np.loadtxt("processed/political_clustering/edge_indices1.txt")).to(torch.int64)
    elif cluster_mech == 1:
        edge_index = torch.tensor(np.loadtxt("processed/distance_clustering/edge_indices2.txt")).to(torch.int64)
    elif cluster_mech == 2:
        edge_index = torch.tensor(np.loadtxt("processed/single_clustering/edge_indices3.txt")).to(torch.int64)
    else:
        raise Exception("Cluster mechanism is not currently supported.")
    
    # create a dataframe to store our predictions
    predictions = pd.DataFrame(data=None, columns=["date"] + list(cases.columns))
    
    # set a seed + initialize our model
    torch.manual_seed(858)
    model = Model()
    
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
        
            ######## TRAINING OUR MODEL ###########
        
            # get our relevant training cases lags
            all_training_lags = np.array([cases.loc[train_date - relativedelta(days=7*(num_lags + fh - 1))\
                                          : train_date - relativedelta(days=7*fh)].values.T \
                                          for train_date in cases.loc[train_start : today].index])
            all_training_targets = cases.loc[train_start : today].values
        
            # perform our log-transform if necessary
            if log_trans == True:
                all_training_lags = np.log1p(all_training_lags)
                all_training_targets = np.log1p(all_training_targets)
        
            # we're doing an expanding window approach -- build our train set!
            train_dataset = StaticGraphTemporalSignal(edge_index=edge_index, 
                                                      edge_weight=torch.ones(edge_index.shape[1]), 
                                                      features=all_training_lags,
                                                      targets=all_training_targets)
        
            # let's have the option to do a random reshuffle if our model is just performing straight poorly for the past 4 weeks
            if shuffle == True:
                if (len(R2s) >= 4) and ((np.array(R2s[-4:]) < 0).mean() == 1.0):
            
                    # re-initialize our model's weights!
                    model = Model()
        
                    # verbose logging
                    if verbose:
                        clear_output(wait=True)
                        print("Resetting model due to negative R^2 over past four predictions.")
                        time.sleep(seconds=3)
        
            # put our model into training mode + create our optimizer
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
            model.train()
            
            # train for many epochs
            for epoch in range(N_EPOCHS):
        
                # reset our cost function
                cost = 0.0
        
                # go thru each graph snapshot in our training dataset that we just built ...
                for t, snapshot in enumerate(train_dataset):
                    
                    # ... make our prediction and increment our cost
                    train_preds = model(x=snapshot.x, edge_index=snapshot.edge_index, edge_weight=snapshot.edge_attr)
                    cost = cost + torch.mean((train_preds - snapshot.y.reshape(-1, 1)) ** 2)
        
                # normalize cost by the number of datapoints, make our weight update and reset
                cost = cost / (t + 1)
                cost.backward()
                optimizer.step()
                optimizer.zero_grad()
        
                # are we going to do some verbose logging?
                if verbose == True:
        
                    # clear output + print loss at each step
                    clear_output(wait=True)
                    cost_np = np.round(float(cost.detach().numpy()), 3)
                    print(f"Today: {today} | Train Loss = {cost_np} | Last Test R^2 = {np.round(R2s[-1], 3)}, Weeks Remaining {str(weeks_remaining).zfill(3)} | Epoch {str(epoch+1).zfill(4)} of {N_EPOCHS}")
        
            ######## MAKING OUR PREDICTION ########
            
            # for prediction mode, let's first set our model to eval mode
            model.eval()
        
            # let's also make sure that no gradient is getting tracked
            with torch.no_grad():
        
                # what are our lags that we will consider
                pred_lags = cases.loc[today - relativedelta(days=(num_lags-1)*7) : today].values
        
                # do we need to log-transform?
                if log_trans == True:
                    pred_lags = np.log1p(pred_lags)
        
                # construct our new snapshot as a torch_geometric data object
                pred_snapshot = Data(x= torch.tensor(pred_lags).T.type(torch.FloatTensor),
                                     edge_index=edge_index, 
                                     edge_attr=torch.ones(edge_index.shape[1]))
            
                # make our predictions
                preds = model(x=pred_snapshot.x, edge_index=pred_snapshot.edge_index, edge_weight=pred_snapshot.edge_attr)\
                .detach().numpy().flatten()
        
                # if we did a log-transformation, we need to reverse it here!
                if log_trans == True:
                    preds = np.expm1(preds)
        
                # add to our dataframe
                predictions.loc[len(predictions.index)] = [pred_date] + list(preds)
        
                # compute R^2 to see how good we're doing ...
                SS_tot = ((cases.loc[pred_date] - cases.loc[pred_date].mean()) ** 2).sum()
                SS_res = ((preds - cases.loc[pred_date]) ** 2).sum()
                R2s.append( 1 - (SS_res / SS_tot) )
                
                # logging
                if verbose == True:
        
                    # how many weeks remaining?
                    weeks_remaining = ((test_end - pred_date) / 7).days
        
            ################

        else:

            # just create a flag just says we didn't have enough days
            print("Not enough observations to start training yet!")
        
        # increment what today is
        today += relativedelta(days=7)
    
    # save our logs at the very end
    predictions.to_csv(f"results/{fname}", index=False)