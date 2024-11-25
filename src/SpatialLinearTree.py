import logging
logging1 = logging.getLogger('SP')
from sklearn.linear_model import *
from .lineartree import LinearTreeRegressor
from .utils import *


class SPALT():
    def __init__(self, X_train, y_train, X_test, y_test, X_valid, y_valid, num_targets, prune=False, spatial=False, drop_train_node=False, id_key="", distance_matrix=None):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.X_valid = X_valid
        self.y_valid = y_valid
        self.prune = prune
        self.spatial = spatial
        self.drop_train_node = drop_train_node
        self.spatial_attempts_prune = 0
        self.spatial_wins_prune = 0
        self.spatial_leaf_count = 0
        self.wins_prune = 0
        self.num_targets = num_targets
        self.id_key = id_key
        self.distance_matrix = distance_matrix


    def check_node_spatial(self, Node, leaves, val_score, prune_leaves=None, weighted_error_leaves=None):
        if self.spatial:
            X_train_sp = add_features_closeness(self.X_train.iloc[Node.samples_id, :], self.X_train, self.distance_matrix)
            y_train_leaves = self.y_train.iloc[Node.samples_id, :]
            cols = [col for col in X_train_sp.columns if "right" in col]

        m = LinearRegression()
        X_train_sp = X_train_sp.reindex(X_train_sp.columns, axis=1)
        m.fit(X_train_sp, y_train_leaves)
        coeff = m.coef_
        intercept = m.intercept_


        #filtering the validation set to include only the examples that have fallen into the specific leaf being considered
        if prune_leaves:
            X_val = self.X_valid.iloc[leaves.loc[lambda x: x.isin(prune_leaves)].index, :].copy()
            y_val = self.y_valid.iloc[leaves.loc[lambda x: x.isin(prune_leaves)].index, :].copy()

            # If pruning is being performed, a linear regression model is fit and tested for the parent node
            model_p = LinearRegression()
            model_p.fit(self.X_train.iloc[Node.samples_id, :].drop(["date", self.id_key], axis=1,errors='ignore'), self.y_train.iloc[Node.samples_id, :])
            coeff_p = model_p.coef_
            intercept_p = model_p.intercept_

            predValidParent = model_p.predict(X_val.drop(["date", self.id_key], axis=1,errors='ignore'))
            realpredParent = pd.concat([y_val.reset_index(), pd.DataFrame(predValidParent).round(3)], axis=1).iloc[:, 1:]
            rseValidParent, _, _ = calculateErrors(realpredParent, self.num_targets)
            rseValidParent = np.round(np.mean(rseValidParent), 3)

        else:
            X_val = self.X_valid.iloc[leaves.loc[lambda x: x == Node.id].index, :].copy()
            y_val = self.y_valid.iloc[leaves.loc[lambda x: x == Node.id].index, :].copy()

        if self.spatial:
            X_valid_sp = add_features_closeness(X_val, self.X_valid, self.distance_matrix)
            predValid = m.predict(X_valid_sp)

        real_pred_valid = pd.concat([y_val.reset_index(), pd.DataFrame(predValid).round(3)], axis=1).iloc[:, 1:]
        real_pred_valid["leaf"] = Node.id

        val_score_spatial = real_pred_valid.groupby("leaf", group_keys=False).apply(apply_score)[["rse"]].drop_duplicates()

        pruning = 0
        if prune_leaves:
            rse_check = rseValidParent
            logging1.info(f'NO {self.spatial,rse_check} - {self.spatial, np.round(val_score_spatial["rse"].values[0], 3)}')
            logging1.info(f'WEIGHTED ERROR {np.round(weighted_error_leaves,3)}')

            if (weighted_error_leaves > np.round(val_score_spatial["rse"].values[0], 3)) | (weighted_error_leaves > rse_check):
                pruning = 1
                self.spatial_attempts_prune += 1

                if (val_score_spatial["rse"].values < rse_check):
                    Node.model.coef_ = coeff
                    Node.model.intercept_ = intercept
                    Node.model.n_features_in_ = coeff.shape[1]
                    Node.spatial_info = X_train_sp[cols].copy()

                    self.spatial_wins_prune += 1
                    # update val_score
                    val_score = val_score.drop(val_score[val_score.leaf.isin(prune_leaves)].index, axis=0)
                    try:
                        val_score.loc[val_score.index[-1] +1] = [Node.id, val_score_spatial["rse"].values[0]]
                    except:
                        val_score.loc[0] = [Node.id, val_score_spatial["rse"].values[0]]

                else:
                    Node.model.coef_ = coeff_p
                    Node.model.intercept_ = intercept_p
                    Node.model.n_features_in_ = coeff_p.shape[1]
                    Node.spatial_info=None
                    self.wins_prune += 1

                    # update val_score
                    val_score = val_score.drop(val_score[val_score.leaf.isin(prune_leaves)].index, axis=0)
                    try:
                        val_score.loc[val_score.index[-1] +1] = [Node.id, rse_check]
                    except:
                        val_score.loc[0] = [Node.id, rse_check]

            else:
                Node.check_prune = True


        else:
            pruning = 0
            logging1.info(f'NO SPATIAL {np.round(val_score[val_score["leaf"] == Node.id]["rse"].values[0], 3)} '
                         f'- {self.spatial, np.round(val_score_spatial["rse"].values[0], 3)}')
            rse_check = val_score[val_score["leaf"] == Node.id]["rse"].values

            if (val_score_spatial["rse"].values < rse_check):
                Node.model.coef_ = coeff
                Node.model.intercept_ = intercept
                Node.model.n_features_in_ = coeff.shape[1]
                Node.spatial_info = X_train_sp[cols].copy()
                self.spatial_leaf_count += 1

                # update val_score
                val_score.loc[val_score.set_index("leaf").index == Node.id, "rse"] = val_score_spatial["rse"].values[0]

        logging1.info(f"Num coeff: {Node.model.coef_.shape}")
        logging1.info("-------")

        return pruning, val_score

    def pruning(self, regr, leaves,real_pred_leaves, val_score):

        leaves_parents = dict()
        for L in regr._leaves.values():
            if self.drop_train_node == True:
                # For each parent node, the leaf child nodes are obtained regardless of the validation set
                if L.parent in regr._nodes  and regr._nodes[L.parent].check_prune == False:
                    if regr._nodes[L.parent].id in leaves_parents.keys():
                        leaves_parents[regr._nodes[L.parent].id].append(L.id)
                    else:
                        leaves_parents[regr._nodes[L.parent].id] = [L.id]
            else:
                if L.parent in regr._nodes and regr._nodes[L.parent].check_prune == False and L.id in leaves.values:

                    if regr._nodes[L.parent].id in leaves_parents.keys():
                        leaves_parents[regr._nodes[L.parent].id].append(L.id)
                    else:
                        leaves_parents[regr._nodes[L.parent].id] = [L.id]

        # Only parent nodes with two leaf children are considered
        leaves_parents = {k: v for k, v in leaves_parents.items() if len(v) == 2}

        if leaves_parents:
            leaf_samples = real_pred_leaves.leaf.value_counts()
            parents_samples = {k:real_pred_leaves.leaf.value_counts()[leaf_samples.index.isin(v)].sum() for k,v in leaves_parents.items()}

            continue_pruning = 0
            for parent, leave in leaves_parents.items():
                parent_node = [k for k, node in regr._nodes.items() if node.id == parent][0]
                leaves_node = [k for k, node in regr._leaves.items() if node.id in leave]

                # Check if instances from the validation set have fallen into one or both of the leaf nodes
                if set(leave).intersection(set(leaf_samples.index)):
                    #compute the weigthed sum of children's nodes errors
                    weighted_sum = 0
                    for leaf in leave:
                        if (leaf in leaf_samples.index):
                            weight = (leaf_samples[leaf_samples.index==leaf]/parents_samples[parent]).squeeze()
                            score = val_score.loc[val_score.set_index("leaf").index == leaf].rse.squeeze()
                        else:
                            weight=0
                            score=0

                        new_score = score * weight
                        weighted_sum += new_score


                    if self.spatial!=False:
                        pruning, val_score= self.check_node_spatial(regr._nodes[parent_node], leaves, val_score, leave, weighted_sum)

                    else:
                        pruning, val_score = self.check_node_prune(regr, regr._nodes[parent_node], leaves, val_score,leave, weighted_sum)

                    if pruning == 1:

                        # The parent node becomes a leaf, and the old leaf nodes are removed
                        regr._leaves[parent_node] = regr._nodes[parent_node]
                        del regr._nodes[parent_node]


                        for L in leaves_node:
                            del regr._leaves[L]

                        continue_pruning = 1
                else:

                    if self.drop_train_node == True:
                        logging1.info("no instances from the validation set have fallen into both leaf nodes")
                        # If no instances from the validation set have fallen into both leaf nodes, they must be dropped
                        regr._leaves[parent_node] = regr._nodes[parent_node]
                        del regr._nodes[parent_node]

                        for L in leaves_node:
                            del regr._leaves[L]

                        continue_pruning = 1


            if (continue_pruning == 1):
                return self.pruning(regr, leaves, real_pred_leaves, val_score)
        else:
            logging1.info("There are no other leaves")



    def check_node_prune(self, regressor, Node, leaves, val_score, prune_leaves=None, weighted_error_leaves=None):

        #filtering the validation set to include only the examples that have fallen into the specific leaf being considered
        if prune_leaves:
            X_val = self.X_valid.iloc[leaves.loc[lambda x: x.isin(prune_leaves)].index, :].drop(["date", self.id_key], axis=1, errors='ignore').copy()
            y_val = self.y_valid.iloc[leaves.loc[lambda x: x.isin(prune_leaves)].index, :].copy()

            model_p = LinearRegression()
            model_p.fit(self.X_train.iloc[Node.samples_id, :].drop(["date", self.id_key], axis=1, errors='ignore'), self.y_train.iloc[Node.samples_id, :])
            coeff_p = model_p.coef_
            intercept_p = model_p.intercept_

            predValidParent = model_p.predict(X_val)
            realpredParent = pd.concat([y_val.reset_index(), pd.DataFrame(predValidParent).round(3)], axis=1).iloc[:, 1:]
            rseValidParent, _ , _ = calculateErrors(realpredParent, self.num_targets)
            rseValidParent = np.round(np.mean(rseValidParent), 3)

        else:
            X_val = self.X_valid.iloc[leaves.loc[lambda x: x == Node.id].index, :].drop(["date", self.id_key], axis=1).copy()
            y_val = self.y_valid.iloc[leaves.loc[lambda x: x == Node.id].index, :].copy()


        predValid = regressor.predict(X_val)

        real_pred_valid = pd.concat([y_val.reset_index(), pd.DataFrame(predValid).round(3)], axis=1).iloc[:, 1:]
        real_pred_valid["leaf"] = Node.id


        pruning = 0
        if prune_leaves:
            rse_check = rseValidParent
            if (weighted_error_leaves > rse_check):
                pruning = 1
                self.wins_prune+=1
                Node.model.coef_ = coeff_p
                Node.model.intercept_ = intercept_p
                Node.model.n_features_in_ = coeff_p.shape[1]
                Node.spatial_info=None
                # update val_score
                val_score = val_score.drop(val_score[val_score.leaf.isin(prune_leaves)].index, axis=0)
                try:
                    val_score.loc[val_score.index[-1] +1] = [Node.id, rse_check]
                except:
                    val_score.loc[0] = [Node.id, rse_check]
            else:
                Node.check_prune = True
        else:
            pruning = 0

        return pruning, val_score

    def ModLinearTree(self):
        regr = LinearTreeRegressor(LinearRegression())
        regr.set_params(**{"min_samples_leaf":0.05, "n_jobs":-1})
        regr.fit(self.X_train.drop(["date",self.id_key], axis=1,errors='ignore'), self.y_train)

        # For each node, save the indices of instances that fell into that node in 'samples_id'
        save_index_sample(regr, self.X_train.drop(["date",self.id_key], axis=1,errors='ignore'))

        y_pred = regr.predict(self.X_valid.drop(["date",self.id_key], axis=1,errors='ignore'))

        # Return the index of the leaf that each sample of Validation is predicted as
        leaves = pd.Series(regr.apply(self.X_valid.drop(["date",self.id_key], axis=1,errors='ignore')), name='leaf')

        real_pred = pd.concat([self.y_valid.reset_index(), pd.DataFrame(y_pred).round(3)], axis=1)
        real_pred_leaves = pd.concat([real_pred, leaves], axis=1).iloc[:, 1:]

        # compute error for each leaf node
        val_score = real_pred_leaves.groupby("leaf", group_keys=False).apply(apply_score)[
            ["leaf", "rse"]].drop_duplicates()

        if self.spatial != False:
            # Compute spatial features for the training data only on leaves where a validation instance has fallen.
            # Save the lat,long + spatial feat structure in the "spatial_info" attribute of the Node class.
            for L in regr._leaves.values():
                if L.id in val_score.leaf.values:
                    _, val_score = self.check_node_spatial(L, leaves, val_score)

            print(f"Leaves with spatial features {self.spatial_leaf_count}/{len(val_score.leaf.unique())}")

        elif self.prune == True:
            print("Pruning")
            for L in regr._leaves.values():
                if L.id in val_score.leaf.values:
                    _, val_score = self.check_node_prune(regr,L, leaves, val_score)

        # def pruning
        if self.prune==True:
            print("PRUNING")
            self.pruning(regr, leaves, real_pred_leaves, val_score)

        spatial_count = 0
        for L in regr._leaves.values():
            if L.spatial_info is not None:
                spatial_count += 1

        spatial_global = [spatial_count, len(regr._leaves), (spatial_count/ len(regr._leaves))*100]

        global_wins = self.spatial_leaf_count + self.spatial_wins_prune
        global_attempts = len(val_score.leaf.unique()) + self.spatial_attempts_prune

        spatial_wins_attempts = [self.spatial_leaf_count, self.spatial_wins_prune,
                              len(val_score.leaf.unique()), self.spatial_attempts_prune,(global_wins/global_attempts) * 100]
        return regr, spatial_global, spatial_wins_attempts, self.wins_prune

    def predict(self, regr, distance_matrix):
        leaves_test = pd.Series(regr.apply(self.X_test.drop(["date",self.id_key], axis=1,errors='ignore')), name='leaf')
        real_pred_test = pd.DataFrame()
        for L in regr._leaves.values():
            if L.id in leaves_test.unique():
                X_test_m = self.X_test.iloc[leaves_test.loc[lambda x: x == L.id].index, :].copy()
                y_test_m = self.y_test.iloc[leaves_test.loc[lambda x: x == L.id].index, :].copy()

                if L.spatial_info is None:
                    y_test_pred = L.model.predict(X_test_m.drop(["date", self.id_key], axis=1,errors='ignore').values)
                elif self.spatial:
                    X_test_sp = add_features_closeness(X_test_m, self.X_test,distance_matrix)
                    y_test_pred = L.model.predict(X_test_sp.values)
                real_pred = pd.concat([y_test_m.reset_index(), pd.DataFrame(y_test_pred).round(3)], axis=1)
                real_pred_test = pd.concat([real_pred_test,real_pred], axis=0, ignore_index=True)

        return real_pred_test





