import collections
from sklearn.metrics import root_mean_squared_error, r2_score, mean_absolute_error
import pandas as pd
import polars as pl
import numpy as np
import gc
from sklearn.metrics.pairwise import haversine_distances
import logging, sys
import parsing_file

parser = parsing_file.create_parser()
args = parser.parse_args()

def save_index_sample(regr, X_train):
    samples = collections.defaultdict(list)
    dec_paths = regr.decision_path(X_train)

    for d, dec in enumerate(dec_paths):
        for i in range(regr.node_count):
            if dec.toarray()[0][i] == 1:
                samples[i].append(d)

    for i in regr._leaves.values():
        i.samples_id = samples[i.id]

    for i in regr._nodes.values():
        i.samples_id = samples[i.id]


def apply_score(x):
    score = pd.Series([1-r2_score(x.iloc[:, i], x.iloc[:, i + args.n_targets]) for i in range(args.n_targets)])
    x["rse"] = score.mean()
    return x


def calculateErrors(df, num_targets):

    rmse = [root_mean_squared_error(df.iloc[:, i], df.iloc[:, i+num_targets]) for i in range(num_targets)]
    #mse = [mean_squared_error(df.iloc[:, i], df.iloc[:, i+num_targets], squared=True) for i in range(num_targets)]
    rse = [1-r2_score(df.iloc[:, i], df.iloc[:, i +num_targets]) for i in range(num_targets)]
    mae = [mean_absolute_error(df.iloc[:, i], df.iloc[:, i+num_targets]) for i in range(num_targets)]

    return rse, rmse, mae



def add_features_closeness(dataset, train, distance_matrix):
    dataset = dataset.copy()
    dataset.loc[:, ("samples_index")] = dataset.index
    dataset = pl.from_pandas(dataset)
    train = pl.from_pandas(train)

    cols = [c for c in dataset.columns if args.targetcol.split("_")[0] in c]
    cols_drop = [elem + "_right" for elem in set(train.columns) - set(cols) if
                elem not in [args.id_key, 'date', 'samples_index'] and "target" not in elem] + \
                [elem for elem in train.columns if "target" in elem]

    train_filtered = train.filter(pl.col(args.id_key).is_in(dataset[args.id_key].unique()))
    join_df = dataset.join(train_filtered, on="date").select(pl.all().exclude(cols_drop))

    join_close = join_df.join(distance_matrix, left_on=[args.id_key, args.id_key+"_right"], right_on=['point1', 'point2'])
    join_close = join_close.with_columns((1 - (pl.col("closeness") / join_close["closeness"].max())).alias("closeness"))
    del join_df

    # compute weigthed features
    cols_mul = [c for c in join_close.columns if args.targetcol.split("_")[0] in c and "right" in c]
    join_close = join_close.with_columns([(pl.col(c) * pl.col("closeness")).alias(c) for c in cols_mul])

    nearest_values = join_close.groupby([args.id_key, 'date']).agg([
        pl.sum(c).alias(c) for c in cols_mul] + [pl.sum("closeness").alias("sum_closeness")])

    #compute weigthed average
    for c in cols_mul:
        nearest_values = nearest_values.with_columns((pl.col(c) / pl.col("sum_closeness")).alias(c))

    nearest_values = nearest_values.drop("sum_closeness")
    join_close = join_close.drop(cols_mul + ["closeness", args.id_key + '_right']).unique()

    dataset = join_close.join(nearest_values, on=[args.id_key, 'date']).drop(['date', args.id_key]).to_pandas()
    dataset.index = dataset["samples_index"]
    dataset.index.name = None
    dataset.sort_index(inplace=True)
    dataset.drop("samples_index", axis=1, inplace=True)
    
    del nearest_values
    del join_close
    gc.collect()

    return dataset

def create_dist_matrix (data):
    coords = data[[args.id_key, "lat", "long"]].drop_duplicates().reset_index(drop=True)
    coords["lat"] = np.radians(coords["lat"])
    coords["long"] = np.radians(coords["long"])
    lat_long = coords.iloc[:, 1:]
    matrix = haversine_distances(lat_long)
    matrix_m = pd.DataFrame(matrix).apply(lambda x: x * 6371000 / 1000)
    closeness = matrix_m.unstack().reset_index()
    closeness.iloc[:, 0] = closeness.iloc[:, 0].map(coords[args.id_key].to_dict())
    closeness.iloc[:, 1] = closeness.iloc[:, 1].map(coords[args.id_key].to_dict())
    closeness.columns = ["point1", "point2", "closeness"]
    cl = closeness[closeness['point1'] != closeness['point2']].sort_values(["point1", "closeness", "point2"],
                                                                           ascending=False)
    return pl.from_pandas(cl)


def train_test_valid(path, validation=None):

    train = pd.read_csv(path + "train.csv", parse_dates=['date'])
    valid = pd.read_csv(path + "valid.csv", parse_dates=['date'])
    test = pd.read_csv(path + "test.csv", parse_dates=['date'])

    pod_test = test[args.id_key]
    date_test = test.date

    col_energy_target = [c for c in train.columns if args.targetcol in c]

    if (validation==None):
        train.drop([args.id_key, "date"], axis=1, inplace=True)
        print(f"Train size {train.shape}", end=" - ")
        print(f"Test size {test.shape}\n")
        X_train, y_train = train[train.columns.difference(col_energy_target)], \
                           train[col_energy_target]
        X_test, y_test = test[test.columns.difference(col_energy_target)], \
                         test[col_energy_target]

        return X_train, y_train, X_test, y_test, pod_test, date_test

    else:
        print(f"Train size {train.shape}", end=" - ")
        print(f"Validation size {valid.shape}", end=" - ")
        print(f"Test size {test.shape}\n")

        X_train, y_train = train[train.columns.difference(col_energy_target)], \
                           train[col_energy_target]
        X_valid, y_valid = valid[valid.columns.difference(col_energy_target)], \
                           valid[col_energy_target]
        X_test, y_test = test[test.columns.difference(col_energy_target)], \
                         test[col_energy_target]

        return X_train, y_train, X_test, y_test, X_valid, y_valid, pod_test, date_test



def setup_logger(logger_name, log_file, level=logging.INFO):
    l = logging.getLogger(logger_name)
    if not l.hasHandlers():
        formatter = logging.Formatter('%(message)s')
        fileHandler = logging.FileHandler(log_file, mode='w')
        fileHandler.setFormatter(formatter)
        l.setLevel(level)
        l.addHandler(fileHandler)
        l.addHandler(logging.StreamHandler(sys.stdout))
    return l


def writeResults(path, conf, realPred, num_targets, time_train, time_test):
    rse, rmse, mae = calculateErrors(realPred, int(num_targets))


    print(f"RSE : {np.round(np.mean(rse), 3)}")
    print(f"RMSE: {np.round(np.mean(rmse), 3)}")
    print(f"MAE: {np.round(np.mean(mae), 3)}")

    with open(path, mode='a') as f:
        f.write(f"{conf},{np.round(np.mean(rse), 3)},{np.round(np.mean(rmse), 3)},"
                f"{np.round(np.mean(mae), 3)},{time_train},{time_test}\n")




