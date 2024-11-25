import pandas as pd
import numpy as np
import polars as pl
from fastparquet import ParquetFile
from random import seed,sample
from datetime import timedelta
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name' ,type=str ,default='ukpv' ,help='dataset name')
parser.add_argument('--data' ,type=str ,default="prepare_data/5min.parquet" ,help='data path')
parser.add_argument('--locations' ,type=str ,default="prepare_data/metadata.csv" ,help='locations file path')
parser.add_argument('--n_targets' ,type=int ,default=6 ,help='number of target time-steps')
parser.add_argument('--n_hist_steps' ,type=int ,default=12 ,help='number of historical features time-steps')
parser.add_argument('--size_train', type=int, default=4, help='size train in terms of days')
parser.add_argument('--size_test', type=int, default=1, help='size test in terms of days')
parser.add_argument('--targetcol', type=str, default="energy_target", help='name column target')
parser.add_argument('--id_key', type=str, default="ss_id", help='name id key')
args = parser.parse_args()

def add_features(df, key, window_size, features, type='lag'):
    if type == 'lag':
        for i in np.arange(1, window_size + 1):
            for feature in features:
                df = df.with_columns(
                    pl.col(feature).shift(i).over(key).alias(feature.replace("_target", "") + '_' + str(i)))
    if type == 'lead':
        for i in np.arange(1, window_size + 1):
            for feature in features:
                df = df.with_columns(pl.col(feature).shift(-(i - 1)).over(key).alias(feature + '_' + str(i)))
    return df

def prep_data_ukpv(args):
    print("preprocessing")
    pf = ParquetFile(args.data)
    df = pf.to_pandas()
    not_rel_g = df.groupby(args.id_key).resample('5min', on="timestamp").mean().drop(args.id_key, axis=1).reset_index()
    not_rel_g['timestamp'] = not_rel_g['timestamp'].dt.tz_localize(None)
    not_rel_g[['generation_wh']] = not_rel_g.groupby([args.id_key])[['generation_wh']].apply(lambda x: x.ffill().bfill()).reset_index(drop=True)
    not_rel_g['year'] = pd.to_datetime(not_rel_g['timestamp']).dt.year
    metadata = pl.read_csv(args.locations, columns=[0,1,2])
    not_rel_g = pl.from_pandas(not_rel_g).join(metadata ,on="ss_id")
    filtered_data = not_rel_g[[args.id_key,"year"]].unique().group_by("ss_id").len()
    new_df = not_rel_g.filter(pl.col(args.id_key).is_in(filtered_data.filter(pl.col("len") > 3)[args.id_key]))
    test_group = new_df.filter(pl.col("generation_wh")==0).group_by("ss_id").len()
    test_group = test_group.join(new_df.group_by("ss_id").len(), on="ss_id")
    take_plants = test_group.filter(pl.col("len") < float(0.50) * pl.col("len_right"))["ss_id"]
    not_rel = not_rel_g.filter(pl.col(args.id_key).is_in(take_plants))
    not_rel = not_rel.rename({"generation_wh": "energy_target", "timestamp":"date","latitude_rounded":"lat","longitude_rounded":"long"})
    #create column day to split randomly
    min_date = not_rel["date"].min().date()
    data = not_rel.with_columns(((pl.col("date").cast(pl.Date) - pl.lit(min_date)).dt.total_days().alias("day"))).drop("year")
    data = data.filter((pl.col("day") > elem - args.size_train) & (pl.col("day") <= elem + args.size_test))
    seed(1)
    # select 8 days
    sequence = [i for i in np.arange(30,data["day"].max()-7)]
    subset = sample(sequence, 8)

    return subset,data

def prep_data_sdwpf(args):
    print("preprocessing")
    data = pd.read_csv(args.data)
    data.columns = [col.lower() for col in data.columns]
    data['start_time'] = data['day'].astype(str) + ' ' + data['tmstamp'].astype(str)
    data[['patv']] = data.groupby(['turbid'])[['patv']].apply(lambda x: x.ffill().bfill()).reset_index(drop=True)
    locations = pl.read_csv(args.locations, new_columns=[args.id_key, "lat", "long"])
    data = data[[args.id_key, "day", "tmstamp", "patv"]]
    data['date'] = pd.to_datetime('2022-01-01') + pd.to_timedelta(data['day'] - 1, unit='D')
    data['date'] = pd.to_datetime(data['date'].astype(str) + ' ' + data['tmstamp'])
    data = pl.from_pandas(data)
    data = data.rename({"patv": "patv_target"})
    data = data.join(locations, on="turbid").drop("tmstamp")
    seed(1)
    sequence = [i for i in np.arange(30, 245)]
    subset = sample(sequence, 8)

    return subset,data


if args.dataset_name == 'ukpv':
    subset, data = prep_data_ukpv(args)
else:
    subset, data = prep_data_sdwpf(args)

for i,elem in enumerate(subset):
    dataset = add_features(df=data, key=[args.id_key], window_size=args.n_hist_steps, features=[args.targetcol], type='lag')
    dataset = add_features(df=dataset, key=[args.id_key], window_size=args.n_targets, features=[args.targetcol], type='lead').drop_nulls()
    dataset = dataset.drop([args.targetcol])

    train = dataset.filter((pl.col("day") <= elem) & (pl.col("day") >= elem-args.size_train)).drop(["day"])
    valid = train.filter(pl.col("date") >= (train.select(pl.col("date")).max()[0, 0] - timedelta(days=args.size_test))).drop(["day"])
    test = dataset.filter((pl.col("day") > elem) & (pl.col("day") <= elem + args.size_test)).drop(["day"])

    outputpath = f"./data/{args.dataset_name}/T{args.n_targets}H{args.n_hist_steps}/fold{i}/"

    isExist = os.path.exists(outputpath)
    if not isExist:
        os.makedirs(outputpath)

    train.write_csv(outputpath + "train.csv")
    valid.write_csv(outputpath + "valid.csv")
    test.write_csv(outputpath + "test.csv")