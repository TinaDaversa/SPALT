from src.LinearTreeBase import *
from src.utils import *
from src.SpatialLinearTree import SPALT
import time
import os
from pathlib import Path
import parsing_file



if __name__ == "__main__":

    parser = parsing_file.create_parser()
    args = parser.parse_args()
    #conf = Path(*Path(args.data).parts[-2:])
    conf = '_'.join(str(Path(*Path(args.data).parts[-2:])).split("\\"))
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    if not os.path.exists(args.output + "./model"):
        os.makedirs(args.output + "./model")

    X_train, y_train, X_test, y_test, X_valid, y_valid, pod, date_test =\
        train_test_valid(args.data, validation=True)

    distance_matrix = create_dist_matrix(X_train)

    Spalt = SPALT(X_train.drop(["lat", "long"], axis=1), y_train, X_test.drop(["lat", "long"], axis=1), y_test,
                  X_valid.drop(["lat", "long"], axis=1), y_valid, args.n_targets,
                  prune=True, spatial=True, drop_train_node=True, id_key=args.id_key, distance_matrix=distance_matrix)

    if args.load_model:
        print("Loading Model")
        regr = pickle.load(open(args.load_model, 'rb'))
        train_time = 0
    else:
        tr_time = time.time()
        regr, spatial_global, spatial_wins_attempts, wins_prune = Spalt.ModLinearTree()
        train_time = time.time() - tr_time
        print("Saving Model")
        pickle.dump(regr, open(f"{args.output}/model/{conf}_{str(args.dataset_name)}", 'wb'))

    distance_matrix = create_dist_matrix(X_test)
    ts_time = time.time()
    Spalt_real_pred = Spalt.predict(regr,distance_matrix)
    test_time = time.time() - ts_time
    writeResults(f"{args.output}{args.dataset_name}.csv", conf, Spalt_real_pred.iloc[:,1:], args.n_targets, np.round(train_time, 3), np.round(test_time, 3))
    np.savetxt(f"{args.output}{conf}_predictions.csv", Spalt_real_pred, fmt='%s', delimiter=',')
    if args.load_model == None:
        with open(f"{args.output}train_info_{args.dataset_name}.csv", mode='w') as f:
            f.write(f"{conf},{len(regr._nodes) + len(regr._leaves)},{spatial_global},{spatial_wins_attempts},{wins_prune}\n")

            

            














