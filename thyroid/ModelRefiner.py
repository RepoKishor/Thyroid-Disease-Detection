from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics  import roc_auc_score,accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import os,sys
from thyroid.logger import logging
from thyroid.exception import ThyroidException


    # def __init__(self):
    #     try:
    #         logging.info(f"{'>>'*20} Find Model {'<<'*20}")
    #         self.clf = RandomForestClassifier()
    #         self.knn = KNeighborsClassifier()

    #     except Exception as e:
    #         raise ThyroidException(e, sys)

def get_best_model(X_train,y_train,X_test,y_test):
    
    logging.info(f"Finding the best model for the data")
    # Finding the best model for KNN
    try:
        knn = KNeighborsClassifier()
        param_grid_knn = {
            'algorithm' : ['ball_tree', 'kd_tree', 'brute'],
            'leaf_size' : [10,17,24,28,30,35],
            'n_neighbors':[4,5,8,10,11],
            'p':[1,2]
        }
        grid = GridSearchCV(knn, param_grid_knn, verbose=3,cv=5)
        grid.fit(X_train,y_train)
        algorithm = grid.best_params_['algorithm']
        leaf_size = grid.best_params_['leaf_size']
        n_neighbors = grid.best_params_['n_neighbors']
        p  = grid.best_params_['p']

        logging.info(f"The best params for KNN are :"+str(grid.best_params_))

        knn = KNeighborsClassifier(algorithm=algorithm, leaf_size=leaf_size, n_neighbors=n_neighbors,p=p,n_jobs=-1)
        knn.fit(X_train,y_train)
        prediction_knn = knn.predict(X_test)
        knn_score = accuracy_score(y_test, prediction_knn)
        logging.info(f"knn accuracy score : {knn_score}")

        # finding the best params for Random classifier
        rcf = RandomForestClassifier()
        param_grid = {"n_estimators": [10, 50, 100, 130], "criterion": ['gini', 'entropy'],
                            "max_depth": range(2, 4, 1), "max_features": ['auto', 'log2']}
        grid_rcf = GridSearchCV(estimator=rcf, param_grid=param_grid, cv=5, verbose=3)
        grid_rcf.fit(X_train,y_train)
        criterion = grid_rcf.best_params_['criterion']
        max_depth = grid_rcf.best_params_['max_depth']
        max_features = grid_rcf.best_params_['max_features']
        n_estimators = grid_rcf.best_params_['n_estimators']

        logging.info(f"The best params for Random Classifier are :"+str(grid_rcf.best_params_))

        rcf = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion,
                                            max_depth=max_depth, max_features=max_features)
        rcf.fit(X_train,y_train)
        prediction_random_forest=rcf.predict(X_test)
        random_forest_score = accuracy_score(y_test, prediction_random_forest)
        logging.info(f"random forest accuracy score : {random_forest_score}")


        #comparing the two model scores for accuracy
        if(random_forest_score <  knn_score):
            return 'KNN',knn
        else:
            return 'RandomForest',rcf

    except Exception as e:
        logging.info(f"Exception occured")
        raise ThyroidException(e, sys) from e
