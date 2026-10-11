import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (confusion_matrix, classification_report,accuracy_score, precision_score, recall_score, f1_score)
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder
import category_encoders as ce
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')
os.environ["LOKY_MAX_CPU_COUNT"] = "1"


class SleepAgent:
    """
    We made this agent to manually execute the complete pipline of protocol 2
    It decides  if duplicates exists and if to drop them, which features have Mi score =0 
    and delete them and which model is more performant

    """

    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        self.df = None
        self.X = None
        self.y = None
        self.results = {}

        # categorial columns of the database
        self.categorical_cols = [
            'gender', 'occupation', 'country',
            'chronotype', 'mental_health_condition', 'season', 'day_type'
        ]

    #structered display                                 #
    def _log(self, step: str, message: str):
        print(f"[{step}] {message}")

    def _separator(self, label: str = ""):
        print("\n" + "=" * 60)
        if label:
            print(f"  {label}")
            print("=" * 60)

    # step 1 is to import data                                            
    def load_data(self):
        self._separator("step 1 getting data")
        self.df = pd.read_csv(self.dataset_path)
        self._log("LOAD", f"Dataset chargé : {self.df.shape[0]} lignes, {self.df.shape[1]} colonnes")
        self._log("LOAD", f"Colonnes : {list(self.df.columns)}")

    #  step 2 Binarization of the target variable                        
    def binarize_target(self):
        self._separator("step 2 Binarization of the target variable")
        self.df['sleep_disorder_risk'] = self.df['sleep_disorder_risk'].apply(
            lambda x: 'Healthy' if x == 'Healthy' else 'At Risk')
        dist = self.df['sleep_disorder_risk'].value_counts()
        self._log("TARGET", f"binary distribution :\n{dist}")

        ratio = dist.min() / dist.max()
        if ratio < 0.7:
            self._log("TARGET", f"[Decision] detected imbalance (ratio={ratio:.2f}) → SMOTE will be applied")
        else:
            self._log("TARGET", f"class is balanced (ratio={ratio:.2f})")

    # step 3 preprocessing                                            #
    def preprocess(self):
        self._separator("step 3 preprocessing")

        # duplicates
        before = self.df.shape[0]
        self.df = self.df.drop_duplicates()
        removed = before - self.df.shape[0]
        self._log("DEDUPLICATION", f"{removed} dupplicates dropped, remaining lines : {self.df.shape[0]}")

        # ID column useless
        if 'person_id' in self.df.columns:
            self.df = self.df.drop('person_id', axis=1)
            self._log("DROP", "column 'person_id' dropped (id is useless)")

        # missing values
        missing_pct = self.df.isnull().mean() * 100
        total_missing = self.df.isnull().sum().sum()
        self._log("MISSING", f"Total missing vals : {total_missing}")
        if total_missing > 0:
            self._log("MISSING", f"[Decision] columns detecteds :\n{missing_pct[missing_pct > 0]}")
        else:
            self._log("MISSING", "no missing vzlues, no action needed")

    #  seperate features and target                            
    def split_features_target(self):
        self._separator("step 4 split features /target")
        self.X = self.df.drop('sleep_disorder_risk', axis=1)
        self.y = self.df['sleep_disorder_risk']
        self._log("SPLIT", f"X : {self.X.shape}  |  y : {self.y.shape}")

    # step 5 encoding categorial columns                                   #
    def encode_categorical(self):
        self._separator("step 5 encoding categorial columns ")
        present_cats = [c for c in self.categorical_cols if c in self.X.columns]
        self._log("ENCODE", f"encoded columns : {present_cats}")
        encoder = ce.OrdinalEncoder(cols=present_cats, drop_invariant=True)
        self.X = encoder.fit_transform(self.X)
        self._log("ENCODE", f"X after encoding : {self.X.shape}")

    #  step 6 select features via Mutual Information             
    def select_features(self):
        self._separator("step 6 select features via Mutual Information  ")
        le = LabelEncoder()
        y_encoded = le.fit_transform(self.y)

        scores = mutual_info_classif(self.X, y_encoded, random_state=42)
        mi_series = pd.Series(scores, index=self.X.columns).sort_values(ascending=False)

        self._log("MI", f"MI scores :\n{mi_series.round(4).to_string()}")

        zero_features = mi_series[mi_series == 0.0].index.tolist()
        if zero_features:
            self._log("MI", f"[Decision] {len(zero_features)} deleted features (MI=0) : {zero_features}")
            self.X = self.X.drop(columns=zero_features)
        else:
            self._log("MI", "[decision] all features are informative so none deleted")

        self._log("MI", f"features kept({self.X.shape[1]}) : {self.X.columns.tolist()}")

    # step 7 training and evaluation with 10 iterations
    def train_and_evaluate(self):
        self._separator("step 7 training and evaluation with 10 iterations")

        models = {
            'RandomForest': RandomForestClassifier,
            'KNN':          KNeighborsClassifier,
        }
        model_params = {
            'RandomForest': {},
            'KNN':          {'n_neighbors': 3},
        }

        for name, ModelClass in models.items():
            self._separator(f"Model : {name}")
            errors, precisions, recalls, f1s = [], [], [], []

            for i in range(10):
                # Split
                X_train, X_test, y_train, y_test = train_test_split(
                    self.X, self.y, test_size=0.2, random_state=i
                )

                # Standardisation
                scaler = StandardScaler()
                X_train_s = pd.DataFrame(scaler.fit_transform(X_train), columns=self.X.columns)
                X_test_s  = pd.DataFrame(scaler.transform(X_test),      columns=self.X.columns)

                # SMOTE
                sm = SMOTE(random_state=i)
                X_train_s, y_train = sm.fit_resample(X_train_s, y_train)

                # training
                model = ModelClass(**model_params[name], random_state=i) \
                        if name == 'RandomForest' \
                        else ModelClass(**model_params[name])
                model.fit(X_train_s, y_train)

                # prediction
                y_pred = model.predict(X_test_s)

                # confusion matrix
                cm = confusion_matrix(y_test, y_pred, labels=['Healthy', 'At Risk'])
                print(f"\n--- Iteration {i+1} [{name}] ---")
                print(f"Confusion matrix:\n{cm}")
                print(f"  TP={cm[1,1]}  TN={cm[0,0]}  FP={cm[0,1]}  FN={cm[1,0]}")
                print(classification_report(y_test, y_pred))

                # Metrics
                acc = accuracy_score(y_test, y_pred)
                errors.append(1 - acc)
                precisions.append(precision_score(y_test, y_pred, pos_label='At Risk'))
                recalls.append(recall_score(y_test, y_pred,pos_label='At Risk'))
                f1s.append(f1_score(y_test, y_pred,pos_label='At Risk'))

            # avg of 10 iterations
            self.results[name] = {
                'error':     np.mean(errors),
                'precision': np.mean(precisions),
                'recall':    np.mean(recalls),
                'f1':        np.mean(f1s),
            }
            print(f"\n  average 10 iterations — {name}")
            print(f"  {'Error Rate':<15} {np.mean(errors):>8.4f}")
            print(f"  {'Precision':<15} {np.mean(precisions):>8.4f}")
            print(f"  {'Recall':<15} {np.mean(recalls):>8.4f}")
            print(f"  {'F-Measure':<15} {np.mean(f1s):>8.4f}")

    
    # step 8 final report with agent decision
    def final_report(self):
        self._separator("step 8 final report with agent decision")

        rf  = self.results.get('RandomForest', {})
        knn = self.results.get('KNN', {})

        print(f"\n  {'sleep_health_dataset':<25} {'Protocol 2 with preprocessing':^55}")
        print("  " + "-" * 80)
        print(f"  {'':25} {'Error Rate':>13} {'Precision':>13} {'Recall':>13} {'F-Measure':>13}")
        print("  " + "-" * 80)
        print(f"  {'RandomForestClassifier':<25} {rf['error']:>13.4f} {rf['precision']:>13.4f} {rf['recall']:>13.4f} {rf['f1']:>13.4f}")
        print(f"  {'KNeighborsClassifier':<25} {knn['error']:>13.4f} {knn['precision']:>13.4f} {knn['recall']:>13.4f} {knn['f1']:>13.4f}")
        print("  " + "-" * 80)

        # autonomous decision of agent
        best = max(self.results, key=lambda m: self.results[m]['f1'])
        worst = min(self.results, key=lambda m: self.results[m]['f1'])
        delta = self.results[best]['f1'] - self.results[worst]['f1']

        print(f"\n  [DeCISION AGENT] best model : {best}")
        print(f"  Gain F1 vs {worst} : +{delta:.4f}")

        if self.results[best]['f1'] >= 0.90:
            print("  [AGENT] Excellent performance (F1 >= 0.90) pipeline validated")
        elif self.results[best]['f1'] >= 0.80:
            print("  [AGENT] Acceptable performance may use some hyperparameters tuning ")
        else:
            print("  [AGENT] insufissinat performance redo feature selection and preprocessing")

    # complete autonomous execution
    def run(self):
        print("\n" + "=" * 60)
        print(" INTELLIGENT AGENT SLEEP HEALTH PIPELINE (Protocol 2)")
        print("=" * 60)

        self.load_data()
        self.binarize_target()
        self.preprocess()
        self.split_features_target()
        self.encode_categorical()
        self.select_features()
        self.train_and_evaluate()
        self.final_report()

        print("\n[AGENT] Pipeline finished successfully\n")



#  launch                                                      #
if __name__ == '__main__':
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    dataset  = os.path.join(BASE_DIR, 'sleep_health_dataset.csv')

    agent = SleepAgent(dataset_path=dataset)
    agent.run()