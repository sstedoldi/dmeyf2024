# Importar las librerías necesarias
import numpy as np
import pandas as pd
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit
from joblib import Parallel, delayed
import seaborn as sns
import matplotlib.pyplot as plt
import optuna

# Definir la clase ModelPipeline
class ModelPipeline:
    def __init__(self, data, seeds, model_type='decision_tree', 
                 ganancia_acierto=273000, costo_estimulo=7000, 
                 threshold=0.025, seed=0, n_jobs=-1, reg=False):
        self.data = data
        self.seeds = seeds
        self.s = seed
        self.n_jobs = int(n_jobs)
        self.model_type = model_type
        self.ganancia_acierto = ganancia_acierto
        self.costo_estimulo = costo_estimulo
        self.threshold = threshold
        self.reg = reg
        self.models = {}
        self.base_params = {'random_state': self.seeds[self.s]}
        self.best_params = None
        self.base_model = None
        self.best_model = None

        # Mapear model_type al clasificador correspondiente
        self.classifier_map = {
            # 'decision_tree': DecisionTreeClassifier,
            # 'random_forest': RandomForestClassifier,
            'xgboost': XGBClassifier,
            'lightgbm': LGBMClassifier
        }

    def def_xy(self, meses, target='clase_ternaria', to_pred=False):
        # X = self.data[self.data['foto_mes'] == mes]
        X = self.data[self.data['foto_mes'].isin(meses)]
        y = X[target]
        X = X.drop(columns=[target])

        numero_de_cliente = X['numero_de_cliente']

        if to_pred:
            return X, numero_de_cliente
        else:
            return X, y

    def ganancia(self, model, X, y, prop=1):
        # Obtener las probabilidades predichas
        y_hat = model.predict_proba(X)

        # Obtener las clases del modelo
        model_classes = model.classes_

        # Identificar la clase objetivo (puede ser 'BAJA+2' o 2)
        if 'BAJA+2' in model_classes:
            target_class = 'BAJA+2'
        elif 2 in model_classes:
            target_class = 2
        else:
            raise ValueError("La clase objetivo 'BAJA+2' o 2 no está en las clases del modelo.")

        # Obtener el índice de la clase objetivo
        class_index = np.where(model_classes == target_class)[0][0]

        # Obtener las probabilidades predichas para la clase objetivo
        probs = y_hat[:, class_index]

        # Calcular la ganancia para cada fila
        gains = np.where(
            probs >= self.threshold,
            np.where(y == target_class, self.ganancia_acierto, -self.costo_estimulo),
            0
        )

        # Sumar las ganancias
        total_gain = gains.sum()/prop

        return total_gain

    def train_and_evaluate(self, train_index, test_index, X, y, params):
        # Instanciar el clasificador basado en model_type
        classifier_class = self.classifier_map[self.model_type]
        model = classifier_class(**params)
        model.fit(X.iloc[train_index], y.iloc[train_index])
        ganancia_value = self.ganancia(model, X.iloc[test_index], y.iloc[test_index], prop=0.3)
        return model, ganancia_value

    def optimize_model(self, X, y, storage_name, study_name, optimize=True, n_trials=200):
        sss_opt = ShuffleSplit(n_splits=5, test_size=0.3, random_state=self.seeds[self.s])

        def objective_xgboost(trial):
            # Hiperparámetros para XGBClassifier

            # Parámetros a optimizar
            n_estimators = trial.suggest_int('n_estimators', 100, 500)
            max_leaves = trial.suggest_int('max_leaves', 10, 256)
            learning_rate = trial.suggest_float('eta', 0.01, 0.3, log=True)  # 'eta' es equivalente a 'learning_rate'
            gamma = trial.suggest_float('gamma', 0, 5)
            min_child_weight = trial.suggest_int('min_child_weight', 1, 10)
            subsample = trial.suggest_float('subsample', 0.5, 0.9)
            colsample_bytree = trial.suggest_float('colsample_bytree', 0.5, 0.9)
            if self.reg:
                reg_lambda = trial.suggest_float('lambda', 0.0, 10.0)
                reg_alpha = trial.suggest_float('alpha', 0.0, 10.0)
            # scale_pos_weight = trial.suggest_float('scale_pos_weight', 1.0, 10.0)

            params = {
                'booster': 'gbtree',
                'n_estimators': 200,
                # 'n_estimators': n_estimators,
                'max_leaves': max_leaves,
                'learning_rate': learning_rate,
                'gamma': gamma,
                'min_child_weight': min_child_weight,
                'subsample': subsample,
                'colsample_bytree': colsample_bytree,
                # 'scale_pos_weight': scale_pos_weight, # default = 1, ya que la ganancia ya contempla desbalance
                'random_state': self.seeds[self.s],
                'enable_categorical': True,
                'use_label_encoder': False,
                'objective': 'multi:softprob',
                'num_class': 3,
                'eval_metric': 'mlogloss',
                'tree_method': 'hist',      # Usar 'hist' para grandes conjuntos de datos
                'grow_policy': 'lossguide', # Necesario cuando se usa 'max_leaves'
            }

            if self.reg:
                params.update({
                    'reg_lambda': reg_lambda,  # 'lambda' es palabra reservada en Python, usamos 'reg_lambda'
                    'reg_alpha': reg_alpha,
                })

            # Ejecutar validación cruzada paralela
            results = Parallel(n_jobs=self.n_jobs)(
                delayed(self.train_and_evaluate)(train_index, test_index, X, y, params)
                for train_index, test_index in sss_opt.split(X, y)
            )

            # Retornar la ganancia media
            return np.mean([result[1] for result in results])

        def objective_lightgbm(trial):
            # Hiperparámetros para LGBMClassifier
            n_estimators = trial.suggest_int('n_estimators', 50, 500)
            num_leaves = trial.suggest_int('num_leaves', 31, 256)
            learning_rate = trial.suggest_float('learning_rate', 0.001, 0.3, log=True)
            min_data_in_leaf = trial.suggest_int('min_data_in_leaf', 20, 100)
            if self.reg:
                lambda_l1 = trial.suggest_float('lambda_l1', 0.0, 10.0)
                lambda_l2 = trial.suggest_float('lambda_l2', 0.0, 10.0)
            min_gain_to_split = trial.suggest_float('min_gain_to_split', 0.0, 1.0)
            feature_fraction = trial.suggest_float('feature_fraction', 0.5, 0.9)
            bagging_fraction = trial.suggest_float('bagging_fraction', 0.5, 0.9)
            bagging_freq = trial.suggest_int('bagging_freq', 1, 7)
            max_bin = trial.suggest_int('max_bin', 64, 255)

            params = {
                'n_estimators': 200,
                # 'n_estimators': n_estimators,
                'num_leaves': num_leaves,
                'learning_rate': learning_rate,
                'min_data_in_leaf': min_data_in_leaf,
                'min_gain_to_split': min_gain_to_split,
                'feature_fraction': feature_fraction,
                'bagging_fraction': bagging_fraction,
                'bagging_freq': bagging_freq,
                'max_bin': max_bin,
                'random_state': self.seeds[self.s],
                'n_jobs': self.n_jobs
            }

            if self.reg:
                params.update({
                    'lambda_l1': lambda_l1,
                    'lambda_l2': lambda_l2,
                })

            results = Parallel(n_jobs=self.n_jobs)(
                delayed(self.train_and_evaluate)(train_index, test_index, X, y, params)
                for train_index, test_index in sss_opt.split(X)
            )

            return np.mean([result[1] for result in results])

        # Mapear model_type a la función objetivo correspondiente
        objective_map = {
            'xgboost': objective_xgboost,
            'lightgbm': objective_lightgbm
        }

        objective = objective_map[self.model_type]

        study = optuna.create_study(
            direction="maximize",
            study_name=study_name,
            storage=storage_name,
            load_if_exists=True
        )

        if optimize:
            print(f"Optimizando {self.model_type} con {n_trials} pruebas")
            study.optimize(objective, n_trials=n_trials)

        best_trial = study.best_trial
        self.best_params = best_trial.params  # Guardar los mejores parámetros

        print(f"Mejores parámetros para {self.model_type}: {best_trial.params}")
        return best_trial.params

    def train_base_model(self, X_train, y_train):
        classifier_class = self.classifier_map[self.model_type]
        self.base_model = classifier_class(**self.base_params)
        self.base_model.fit(X_train, y_train)

    def train_best_model(self, X_train, y_train):
        if self.best_params is None:
            print("No se encontraron mejores parámetros. Por favor, ejecuta optimize_model primero.")
            return
        classifier_class = self.classifier_map[self.model_type]
        self.best_model = classifier_class(**self.best_params)
        self.best_model.fit(X_train, y_train)

    def compare_models(self, X, y):
        sss = StratifiedShuffleSplit(n_splits=30, test_size=0.3, random_state=self.seeds[self.s])

        results_base = Parallel(n_jobs=self.n_jobs)(
            delayed(self.train_and_evaluate)(train_index, test_index, X, y, self.base_params)
            for train_index, test_index in sss.split(X, y)
        )
        results_best = Parallel(n_jobs=self.n_jobs)(
            delayed(self.train_and_evaluate)(train_index, test_index, X, y, self.best_params)
            for train_index, test_index in sss.split(X, y)
        )

        return results_base, results_best

    def plot_comparisons(self, results_base, results_best):
        df_pred = pd.DataFrame({'Ganancia': [result[1] for result in results_base], 'Modelo': 'Base'})
        df_pred_best = pd.DataFrame({'Ganancia': [result[1] for result in results_best], 'Modelo': 'Best'})
        df_combined = pd.concat([df_pred, df_pred_best])

        g = sns.FacetGrid(df_combined, row="Modelo", aspect=2)
        g.map(sns.histplot, "Ganancia", kde=True)
        plt.show()

        mean_base = df_combined[df_combined['Modelo'] == 'Base']['Ganancia'].mean()
        mean_best = df_combined[df_combined['Modelo'] == 'Best']['Ganancia'].mean()

        print(f"Ganancia media del modelo base: {mean_base}")
        print(f"Ganancia media del modelo Best: {mean_best}")

    def test_model(self, model, X, y):
        ganancia_test = self.ganancia(model, X, y)
        print(f"Ganancia del modelo en el conjunto de test: {ganancia_test}")
        return ganancia_test

    def test_base_model(self, X, y):
        return self.test_model(self.base_model, X, y)

    def test_best_model(self, X, y):
        return self.test_model(self.best_model, X, y)

    def simulate_kaggle_split(self, mes_futuro, imputer=None):
        """
        Simula el split público/privado como en una competencia de Kaggle.
        """
        # Obtener los datos futuros
        X_futuro, y_futuro = self.def_xy(mes_futuro, target='clase_ternaria', to_pred=False)
        if imputer is not None:
            X_futuro = pd.DataFrame(imputer.fit_transform(X_futuro), columns=X_futuro.columns)

        # Simular el split público/privado
        sss_futuro = StratifiedShuffleSplit(n_splits=30, test_size=0.3, random_state=self.seeds[self.s])

        ganancias_futuro_privada_best = []
        ganancias_futuro_privada_base = []
        ganancias_futuro_publica_best = []
        ganancias_futuro_publica_base = []

        for train_index, test_index in sss_futuro.split(X_futuro, y_futuro):
            # Privado (70% de los datos)
            ganancias_futuro_privada_best.append(
                self.ganancia(self.best_model, X_futuro.iloc[train_index], y_futuro.iloc[train_index], prop=0.7)
            )
            ganancias_futuro_privada_base.append(
                self.ganancia(self.base_model, X_futuro.iloc[train_index], y_futuro.iloc[train_index], prop=0.7)
            )
            # Público (30% de los datos)
            ganancias_futuro_publica_best.append(
                self.ganancia(self.best_model, X_futuro.iloc[test_index], y_futuro.iloc[test_index], prop=0.3)
            )
            ganancias_futuro_publica_base.append(
                self.ganancia(self.base_model, X_futuro.iloc[test_index], y_futuro.iloc[test_index], prop=0.3)
            )

        # Crear DataFrames para visualización
        df_pred_1_best = pd.DataFrame({
            'Ganancia': ganancias_futuro_privada_best,
            'Modelo': 'Best',
            'Grupo': 'Privado'
        })
        df_pred_2_best = pd.DataFrame({
            'Ganancia': ganancias_futuro_publica_best,
            'Modelo': 'Best',
            'Grupo': 'Publico'
        })
        df_pred_1_base = pd.DataFrame({
            'Ganancia': ganancias_futuro_privada_base,
            'Modelo': 'Base',
            'Grupo': 'Privado'
        })
        df_pred_2_base = pd.DataFrame({
            'Ganancia': ganancias_futuro_publica_base,
            'Modelo': 'Base',
            'Grupo': 'Publico'
        })

        df_combined = pd.concat([df_pred_1_base, df_pred_2_base, df_pred_1_best, df_pred_2_best])

        # Visualización
        g = sns.FacetGrid(df_combined, col="Grupo", row="Modelo", aspect=2)
        g.map(sns.histplot, "Ganancia", kde=True)
        plt.show()

        # Cálculo de ganancias medias
        mean_base_privado = df_combined[
            (df_combined['Modelo'] == 'Base') & (df_combined['Grupo'] == 'Privado')
        ]['Ganancia'].mean()
        mean_base_publico = df_combined[
            (df_combined['Modelo'] == 'Base') & (df_combined['Grupo'] == 'Publico')
        ]['Ganancia'].mean()
        mean_best_privado = df_combined[
            (df_combined['Modelo'] == 'Best') & (df_combined['Grupo'] == 'Privado')
        ]['Ganancia'].mean()
        mean_best_publico = df_combined[
            (df_combined['Modelo'] == 'Best') & (df_combined['Grupo'] == 'Publico')
        ]['Ganancia'].mean()

        print(f"Ganancia media del modelo base en privado: {mean_base_privado}")
        print(f"Ganancia media del modelo base en público: {mean_base_publico}")
        print(f"Ganancia media del modelo Best en privado: {mean_best_privado}")
        print(f"Ganancia media del modelo Best en público: {mean_best_publico}")