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
    def __init__(self, data=None, seeds=[1], model_type='decision_tree', 
                 ganancia_acierto=273000, costo_estimulo=7000, 
                 threshold=0.025, seed=0, n_jobs=-1, reg=False,
                 meses_opt=1, meses_test=1):
        self.data = data
        self.seeds = seeds
        self.s = seed
        self.n_jobs = int(n_jobs)
        self.model_type = model_type
        self.ganancia_acierto = ganancia_acierto
        self.costo_estimulo = costo_estimulo
        self.threshold = threshold
        self.reg = reg
        self.meses_opt = meses_opt
        self.meses_test = meses_test
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
        if self.data == None:
            print("def_xy is not possible as data == None")
            print("You may perform X vs y separation out of the pipe")
            return
        else:
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
        total_gain = gains.sum()/(prop)

        return total_gain

    def train_and_evaluate(self, train_index, test_index, X, y, params, prop):
        # Instanciar el clasificador basado en model_type
        classifier_class = self.classifier_map[self.model_type]
        model = classifier_class(**params)
        model.fit(X.iloc[train_index], y.iloc[train_index])
        ganancia_value = self.ganancia(model, X.iloc[test_index], y.iloc[test_index], prop=prop)
        return model, ganancia_value

    def optimize_model(self, X, y, storage_name, study_name, test_size=0.3, optimize=True, n_trials=200):
        sss_opt = ShuffleSplit(n_splits=5, test_size=test_size, random_state=self.seeds[self.s])

        def objective_xgboost(trial):
            # Hiperparámetros para XGBClassifier

            # Parámetros a optimizar
            n_estimators = trial.suggest_int('n_estimators', 400, 800)
            max_leaves = trial.suggest_int('max_leaves', 100, 356)
            learning_rate = trial.suggest_float('eta', 0.015, 0.1, log=True)  # 'eta' es equivalente a 'learning_rate'
            min_child_weight = trial.suggest_int('min_child_weight', 7, 15)
            subsample = trial.suggest_float('subsample', 0.7, 0.95)
            colsample_bytree = trial.suggest_float('colsample_bytree', 0.2, 0.6)
            if self.reg:
                reg_lambda = trial.suggest_float('lambda', 0.0, 10.0)
                reg_alpha = trial.suggest_float('alpha', 0.0, 10.0)
            # scale_pos_weight = trial.suggest_float('scale_pos_weight', 1.0, 10.0)

            params = {
                'n_estimators': n_estimators,
                'max_leaves': max_leaves,
                'learning_rate': learning_rate,
                'min_child_weight': min_child_weight,
                'colsample_bytree': colsample_bytree,
                'subsample': subsample,
                # 'random_state': self.seeds[self.s], # Opt sin semilla para robustez
                'objective': 'multi:softprob',
                'num_class': 3,
                'eval_metric': 'mlogloss', 
                'booster': 'gbtree',
                'grow_policy': 'lossguide', # Necesario cuando se usa 'max_leaves'
                'tree_method': 'hist',      # Usar 'hist' para grandes conjuntos de datos
                'n_jobs': self.n_jobs,
            }

            if self.reg:
                params.update({
                    'reg_lambda': reg_lambda,  # 'lambda' es palabra reservada en Python, usamos 'reg_lambda'
                    'reg_alpha': reg_alpha,
                })

            # Ejecutar validación cruzada paralela
            results = Parallel(n_jobs=self.n_jobs)(
                delayed(self.train_and_evaluate)(train_index, test_index, X, y, params, test_size*self.meses_opt)
                for train_index, test_index in sss_opt.split(X, y)
            )

            # Retornar la ganancia media
            return np.mean([result[1] for result in results])

        def objective_lightgbm(trial):
            # Hiperparámetros para LGBMClassifier
            n_estimators = trial.suggest_int('n_estimators', 400, 800)
            num_leaves = trial.suggest_int('num_leaves', 20, 80)
            learning_rate = trial.suggest_float('learning_rate', 0.01, 0.1, log=True)
            min_data_in_leaf = trial.suggest_int('min_data_in_leaf', 80, 200)
            feature_fraction = trial.suggest_float('feature_fraction', 0.2, 0.6)
            if self.reg:
                lambda_l1 = trial.suggest_float('lambda_l1', 0.0, 10.0)
                lambda_l2 = trial.suggest_float('lambda_l2', 0.0, 10.0)

            params = {
                'n_estimators': n_estimators,
                'num_leaves': num_leaves,
                'learning_rate': learning_rate,
                'min_data_in_leaf': min_data_in_leaf,
                'feature_fraction': feature_fraction,
                'extra_trees': False,
                # 'random_state': self.seeds[self.s], # Opt sin semilla para robustez
                'n_jobs': self.n_jobs
            }

            if self.reg:
                params.update({
                    'lambda_l1': lambda_l1,
                    'lambda_l2': lambda_l2,
                })

            results = Parallel(n_jobs=self.n_jobs)(
                delayed(self.train_and_evaluate)(train_index, test_index, X, y, params, test_size*self.meses_opt)
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
        self.best_params.update({'random_state': self.seeds[self.s]}) # misma semilla que en el base
        classifier_class = self.classifier_map[self.model_type]
        self.best_model = classifier_class(**self.best_params)
        self.best_model.fit(X_train, y_train)

    def compare_models(self, X, y, test_size=0.3):
        sss = StratifiedShuffleSplit(n_splits=30, test_size=test_size, random_state=self.seeds[self.s])

        results_base = Parallel(n_jobs=self.n_jobs)(
            delayed(self.train_and_evaluate)(train_index, test_index, X, y, self.base_params, prop=test_size)
            for train_index, test_index in sss.split(X, y)
        )
        results_best = Parallel(n_jobs=self.n_jobs)(
            delayed(self.train_and_evaluate)(train_index, test_index, X, y, self.best_params, prop=test_size)
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
        ganancia_test = self.ganancia(model, X, y, prop=self.meses_test)
        print(f"Ganancia del modelo en el conjunto de test: {ganancia_test} por mes")
        print(f"Según un promedio de los {self.meses_test} meses evaluados")
        return ganancia_test

    def test_base_model(self, X, y):
        return self.test_model(self.base_model, X, y)

    def test_best_model(self, X, y):
        return self.test_model(self.best_model, X, y)

    def simulate_kaggle_split(self, X_futuro, y_futuro, test_size=0.3, imputer=None, to_drop=None):
        """
        Simula el split público/privado como en una competencia de Kaggle.
        """
        # Simular el split público/privado
        sss_futuro = StratifiedShuffleSplit(n_splits=50, test_size=test_size, random_state=self.seeds[self.s])

        gan_fut_priv_best = []
        gan_fut_priv_base = []
        gan_fut_pub_best = []
        gan_fut_pub_base = []

        for train_index, test_index in sss_futuro.split(X_futuro, y_futuro):
            # Privado (70% de los datos)
            gan_fut_priv_best.append(
                self.ganancia(self.best_model, X_futuro.iloc[train_index], y_futuro.iloc[train_index], prop=1-test_size)
            )
            gan_fut_priv_base.append(
                self.ganancia(self.base_model, X_futuro.iloc[train_index], y_futuro.iloc[train_index], prop=1-test_size)
            )
            # Público (30% de los datos)
            gan_fut_pub_best.append(
                self.ganancia(self.best_model, X_futuro.iloc[test_index], y_futuro.iloc[test_index], prop=test_size)
            )
            gan_fut_pub_base.append(
                self.ganancia(self.base_model, X_futuro.iloc[test_index], y_futuro.iloc[test_index], prop=test_size)
            )

        # Crear DataFrames para visualización
        df_pred_1_best = pd.DataFrame({
            'Ganancia': gan_fut_priv_best,
            'Modelo': 'Best',
            'Grupo': 'Privado'
        })
        df_pred_2_best = pd.DataFrame({
            'Ganancia': gan_fut_pub_best,
            'Modelo': 'Best',
            'Grupo': 'Publico'
        })
        df_pred_1_base = pd.DataFrame({
            'Ganancia': gan_fut_priv_base,
            'Modelo': 'Base',
            'Grupo': 'Privado'
        })
        df_pred_2_base = pd.DataFrame({
            'Ganancia': gan_fut_pub_base,
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

        return gan_fut_priv_best, \
                gan_fut_priv_base, \
                gan_fut_pub_best, \
                gan_fut_pub_base


def plot_comparisons_on_kaggle_split(name_model_a, results_a_priv, results_a_pub,
                     name_model_b, results_b_priv, results_b_pub, 
                     alfa=0.05):
    
    print(f"Comparando modelos: {name_model_a} vs. {name_model_b}")

    df_pred_a_priv = pd.DataFrame({'Ganancia': results_a_priv, 
                                   'Modelo': f'{name_model_a}_best',
                                   'Grupo': 'Privado'})
    df_pred_a_pub = pd.DataFrame({'Ganancia': results_a_pub, 
                                'Modelo': f'{name_model_a}_best',
                                'Grupo': 'Publico'})
    df_pred_b_priv = pd.DataFrame({'Ganancia': results_b_priv, 
                                   'Modelo': f'{name_model_b}_best',
                                   'Grupo': 'Privado'})
    df_pred_b_pub = pd.DataFrame({'Ganancia': results_b_pub, 
                                'Modelo': f'{name_model_b}_best',
                                'Grupo': 'Publico'})
    
    df_combined = pd.concat([df_pred_a_priv, df_pred_a_pub, df_pred_b_priv, df_pred_b_pub])

    # Gráfico de las distribuciones
    g = sns.FacetGrid(df_combined, col="Grupo", row="Modelo", aspect=2)
    g.map(sns.histplot, "Ganancia", kde=True)
    plt.show()

    # Cálculo de ganancias medias
    mean_pred_a_priv = df_combined[
        (df_combined['Modelo'] == f'{name_model_a}_best') & (df_combined['Grupo'] == 'Privado')
    ]['Ganancia'].mean()
    mean_pred_a_pub = df_combined[
        (df_combined['Modelo'] == f'{name_model_a}_best') & (df_combined['Grupo'] == 'Publico')
    ]['Ganancia'].mean()
    mean_pred_b_priv = df_combined[
        (df_combined['Modelo'] == f'{name_model_b}_best') & (df_combined['Grupo'] == 'Privado')
    ]['Ganancia'].mean()
    mean_pred_b_pub = df_combined[
        (df_combined['Modelo'] == f'{name_model_b}_best') & (df_combined['Grupo'] == 'Publico')
    ]['Ganancia'].mean()

    print(f"Ganancia media del modelo {name_model_a} privado: {mean_pred_a_priv}")
    print(f"Ganancia media del modelo {name_model_a} publico: {mean_pred_a_pub}")
    print(f"Ganancia media del modelo {name_model_b} privado: {mean_pred_b_priv}")
    print(f"Ganancia media del modelo {name_model_b} publico: {mean_pred_b_pub}")

    # Importar la función para el test estadístico
    from scipy.stats import mannwhitneyu

    # Realizar el test de Mann-Whitney U - Privado
    estadistico_u_priv, p_valor_priv = mannwhitneyu(
        df_pred_a_priv['Ganancia'], df_pred_b_priv['Ganancia'], alternative='less'
    )

    # Mostrar los resultados del test
    print(f"\nResultado del test estadístico Mann-Whitney U (privado):")
    print(f"Estadístico U = {estadistico_u_priv}")
    print(f"P-valor = {p_valor_priv}")

    # Interpretación del resultado
    if p_valor_priv < alfa:
        print(f"Rechazamos la hipótesis nula. Hay evidencia estadística de que la distribución de {name_model_b}_priv es mayor que la de {name_model_a}_priv.")
    else:
        print(f"No se rechaza la hipótesis nula. No hay evidencia suficiente para afirmar que la distribución de {name_model_b}_priv es mayor que la de {name_model_a}_priv.")


    # Realizar el test de Mann-Whitney U - Publico
    estadistico_u_pub, p_valor_pub = mannwhitneyu(
        df_pred_a_pub['Ganancia'], df_pred_b_pub['Ganancia'], alternative='less'
    )

    # Mostrar los resultados del test
    print(f"\nResultado del test estadístico Mann-Whitney U (privado):")
    print(f"Estadístico U = {estadistico_u_pub}")
    print(f"P-valor = {p_valor_pub}")

    # Interpretación del resultado
    if p_valor_priv < alfa:
        print(f"Rechazamos la hipótesis nula. Hay evidencia estadística de que la distribución de {name_model_b}_pub es mayor que la de {name_model_a}_pub.")
    else:
        print(f"No se rechaza la hipótesis nula. No hay evidencia suficiente para afirmar que la distribución de {name_model_b}_pub es mayor que la de {name_model_a}_pub.")


def analyze_study(db_path, study_name):
    # Load the study from the database
    study = optuna.load_study(study_name=study_name, storage=db_path)

    # Extract all trial data as a DataFrame
    df = study.trials_dataframe()
    print("\nStudy Trials DataFrame:\n")
    print(df.head(10).to_markdown())

    # Display basic statistics about the trials
    print("\nBasic Statistics of Trials:\n")
    print(df.describe().to_markdown())

    # Identify the best trial
    best_trial = study.best_trial
    print("\nBest Trial:\n")
    print(f"  Value: {best_trial.value}")
    print("  Params:")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")

    return study
