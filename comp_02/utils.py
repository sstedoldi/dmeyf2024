import pandas as pd
import numpy as np

# psi function for data drifting prepro
def psi(expected, actual, buckets=10):

    def psi_formula(expected_prop, actual_prop):
        result = (actual_prop - expected_prop) * np.log(actual_prop / expected_prop)
        return result

    expected_not_null = expected.dropna()
    actual_not_null = actual.dropna()

    bin_edges = pd.qcut(expected_not_null, q=buckets, duplicates='drop').unique()
    bin_edges2 = [edge.left for edge in bin_edges] + [edge.right for edge in bin_edges]
    breakpoints = sorted(list(set(bin_edges2)))

    expected_counts, _ = np.histogram(expected_not_null, bins=breakpoints)
    actual_counts, _ = np.histogram(actual_not_null, bins=breakpoints)

    expected_prop = expected_counts / len(expected_not_null)
    actual_prop = actual_counts / len(actual_not_null)

    psi_not_null = psi_formula(expected_prop, actual_prop).sum()

    psi_null = 0

    if expected.isnull().sum() > 0 and actual.isnull().sum() > 0 :
      expected_null_percentage = expected.isnull().mean()
      actual_null_percentage = actual.isnull().mean()
      psi_null = psi_formula(expected_null_percentage, actual_null_percentage)

    return psi_not_null + psi_null

# Diferentes funciones y métodos para corregir el efecto de la inflación
def drift_uva(dataset, campos_monetarios, tb_indices):
    print("inicio drift_UVA()")
    dataset = dataset.merge(tb_indices[['foto_mes', 'UVA']], on='foto_mes', how='left')
    for campo in campos_monetarios:
        dataset[campo] *= dataset['UVA']
    dataset.drop(columns=['UVA'], inplace=True)
    print("fin drift_UVA()")
    return dataset

def drift_deflacion(dataset, campos_monetarios, tb_indices):
    print("inicio drift_deflacion()")
    dataset = dataset.merge(tb_indices[['foto_mes', 'IPC']], on='foto_mes', how='left')
    for campo in campos_monetarios:
        dataset[campo] *= dataset['IPC']
    dataset.drop(columns=['IPC'], inplace=True)
    print("fin drift_deflacion()")
    return dataset

# Función para estandarizar datos
def drift_estandarizar(dataset, campos_drift):
    print("inicio drift_estandarizar()")
    for campo in campos_drift:
        dataset[campo + "_normal"] = dataset.groupby('foto_mes')[campo].transform(lambda x: (x - x.mean()) / x.std())
        dataset.drop(columns=[campo], inplace=True)
    print("fin drift_estandarizar()")
    return dataset