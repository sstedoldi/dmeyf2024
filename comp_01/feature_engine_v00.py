import pandas as pd
import numpy as np

def create_transaction_features(df):
    # Tendencia de transacciones (diferencia entre trimestres)
    df['ctrx_trend'] = df.groupby('numero_de_cliente')['ctrx_quarter'].diff().fillna(0)
    
    # Consistencia de transacciones (desviación estándar a lo largo de trimestres)
    df['ctrx_std'] = df.groupby('numero_de_cliente')['ctrx_quarter'].transform('std').fillna(0)
    
    return df

def create_balance_features(df):
    # Saldos promedio
    df['saldo_promedio'] = df[['mcuentas_saldo', 'mcaja_ahorro', 'mcuenta_corriente']].mean(axis=1).fillna(0)
    
    # Volatilidad del saldo
    df['saldo_volatilidad'] = df[['mcuentas_saldo', 'mcaja_ahorro', 'mcuenta_corriente']].std(axis=1).fillna(0)
    
    # Saldo mínimo y máximo
    df['saldo_minimo'] = df[['mcuentas_saldo', 'mcaja_ahorro', 'mcuenta_corriente']].min(axis=1).fillna(0)
    df['saldo_maximo'] = df[['mcuentas_saldo', 'mcaja_ahorro', 'mcuenta_corriente']].max(axis=1).fillna(0)
    
    return df

def create_credit_card_features(df):
    # Ratio de uso de tarjeta de crédito
    df['Master_ratio_uso'] = (df['Master_msaldototal'] / df['Master_mfinanciacion_limite']).replace([np.inf, -np.inf], 0).fillna(0)
    df['Visa_ratio_uso'] = (df['Visa_msaldototal'] / df['Visa_mfinanciacion_limite']).replace([np.inf, -np.inf], 0).fillna(0)
    
    # Periodo activo de la tarjeta en días
    df['Master_periodo_activo'] = (pd.to_datetime(df['foto_mes'], format='%Y%m') - pd.to_datetime(df['Master_fechaalta'])).dt.days.fillna(0)
    df['Visa_periodo_activo'] = (pd.to_datetime(df['foto_mes'], format='%Y%m') - pd.to_datetime(df['Visa_fechaalta'])).dt.days.fillna(0)
    
    return df

def create_loan_features(df):
    # Ratio préstamo-saldo
    df['ratio_prestamo_saldo'] = (df['mprestamos_personales'] / df['mcuentas_saldo']).replace([np.inf, -np.inf], 0).fillna(0)
    
    # Tasa de pago del préstamo
    df['tasa_pago_prestamo'] = df.groupby('numero_de_cliente')['mprestamos_personales'].diff().fillna(0)
    
    return df

def create_payroll_features(df):
    # Consistencia de la nómina
    df['payroll_consistencia'] = df.groupby('numero_de_cliente')['mpayroll'].transform('std').fillna(0)
    
    # Crecimiento de la nómina
    df['payroll_crecimiento'] = df.groupby('numero_de_cliente')['mpayroll'].diff().fillna(0)
    
    return df

def create_age_experience_features(df):
    # Grupos de edad
    df['grupo_edad'] = pd.cut(df['cliente_edad'], bins=[0, 30, 50, 70, 100], labels=[0, 1, 2, 3]).fillna(0)

    # Antigüedad del cliente en días
    df['cliente_antiguedad_dias'] = (pd.to_datetime(df['foto_mes'], format='%Y%m') - pd.to_datetime(df['cliente_antiguedad'])).dt.days.fillna(0)
    
    return df

def create_profitability_features(df):
    # Tendencia de rentabilidad
    df['rentabilidad_trend'] = df.groupby('numero_de_cliente')['mrentabilidad_annual'].diff().fillna(0)
    
    # Ratio de rentabilidad
    df['ratio_rentabilidad'] = (df['mactivos_margen'] / df['mpasivos_margen']).replace([np.inf, -np.inf], 0).fillna(0)
    
    return df

def create_commission_features(df):
    # Comisiones totales
    df['comisiones_totales'] = df[['mcomisiones', 'mcomisiones_otras', 'mcomisiones_mantenimiento']].sum(axis=1).fillna(0)
    
    # Tendencia de comisiones
    df['comisiones_trend'] = df.groupby('numero_de_cliente')['comisiones_totales'].diff().fillna(0)
    
    return df

def engineer_features(df):
    df = create_transaction_features(df)
    df = create_balance_features(df)
    df = create_credit_card_features(df)
    df = create_loan_features(df)
    df = create_payroll_features(df)
    df = create_age_experience_features(df)
    df = create_profitability_features(df)
    df = create_commission_features(df)
    
    # Reemplazo de valores infinitos por cero en todo el dataframe como paso final
    df = df.replace([np.inf, -np.inf], 0)
    
    return df
