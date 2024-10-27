# Funciones de adaptación

def convert_to_float(df, columns):
    df[columns] = df[columns].astype(float)
    return df

def convert_to_int(df, columns):
    df[columns] = df[columns].fillna(0).astype(int)    
    return df

def cut_high_5(df, columns, quantile=0.95):
    for col in columns:
        high_cutoff = df[col].quantile(quantile)
        df = df[df[col] <= high_cutoff]
    return df

def cut_high_10(df, columns, quantile=0.9):
    for col in columns:
        high_cutoff = df[col].quantile(quantile)
        df = df[df[col] <= high_cutoff]
    return df

def cut_low_5(df, columns, quantile=0.05):
    for col in columns:
        low_cutoff = df[col].quantile(quantile)
        df = df[df[col] >= low_cutoff]
    return df

def cut_low_10(df, columns, quantile=0.1):
    for col in columns:
        low_cutoff = df[col].quantile(quantile)
        df = df[df[col] >= low_cutoff]
    return df

# Cut 5% of lower and higher samples
def cut_low_high(df, columns, low_quantile=0.05, high_quantile=0.95):
    for col in columns:
        low_cutoff = df[col].quantile(low_quantile)
        high_cutoff = df[col].quantile(high_quantile)
        df = df[(df[col] >= low_cutoff) & (df[col] <= high_cutoff)]
    return df

# Apply logarithmic transformation
def apply_log_transform(df, columns):
    for col in columns:
        df[col] = np.log1p(df[col])  # log1p is used to avoid issues with log(0)
    return df

def adapt_columns(df, to_float, to_int, to_cut_high_5,
                  to_cut_high_10, to_cut_low_5, to_cut_low_10, 
                  to_cut_low_high, to_log_transform=[]):
    
    df_ = df.copy()

    # Convert columns to float
    df_ = convert_to_float(df_, to_float)
    
    # Convert columns to integer
    df_ = convert_to_int(df_, to_int)
    
    # Cut 5% of higher samples
    df_ = cut_high_5(df_, to_cut_high_5)

    # Cut 10% of higher samples
    df_ = cut_high_10(df_, to_cut_high_10)

    # Cut 10% of lower and higher samples
    df_ = cut_low_5(df_, to_cut_low_5)

    # Cut 10% of lower and higher samples
    df_ = cut_low_10(df_, to_cut_low_10)
    
    # Cut 5% of lower and higher samples
    df_ = cut_low_high(df_, to_cut_low_high)
       
    return df_

# Listas de adaptación
# adaptado para Script_comp01_miranda

to_float = [
            'ctrx_quarter', 
            'cliente_edad',
            'ctarjeta_debito'
            ]

to_int = [
          'cprestamos_personales', 
          'cproductos', 
          'Master_status',
          'Visa_status', 
          'ctarjeta_debito', 
          'ccajas_consultas'
          ] #

to_cut_high_5 = [
                 'ctrx_quarter', 
                 'cprestamos_personales',
                 'Visa_mpagominimo', 
                 'Master_fultimo_cierre', 
                 'mtransferencias_recibidas', 
                 'ccajas_consultas', 
                 'mcomisiones', 
                 'Visa_mfinanciacion_limite', 
                 'Visa_msaldodolares',
                 'max_3_mpayroll', #
                 'cpayroll_trx', #
                 'mpayroll', #
                 'mov_avg_3_mpayroll', #
                 'min_3_mpayroll', #
                 'lag_1_mpayroll', #
                 'saldo_total_completo', #
                 'ctarjeta_visa_transacciones', #
                 'mtarjeta_visa_consumo', #
                 'mpasivos_margen', #
                 'saldo_total_cuentas', #
                 'avg_3_cuenta_corriente', #
                 'mtarjeta_master_descuentos', #
                 'mtarjeta_visa_descuentos', #
                 'mcajeros_propios_descuentos', #
                 'mpagomiscuentas', #
                 ]

to_cut_high_10 = [
                  'mcaja_ahorro', 
                  'mprestamos_personales',
                  'mpayroll',
                  'Master_mlimitecompra',
                  'Visa_mpagado'
                  ]

to_cut_low_5 = [ 
                 'Master_Fvencimiento', 
                 'Master_mpagospesos', 
                 'Visa_msaldodolares'
                 ]

to_cut_low_10 = []

to_cut_low_high = [
                   'mcuentas_saldo', 
                #    'mpasivos_margen', 
                   'mrentabilidad_annual',
                   'mrentabilidad', 
                   'mactivos_margen', 
                   'mcomisiones_otras', 
                   'Visa_msaldopesos', 
                   'mcomisiones_mantenimiento', 
                #    'mtarjeta_visa_consumo', 
                   'Master_Fvencimiento', 
                   'Visa_msaldototal', 

                   ]

to_log = [
          'mcuentas_saldo',
          'mcuenta_corriente'
          ]

