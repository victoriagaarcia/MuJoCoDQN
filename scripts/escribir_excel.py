import pandas as pd
import os

def save_experiment_to_excel(row_dict, filename="runs/experiments.xlsx"):
    # Convertimos el diccionario en un DataFrame de una sola fila
    new_df = pd.DataFrame([row_dict])
    
    # Comprobamos si el archivo ya existe
    if not os.path.isfile(filename):
        # Si no existe, creamos el archivo con cabeceras
        new_df.to_excel(filename, index=False, engine='openpyxl')
    else:
        # Si ya existe, abrimos el archivo y añadimos la fila al final
        with pd.ExcelWriter(filename, engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
            # Cargamos la hoja actual para saber dónde escribir
            try:
                start_row = writer.book['Sheet1'].max_row
            except KeyError:
                start_row = 0
            
            # Escribimos los datos sin repetir la cabecera (header=False)
            new_df.to_excel(writer, index=False, header=False, startrow=start_row, sheet_name='Sheet1')

# --- Ejemplo de uso ---
row = row = {
        "model_dir": "Feb12_02_10_58",
        "seed": 42,

        # hiperparámetros
        "total_steps": 1_000_000,
        "buffer_size": 100_000,
        "batch_size": 32,
        "gamma": 0.99,
        "lr": 1e-4,
        "target_update": 5_000,
        "start_training": 50_000,
        "eps_start": 1.0,
        "eps_end": 0.1,
        "eps_decay": 600_000,

        # métricas resumen
        f"avg_eval_reward": 260.28,
        "n_episodes": 10058,
    }
save_experiment_to_excel(row)
print("Fila añadida correctamente al Excel.")