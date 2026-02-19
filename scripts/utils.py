import os
import pandas as pd
import torch
import torch.nn.functional as F

def epsilon(step, eps_end, eps_start, start_decay, eps_decay):
    # return max(EPS_END, EPS_START - (step  / EPS_DECAY))
    return max(eps_end, eps_start - (max(0, step - start_decay) / eps_decay)) # Decay lineal con fase inicial de epsilon constante

@torch.no.grad()
def preprocess_rgb_batch_torch(rgb_bhwc: np.ndarray, out_size=84, device="cpu") -> torch.Tensor:
    """
    rgb_bhwc: (B,H,W,3) uint8
    return: (B,1,out_size,out_size) uint8
    """
    x = torch.from_numpy(rgb_bhwc).to(device, non_blocking=True)  # uint8
    x = x.permute(0, 3, 1, 2).float().div_(255.0)                 # (B,3,H,W) float

    # grayscale luminance
    x = 0.2989 * x[:, 0:1] + 0.5870 * x[:, 1:2] + 0.1140 * x[:, 2:3]  # (B,1,H,W)

    # resize
    x = F.interpolate(x, size=(out_size, out_size), mode="bilinear", align_corners=False)

    # back to uint8 for buffer
    x = (x * 255.0).clamp_(0, 255).to(torch.uint8)
    return x

# Escala target update por NUM_ENVS (en términos de transiciones reales)
def should_update_target(gs: int, target_update: int, num_envs: int) -> bool:
    return (gs % target_update) < num_envs

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
