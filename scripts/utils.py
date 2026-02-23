import os
import pandas as pd
import torch
import torch.nn.functional as F
import numpy as np
import cv2

def epsilon(step, eps_end, eps_start, start_decay, eps_decay):
    # return max(EPS_END, EPS_START - (step  / EPS_DECAY))
    return max(eps_end, eps_start - (max(0, step - start_decay) / eps_decay)) # Decay lineal con fase inicial de epsilon constante

@torch.no_grad()
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
    prev = gs - num_envs
    return (prev // target_update) != (gs //target_update)

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
            
def preprocess(frame, size=84):
    """
    RGB uint8 (H,W,3) -> grayscale float32 (84,84) en [0,1]
    """
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frame = cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA) # Normaliza imagen a 84x84
    return frame

def to_uint8_stack(obs: np.ndarray) -> torch.Tensor:
    """
    Convierte observación (B,4,84,84) a torch.uint8 CPU.
    Soporta:
      - obs float32 en [0,1]  -> uint8 [0,255]
      - obs uint8  en [0,255] -> uint8
    """
    # Asegura contigüidad para from_numpy rápido
    obs = np.ascontiguousarray(obs)

    if obs.dtype == np.uint8:
        return torch.from_numpy(obs)  # (B,4,84,84) uint8
    else:
        # asumimos float en [0,1] (como el wrapper antiguo)
        # si por algún motivo ya estuviera en [0,255] float, lo saturamos igual
        x = np.clip(obs * 255.0, 0.0, 255.0).astype(np.uint8, copy=False)
        return torch.from_numpy(x)
