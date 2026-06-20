import os
from google.colab import userdata

# 1. Configuración del token de Hugging Face (Requerido para descargar pesos de SAM3)
os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")

# 2. Instalar Triton (básico para operadores matemáticos de la GPU) y Supervision
print(" Instalando dependencias base del sistema...")
!pip install -q supervision jupyter_bbox_widget triton

# 3. Clonado y compilación nativa de los operadores C++/CUDA de SAM3
if not os.path.exists("sam3"):
    print(" Clonando repositorio de SAM3 y compilando extensiones CUDA...")
    !git clone https://github.com/facebookresearch/sam3.git
    %cd sam3
    !pip install -e ".[notebooks]"
    %cd /content
    # cc_torch y torch_generic_nms aceleran la supresión de no máximos (NMS) en hardware
    !pip uninstall -y cc_torch; TORCH_CUDA_ARCH_LIST="8.0 9.0"; pip install git+https://github.com/ronghanghu/cc_torch
    !pip uninstall -y torch_generic_nms; TORCH_CUDA_ARCH_LIST="8.0 9.0"; pip install git+https://github.com/ronghanghu/torch_generic_nms
    print("✅ Compilación de SAM3 completada.")

print("\n 'Reiniciar sesión'.")
print("Después de reiniciar, no vuelvas a correr esta celda y pasa directo a la Celda 2.")

import os
import torch
import torchvision
from pathlib import Path
from google.colab import drive
from sam3.model_builder import build_sam3_video_predictor

# 1. Montaje físico del almacenamiento en la nube de Google Drive
if not os.path.exists('/content/drive'):
    print(" Conectando con Google Drive...")
    drive.mount('/content/drive')

# 2. Configuración estricta de rutas de Drive y Entorno Local
CARPETA_PROYECTO = "/content/drive/MyDrive/MiRobótica/" 
NOMBRE_VIDEO = "video_ligero(10).mp4"  # Nombre de tu video actual

VIDEO_ORIGINAL = os.path.join(CARPETA_PROYECTO, NOMBRE_VIDEO)
OUTPUT_CSV = os.path.join(CARPETA_PROYECTO, "analisis_robots_balon.csv")
OUTPUT_ANIMACION_MP4 = os.path.join(CARPETA_PROYECTO, "mapa_calor_animado.mp4")

if not os.path.exists(VIDEO_ORIGINAL):
    raise FileNotFoundError(f" No se encontró el archivo de video en: {VIDEO_ORIGINAL}")

HOME = Path.cwd()
DIR_CHUNKS = HOME / "video_chunks"
DIR_CHUNKS.mkdir(parents=True, exist_ok=True)

# 3. Inicialización del Predictor de SAM3 sobre la GPU T4
print(f"PyTorch Version: {torch.__version__}")
if not torch.cuda.is_available():
    raise RuntimeError(" La GPU T4 no está activa en tu entorno de Colab. Actívala en los menús.")

predictor = build_sam3_video_predictor(
    bpe_path="/content/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz", 
    gpus_to_use=[0]
)
print(" Predictor cargado en VRAM con éxito. Pasa a la Celda 3.")

import gc
import cv2
import subprocess
import numpy as np
import pandas as pd
import seaborn as sns
import supervision as sv
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

# =====================================================================
# 1. SEGMENTACIÓN CON FFMPEG (CORTES DE 3 SEGUNDOS A 480P)
# =====================================================================
for f in DIR_CHUNKS.glob("*.mp4"): os.remove(f)

print(" Cortando el video en fragmentos físicos de 3 segundos a 480p...")
cmd_segmentar = [
    "ffmpeg", "-y", "-loglevel", "error",
    "-i", str(VIDEO_ORIGINAL),
    "-vf", "scale=480:-2", "-r", "15",
    "-vcodec", "libx264", "-crf", "32",
    "-f", "segment", "-segment_time", "3",
    "-reset_timestamps", "1",
    f"{DIR_CHUNKS}/chunk_%03d.mp4"
]
subprocess.run(cmd_segmentar)
videos_fragmentados = sorted(list(DIR_CHUNKS.glob("*.mp4")))
print(f"Video segmentado con éxito en {len(videos_fragmentados)} fragmentos.")

# =====================================================================
# 2. PIPELINE DE TRACKING CÍCLICO (PROTECCIÓN ANTI-CRASH DE RAM)
# =====================================================================
datos_balon, datos_robots = [], []
frame_global_offset = 0  

def from_sam(result: dict) -> sv.Detections:
    if "out_binary_masks" not in result or result["out_binary_masks"] is None:
        return sv.Detections.empty()
    return sv.Detections(
        xyxy=sv.mask_to_xyxy(result["out_binary_masks"]), 
        mask=result["out_binary_masks"], 
        confidence=result["out_probs"], 
        tracker_id=result["out_obj_ids"]
    )

print(" Ejecutando tracking secuencial por lotes...")
with torch.no_grad(): # Desactiva gradientes para ahorrar un ~50% de memoria RAM gráfica
    for idx, ruta_chunk in enumerate(videos_fragmentados):
        gc.collect()
        torch.cuda.empty_cache()
        
        try:
            response = predictor.handle_request(request=dict(type="start_session", resource_path=ruta_chunk.as_posix()))
            session_id = response["session_id"]
            _ = predictor.handle_request(request=dict(type="reset_session", session_id=session_id))
            
            _ = predictor.handle_request(request=dict(type="add_prompt", session_id=session_id, frame_index=0, text="orange ball"))
            _ = predictor.handle_request(request=dict(type="add_prompt", session_id=session_id, frame_index=0, text="circular robot"))
            
            frame_outputs_chunk = {}
            for resp in predictor.handle_stream_request(request=dict(type="propagate_in_video", session_id=session_id)):
                frame_outputs_chunk[resp["frame_index"]] = resp["outputs"]
                
            cant_frames_procesados = len(frame_outputs_chunk)
            
            for f_idx, output in sorted(frame_outputs_chunk.items()):
                detections = from_sam(output)
                if len(detections.mask) == 0: continue
                
                objetos_frame = []
                for i, mask in enumerate(detections.mask):
                    if not mask.any(): continue
                    area = np.sum(mask)
                    posiciones = np.argwhere(mask)
                    cy, cx = posiciones[:, 0].mean(), posiciones[:, 1].mean()
                    objetos_frame.append({"area": area, "cx": cx, "cy": cy})
                    
                if not objetos_frame: continue
                objetos_ordenados = sorted(objetos_frame, key=lambda x: x["area"])
                real_frame_idx = f_idx + frame_global_offset
                
                # Balón (Máscara más pequeña)
                balon = objetos_ordenados[0]
                datos_balon.append({"Frame": real_frame_idx, "Ball_X": balon["cx"], "Ball_Y": balon["cy"]})
                
                # Robots (Asignación secuencial)
                for r_idx, r_data in enumerate(objetos_ordenados[1:5]):
                    datos_robots.append({
                        "Frame": real_frame_idx, 
                        "Robot_ID": r_idx + 1, 
                        "Robot_X": r_data["cx"], 
                        "Robot_Y": r_data["cy"]
                    })
            frame_global_offset += cant_frames_procesados
        except Exception as e:
            print(f"⚠ Fragmento omitido {ruta_chunk.name} debido a error: {e}")
        finally:
            try: _ = predictor.handle_request(request=dict(type="end_session", session_id=session_id))
            except: pass
            del frame_outputs_chunk

print(" Tracking e indexación temporal completada.")
df_balon = pd.DataFrame(datos_balon)
df_robots = pd.DataFrame(datos_robots)

if not df_robots.empty:
    df_robots.to_csv(OUTPUT_CSV, index=False)

# =====================================================================
# 3. CONTEO DE TOQUES POR PROXIMIDAD
# =====================================================================
UMBRAL_DISTANCIA = 22  
toques_por_robot = {1: 0, 2: 0, 3: 0, 4: 0}
ultimo_robot_contacto, cooldown_ticks = None, 0

if not df_balon.empty and not df_robots.empty:
    df_merge = pd.merge(df_robots, df_balon, on="Frame")
    df_merge['Distancia'] = np.sqrt((df_merge['Robot_X'] - df_merge['Ball_X'])**2 + (df_merge['Robot_Y'] - df_merge['Ball_Y'])**2)
    cerca = df_merge[df_merge['Distancia'] <= UMBRAL_DISTANCIA].sort_values(by="Frame")
    
    for _, row in cerca.iterrows():
        r_id = int(row['Robot_ID'])
        if ultimo_robot_contacto == r_id and cooldown_ticks > 0:
            cooldown_ticks -= 1
            continue
        toques_por_robot[r_id] += 1
        ultimo_robot_contacto = r_id
        cooldown_ticks = 8

# =====================================================================
# 4. COMPILACIÓN DEL VIDEO CON MAPA DE CALOR ANIMADO SINCRO
# =====================================================================
if not df_robots.empty and not df_balon.empty:
    print(" Renderizando animación sincrónica frame por frame...")
    
    robots_unicos = sorted(df_robots['Robot_ID'].unique())
    num_plots = len(robots_unicos)
    frames_totales = sorted(df_robots['Frame'].unique())
    
    fig, axes = plt.subplots(1, num_plots, figsize=(4 * num_plots, 4), squeeze=False)
    lines_robots, scatters_robots, scatters_balon = {}, {}, {}

    max_x = max(df_robots['Robot_X'].max(), df_balon['Ball_X'].max()) + 20
    max_y = max(df_robots['Robot_Y'].max(), df_balon['Ball_Y'].max()) + 20

    for idx, r_id in enumerate(robots_unicos):
        ax = axes[0, idx]
        ax.set_xlim(0, max_x)
        ax.set_ylim(0, max_y)
        ax.invert_yaxis()
        ax.grid(True, linestyle='--', alpha=0.3)
        
        lines_robots[r_id], = ax.plot([], [], color='black', alpha=0.4, linewidth=1)
        scatters_robots[r_id] = ax.scatter([], [], color='blue', s=15, alpha=0.7, label="Robot")
        scatters_balon[r_id] = ax.scatter([], [], color='orange', s=25, edgecolors='black', label="Balón")
        if idx == 0: ax.legend(loc="upper right", fontsize=8)

    def update(frame_actual):
        df_acumulado_robots = df_robots[df_robots['Frame'] <= frame_actual]
        df_instante_balon = df_balon[df_balon['Frame'] == frame_actual]
        
        for idx, r_id in enumerate(robots_unicos):
            ax = axes[0, idx]
            df_sub_r = df_acumulado_robots[df_acumulado_robots['Robot_ID'] == r_id]
            
            if not df_sub_r.empty:
                lines_robots[r_id].set_data(df_sub_r['Robot_X'].values, df_sub_r['Robot_Y'].values)
                scatters_robots[r_id].set_offsets(np.c_[df_sub_r['Robot_X'].values, df_sub_r['Robot_Y'].values])
                
                # Pintar densidad acumulada cada 10 frames para optimizar velocidad de dibujo
                if len(df_sub_r) > 5 and frame_actual % 10 == 0:
                    for coll in list(ax.collections):
                        if coll not in [scatters_robots[r_id], scatters_balon[r_id]]:
                            coll.remove()
                    try:
                        sns.kdeplot(x=df_sub_r['Robot_X'], y=df_sub_r['Robot_Y'], cmap="Blues", fill=True, bw_adjust=0.6, ax=ax, thresh=0.1, alpha=0.4, zorder=0)
                    except: pass

            if not df_instante_balon.empty:
                scatters_balon[r_id].set_offsets(np.c_[df_instante_balon['Ball_X'].values, df_instante_balon['Ball_Y'].values])
            
            ax.set_title(f'Robot #{r_id}\n({toques_por_robot.get(r_id, 0)} Toques)')
        return list(lines_robots.values()) + list(scatters_robots.values()) + list(scatters_balon.values())

    ani = FuncAnimation(fig, update, frames=frames_totales, blit=False, interval=66)
    
    print(" Guardando video animado en Google Drive...")
    ani.save(OUTPUT_ANIMACION_MP4, writer='ffmpeg', fps=15)
    plt.close()
    
    print(f" Animación exportada. Ubicación en tu Drive: {OUTPUT_ANIMACION_MP4}")
else:
    print("Datos insuficientes para renderizar la secuencia animada.")
