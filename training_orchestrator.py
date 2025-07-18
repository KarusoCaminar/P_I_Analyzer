import os
import sys
import json
import time
import logging
import random
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from filelock import FileLock, Timeout # Für sicheres Schreiben der config.yaml
import yaml # Hinzugefügt für load_config

# Stelle sicher, dass die Root-Verzeichnisse für Imports im PYTHONPATH sind
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

# Importiere deine Module
# cli_runner wird als Subprozess aufgerufen, daher kein direkter Modul-Import nötig
import utils # Für Hilfsfunktionen wie create_output_directory etc.
from core_processor import Core_Processor, ProgressCallback # Direkter Import für Vortraining
from llm_handler import LLM_Handler
from knowledge_bases import KnowledgeManager


# Logging für den Orchestrator konfigurieren
logging.basicConfig(level=logging.INFO, format='[%(asctime)s - %(levelname)s - %(name)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger("TrainingOrchestrator")

# Hilfsfunktion zum Laden der Konfiguration (hier direkt im Orchestrator definiert)
def load_config(path: str) -> dict:
    """Lädt die zentrale YAML-Konfigurationsdatei."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logger.critical(f"FATAL: Konfigurationsdatei nicht gefunden: {path}")
        sys.exit(1)
    except yaml.YAMLError as e:
        logger.critical(f"FATAL: Fehler beim Parsen der Konfigurationsdatei {path}: {e}")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"FATAL: Ein unerwarteter Fehler ist beim Laden der Konfiguration aufgetreten: {e}", exc_info=True)
        sys.exit(1)

# Pfade aus der Konfiguration laden
CONFIG_PATH = SCRIPT_DIR / "config.yaml"
# Ein FileLock für die config.yaml, da sie vom Orchestrator überschrieben wird
CONFIG_LOCK = FileLock(str(CONFIG_PATH) + ".lock", timeout=60) 

APP_CONFIG = load_config(str(CONFIG_PATH)) # Initialisiere APP_CONFIG einmalig

# Globale Backend-Komponenten initialisieren (für direkten Zugriff, z.B. Vortraining)
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID")
GCP_LOCATION = os.getenv("GCP_LOCATION", "us-central1")

if not GCP_PROJECT_ID:
    logger.critical("FATAL: GCP_PROJECT_ID Umgebungsvariable nicht gesetzt. Kann Training nicht starten.")
    sys.exit(1)

orchestrator_llm_handler = LLM_Handler(GCP_PROJECT_ID, GCP_LOCATION, APP_CONFIG)
orchestrator_knowledge_manager = KnowledgeManager(
    APP_CONFIG["paths"]["element_type_list"], 
    APP_CONFIG["paths"]["learning_db"], 
    orchestrator_llm_handler,
    APP_CONFIG
)

# ==============================================================================
# Trainingslager-Konfiguration
# ==============================================================================

# Definiere deine Trainingsdaten (Passe diese Pfade an deine Ordnerstruktur an!)
TRAINING_DATA_DIR = SCRIPT_DIR / "training_data" 
# Unterordner für Schwierigkeitsgrade
SIMPLE_PIDS_DIR = TRAINING_DATA_DIR / "simple_pids"
COMPLEX_PIDS_DIR = TRAINING_DATA_DIR / "complex_pids"

SYMBOL_TRAINING_DIR = SCRIPT_DIR / "pretraining_symbols" # Ordner mit Pid-symbols-PDF_page_1.jpg

# Konfiguration des Trainingszyklus
TRAINING_DURATION_HOURS = 24 # Gesamtdauer des Trainings in Stunden
MAX_CYCLES_PER_STRATEGY = 5 # Maximale Analysezyklen pro Strategie/Parameter-Kombination, bevor gewechselt wird

# Strategien, die getestet werden sollen (aus config.yaml)
TRAINING_STRATEGIES_AND_MODELS = [
    {"type": "strategy", "name": "high_accuracy"}, # Beginne mit dem besten Modell
    {"type": "strategy", "name": "default_flash"},  # Dann das schnellere Flash-Modell
    # Füge hier weitere Strategien oder Modelle hinzu, die du testen möchtest
]

# Parameter für die iterative Optimierung (Magic Numbers, die variiert werden)
# Wir durchlaufen alle möglichen Kombinationen dieser Parameter
OPTIMIZABLE_PARAMETERS = {
    "min_quality_to_keep_bbox": [0.4, 0.5, 0.6, 0.7], 
    "iou_match_threshold": [0.4, 0.5, 0.6],          
    "visual_symbol_similarity_threshold": [0.75, 0.85, 0.9], 
    "graph_completion_distance_threshold": [0.03, 0.05, 0.07], 
}

def generate_parameter_combinations(params_dict: Dict[str, List[float]]) -> List[Dict[str, float]]:
    keys = list(params_dict.keys())
    values = list(params_dict.values())
    from itertools import product
    combinations = []
    for combination_values in product(*values):
        combo = dict(zip(keys, combination_values))
        combinations.append(combo)
    return combinations

ALL_PARAMETER_COMBINATIONS = generate_parameter_combinations(OPTIMIZABLE_PARAMETERS)
logger.info(f"Generierte {len(ALL_PARAMETER_COMBINATIONS)} Parameter-Kombinationen zum Testen.")


# ==============================================================================
# Hilfsfunktionen für den Trainingszyklus
# ==============================================================================

class CLITrainingProgressCallback(ProgressCallback):
    def update_progress(self, value: int, message: str):
        logger.info(f"Fortschritt: {message}")

    def update_status_label(self, text: str):
        logger.info(f"Status: {text}")

def run_single_analysis_cli(image_path: Path, training_output_base_dir: Path, strategy_or_model: Dict, params_override: Optional[Dict] = None) -> Optional[Dict]:
    logger.info(f"Starte Analyse für '{image_path.name}' mit Strategie/Modell: {strategy_or_model.get('name', strategy_or_model.get('id', 'unknown'))} (Params: {params_override})")
    
    # Temporäre Konfiguration erstellen, die an cli_runner übergeben wird
    temp_config_path = training_output_base_dir / f"temp_run_config_{os.getpid()}_{random.randint(0, 10000)}.json" 
    
    # Lade die aktuelle Master-Konfiguration, um sie anzupassen und zu übergeben
    # Wichtig: Hierfür brauchen wir den Lock, um Race Conditions zu vermeiden
    try:
        with CONFIG_LOCK: # Sperre die config.yaml für Lesung/Schreibung
            current_master_config_for_run = load_config(str(CONFIG_PATH)) # Nutze die lokale load_config

            # Wende Parameter-Overrides für diesen spezifischen Lauf an
            current_master_config_for_run.setdefault("logic_parameters", {}).update(params_override or {})

            # Setze die effektive Strategie für diesen Lauf in die temporäre Konfig
            run_strategy_config_for_cli = {}
            if strategy_or_model["type"] == "strategy":
                strategy_name = strategy_or_model["name"]
                strategy_def = current_master_config_for_run.get("strategies", {}).get(strategy_name)
                if not strategy_def:
                    logger.error(f"Strategie '{strategy_name}' nicht in der Master-Konfiguration gefunden.")
                    return None
                for step, model_display_name in strategy_def.items():
                    model_info = current_master_config_for_run.get("models", {}).get(model_display_name)
                    if not model_info:
                        logger.error(f"Modell '{model_display_name}' aus Strategie '{strategy_name}' nicht in Konfiguration gefunden.")
                        return None
                    run_strategy_config_for_cli[step] = {"id": model_info["id"], **model_info}
            elif strategy_or_model["type"] == "model_id":
                model_id = strategy_or_model["id"]
                model_info_from_config = current_master_config_for_run.get("models", {}).get(model_id) 
                if not model_info_from_config:
                    model_info_from_config = next((m for m_name, m in current_master_config_for_run.get("models", {}).items() if m.get("id") == model_id), None)
                if not model_info_from_config:
                    logger.error(f"Modell-ID '{model_id}' nicht in der Master-Konfiguration gefunden.")
                    return None
                temp_model_info = model_info_from_config.copy()
                if "location" in strategy_or_model:
                    temp_model_info["location"] = strategy_or_model["location"]
                run_strategy_config_for_cli = {
                    "meta_model": temp_model_info, "hotspot_model": temp_model_info,
                    "detail_model": temp_model_info, "correction_model": temp_model_info
                }
            
            # Speichere die effektive Strategie im temporären Konfigurations-Dict
            current_master_config_for_run["_active_run_strategy"] = run_strategy_config_for_cli

            with open(temp_config_path, 'w', encoding='utf-8') as f:
                json.dump(current_master_config_for_run, f, indent=2)
            logger.debug(f"Temporäre Konfigurationsdatei für Lauf erstellt: {temp_config_path}")

    except Timeout:
        logger.warning(f"Konnte den config.yaml-Lock für temporäre Konfig {temp_config_path} nicht erhalten. Lauf übersprungen.")
        return None
    except Exception as e:
        logger.error(f"Fehler beim Erstellen der temporären Konfigurationsdatei für '{image_path.name}': {e}", exc_info=True)
        return None

    command = [
        sys.executable, str(SCRIPT_DIR / "cli_runner.py"),
        "--image", str(image_path),
        "--config_path", str(temp_config_path), 
    ]
    
    subprocess_timeout = APP_CONFIG["logic_parameters"]["llm_default_timeout"] + 120 
    process = None
    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8')
        stdout, stderr = process.communicate(timeout=subprocess_timeout)

        if process.returncode != 0:
            logger.error(f"Analyse für '{image_path.name}' fehlgeschlagen mit Fehlercode {process.returncode}.")
            logger.error(f"Subprocess Output: \n{stdout}")
            return None
        
        potential_output_dirs = sorted(
            list(image_path.parent.glob(f"{image_path.stem}_output_*")),
            key=os.path.getmtime, reverse=True
        )
        
        if potential_output_dirs:
            latest_output_dir = potential_output_dirs[0]
            summary_filename = f"{image_path.stem}_summary.json"
            summary_path = latest_output_dir / summary_filename
            
            if summary_path.exists():
                with open(summary_path, 'r', encoding='utf-8') as f:
                    summary_data = json.load(f)
                    logger.debug(f"Summary geladen von: {summary_path}")
                    return summary_data
            else:
                logger.warning(f"Summary file '{summary_path}' not found after cli_runner execution for '{image_path.name}'.")
                logger.debug(f"Subprocess Output: \n{stdout}")
                return None
        else:
            logger.warning(f"Kein Output-Verzeichnis für '{image_path.name}' gefunden nach cli_runner Ausführung.")
            logger.debug(f"Subprocess Output: \n{stdout}")
            return None

    except subprocess.TimeoutExpired:
        logger.error(f"Analyse für '{image_path.name}' hat das Zeitlimit ({subprocess_timeout}s) überschritten und wurde beendet.")
        if process:
            process.kill()
            stdout, stderr = process.communicate()
            logger.error(f"Subprocess Output (Timeout): \n{stdout}")
        return None
    except Exception as e:
        logger.error(f"Fehler beim Aufruf des cli_runner für '{image_path.name}': {e}", exc_info=True)
        return None
    finally:
        if temp_config_path.exists():
            os.remove(temp_config_path)

# ==============================================================================
# Haupt-Trainingszyklus
# ==============================================================================

def start_training_camp():
    logger.info(f"===== Starte KI-Trainingslager für {TRAINING_DURATION_HOURS} Stunden =====")
    
    start_time = time.time()
    end_time = start_time + TRAINING_DURATION_HOURS * 3600 # Sekunden

    # Lade Bilder nach Schwierigkeitsgrad (Annahme: Ordnerstruktur wie oben beschrieben)
    all_simple_images = list(SIMPLE_PIDS_DIR.rglob("*.png")) + \
                        list(SIMPLE_PIDS_DIR.rglob("*.jpg")) + \
                        list(SIMPLE_PIDS_DIR.rglob("*.jpeg"))
    all_complex_images = list(COMPLEX_PIDS_DIR.rglob("*.png")) + \
                         list(COMPLEX_PIDS_DIR.rglob("*.jpg")) + \
                         list(COMPLEX_PIDS_DIR.rglob("*.jpeg"))
    all_training_images = all_simple_images + all_complex_images # Kombiniere alle Bilder für die Phase "ohne Truth"

    if not all_simple_images and not all_complex_images:
        logger.critical(f"Keine Trainingsbilder in '{TRAINING_DATA_DIR}' gefunden. Bitte Pfad prüfen.")
        return

    logger.info(f"Gefundene einfache Trainingsbilder: {len(all_simple_images)}")
    logger.info(f"Gefundene komplexe Trainingsbilder: {len(all_complex_images)}")
    
    best_overall_score = -1.0
    best_config_achieved: Dict[str, Any] = {}
    
    cycle_count = 0
    param_combo_idx = 0 
    
    consecutive_high_scores = 0
    MODE_CHANGE_THRESHOLD_SIMPLE = 3 
    MODE_CHANGE_THRESHOLD_COMPLEX = 5 

    current_image_set = all_simple_images 
    training_phase_name = "Einfache PIDs mit Truth"
    current_min_success_score = APP_CONFIG.get('logic_parameters', {}).get('min_score_for_few_shot_pattern', 85) 
    
    while time.time() < end_time:
        cycle_count += 1
        logger.info(f"\n--- Trainingszyklus {cycle_count} gestartet (Phase: {training_phase_name}) ---")
        logger.info(f"Verstrichen: {((time.time() - start_time) / 3600):.2f}/{TRAINING_DURATION_HOURS} Stunden.")

        # Symbol-Vortraining (strategisch)
        if cycle_count == 1 or (cycle_count % 10 == 0): 
            logger.info("Führe Symbol-Vortraining durch...")
            try:
                multimodal_embedding_model_info = APP_CONFIG.get("models", {}).get("MultiModal Embedding (Image)")
                if not multimodal_embedding_model_info:
                    logger.error("MultiModal Embedding Modell (Image) nicht in Konfiguration gefunden. Symbol-Vortraining übersprungen.")
                else:
                    temp_core_processor = Core_Processor(
                        orchestrator_llm_handler, 
                        orchestrator_knowledge_manager, 
                        APP_CONFIG.get("strategies", {}).get("default_flash"),
                        APP_CONFIG 
                    )
                    pretraining_report = temp_core_processor.run_pretraining(SYMBOL_TRAINING_DIR, multimodal_embedding_model_info)
                    logger.info(f"Symbol-Vortraining abgeschlossen. {len(pretraining_report)} Symbole verarbeitet.")
            except Exception as e:
                logger.error(f"Fehler während des Symbol-Vortrainings: {e}", exc_info=True)
        
        # Parameter-Kombination für diesen Zyklus auswählen (Iterative Optimierung)
        if not ALL_PARAMETER_COMBINATIONS: # Fallback, falls keine Kombis generiert
            current_params_override = {}
            logger.warning("Keine Parameter-Kombinationen zum Testen definiert. Verwende Standardparameter.")
        else:
            current_params_override = ALL_PARAMETER_COMBINATIONS[param_combo_idx]
            logger.info(f"Aktuelle Parameter-Kombination (Zyklus {cycle_count}, Index {param_combo_idx}): {current_params_override}")
            param_combo_idx = (param_combo_idx + 1) % len(ALL_PARAMETER_COMBINATIONS)

        # Schleife über Strategien/Modelle
        for strategy_or_model in TRAINING_STRATEGIES_AND_MODELS:
            current_strategy_name = strategy_or_model.get('name', strategy_or_model.get('id', 'unknown'))
            logger.info(f"Teste Strategie/Modell: '{current_strategy_name}' in Zyklus {cycle_count}")
            
            # Schleife über Bilder des aktuellen Schwierigkeitsgrades
            random.shuffle(current_image_set) 
            
            cycle_scores: List[float] = [] 
            
            for image_path in current_image_set:
                if time.time() >= end_time: 
                    logger.info("Zeitlimit für Training erreicht. Beende den Zyklus vorzeitig.")
                    break

                analysis_summary = run_single_analysis_cli(
                    image_path=image_path,
                    training_output_base_dir=SCRIPT_DIR, 
                    strategy_or_model=strategy_or_model,
                    params_override=current_params_override 
                )

                if analysis_summary:
                    current_score = analysis_summary.get("recognition_quality_score", 0.0)
                    cycle_scores.append(current_score)
                    logger.info(f"Analyse für '{image_path.name}' mit '{current_strategy_name}' (Score: {current_score:.2f}) abgeschlossen.")
                    
                    if current_score > best_overall_score:
                        best_overall_score = current_score
                        best_config_achieved = {
                            "strategy_used": strategy_or_model,
                            "logic_parameters": current_params_override,
                            "recognition_quality_score": best_overall_score,
                            "timestamp": datetime.now().isoformat(),
                            "image_name_of_best_run": image_path.name,
                            "kpis": analysis_summary.get("final_kpis", {})
                        }
                        logger.info(f"*** NEUER BESTER SCORE: {best_overall_score:.2f} mit Konfiguration: {best_config_achieved} ***")
                        
                        try:
                            with CONFIG_LOCK: 
                                current_master_config = load_config(str(CONFIG_PATH)) # Nutze die lokale load_config
                                current_master_config.setdefault("logic_parameters", {}).update(best_config_achieved["logic_parameters"])
                                with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
                                    json.dump(current_master_config, f, indent=2)
                                logger.info(f"Beste Konfiguration ({best_overall_score:.2f}) in config.yaml gespeichert.")
                        except Timeout:
                            logger.warning("Konnte die Sperre für config.yaml nicht erhalten. Beste Konfiguration nicht persistent gespeichert.")
                        except Exception as e:
                            logger.error(f"Fehler beim Speichern der besten Konfiguration in config.yaml: {e}", exc_info=True)
                        
                else:
                    logger.warning(f"Analyse für '{image_path.name}' mit '{current_strategy_name}' fehlgeschlagen oder keine Summary erhalten.")
            
            if time.time() >= end_time: break 

        # Logik zum Wechseln der Trainingsphase (Einfach -> Komplex -> Ohne Truth)
        if cycle_scores: # Nur wechseln, wenn in diesem Zyklus Analysen durchgeführt wurden
            avg_cycle_score = sum(cycle_scores) / len(cycle_scores)
            if avg_cycle_score >= current_min_success_score:
                consecutive_high_scores += 1
                logger.info(f"Durchschnittlicher Score dieses Strategie/Parameter-Sets: {avg_cycle_score:.2f}. Aufeinanderfolgende Top-Scores: {consecutive_high_scores}")
            else:
                consecutive_high_scores = 0 
        else:
            consecutive_high_scores = 0 # Zurücksetzen, wenn keine Scores gesammelt wurden

        if training_phase_name == "Einfache PIDs mit Truth" and consecutive_high_scores >= MODE_CHANGE_THRESHOLD_SIMPLE:
            logger.info(f"\n--- WECHSEL PHASE: Übergang zu komplexen PIDs (nach {MODE_CHANGE_THRESHOLD_SIMPLE} Erfolgen auf einfachen PIDs) ---")
            training_phase_name = "Komplexe PIDs mit Truth"
            current_image_set = all_complex_images
            consecutive_high_scores = 0 

        elif training_phase_name == "Komplexe PIDs mit Truth" and consecutive_high_scores >= MODE_CHANGE_THRESHOLD_COMPLEX:
            logger.info(f"\n--- WECHSEL PHASE: Übergang zu PIDs ohne Truth-Dateien (nach {MODE_CHANGE_THRESHOLD_COMPLEX} Erfolgen auf komplexen PIDs) ---")
            training_phase_name = "PIDs ohne Truth (Test Generalisierung)"
            current_image_set = all_training_images + all_simple_images + all_complex_images 
            consecutive_high_scores = 0 

        logger.info(f"--- Trainingszyklus {cycle_count} abgeschlossen ---")
        if time.time() >= end_time: break 

    logger.info(f"\n===== KI-Trainingslager beendet nach {cycle_count} Zyklen =====")
    logger.info(f"Gesamt-Laufzeit: {((time.time() - start_time) / 3600):.2f} Stunden.")
    logger.info(f"Bestes erreichtes Ergebnis: {best_config_achieved}")
    logger.info("Bitte prüfen Sie die learning_db.json und die einzelnen Output-Ordner für Details.")

# ==============================================================================
# Skript-Startpunkt
# ==============================================================================
if __name__ == "__main__":
    temp_symbol_storage_dir = Path(APP_CONFIG.get('paths', {}).get('temp_symbol_dir', SCRIPT_DIR / "temp_symbols_for_embeddings"))
    temp_symbol_storage_dir.mkdir(parents=True, exist_ok=True)
    
    start_training_camp()