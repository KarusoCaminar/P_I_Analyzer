# Standardbibliotheken
import os
import sys
import time
import json
import logging
import argparse

# Drittanbieter-Bibliotheken
import yaml
from tqdm import tqdm
import dotenv

# Eigene Module
from llm_handler import LLM_Handler
from knowledge_bases import KnowledgeManager
from core_processor import Core_Processor, ProgressCallback
import utils

# Typing
from typing import Dict, Any, Optional, List, Callable, cast
from pathlib import Path

# Umgebung laden
dotenv.load_dotenv()


# --- Konfiguration & Konstanten ---
logger = logging.getLogger(__name__)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(SCRIPT_DIR, "config.yaml")
# --- Hilfsfunktionen ---

def load_config(path: str) -> dict:
    """Lädt die zentrale YAML-Konfigurationsdatei."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logger.error(f"Konfigurationsdatei nicht gefunden: {path}")
        sys.exit(1)
    except yaml.YAMLError as e:
        logger.error(f"Fehler beim Parsen der Konfigurationsdatei {path}: {e}")
        sys.exit(1)

def save_analysis_artifacts(results: dict, output_dir: str, image_path: str):
    """Speichert alle Artefakte aus den Analyseergebnissen."""
    logging.info("Speichere Analyse-Artefakte...")
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    utils.save_json_output(results, os.path.join(output_dir, f"{base_name}_raw_ai_output.json"))

    summary_data = {
        "image_name": os.path.basename(image_path),
        "model_strategy_used": results.get("model_strategy_used"),
        "final_kpis": results.get("final_kpis", {}),
        "recognition_quality_score": results.get("quality_score", 0.0),
        "model_name": results.get("model_name_for_summary", "unknown") # NEU: Für den Benchmark-Report
    }
    utils.save_json_output(summary_data, os.path.join(output_dir, f"{base_name}_summary.json"))
    
    logging.info(f"Alle Artefakte für '{os.path.basename(image_path)}' erfolgreich gespeichert.")

# --- Hauptlogik ---
# NEU: model_id und location sind jetzt optionale Parameter, die eine Strategie überschreiben können
def run_single_analysis(image_path: str, config: dict,
                        strategy_name: Optional[str] = None,
                        model_id: Optional[str] = None, location: Optional[str] = None,
                        explicit_model_strategy_config: Optional[Dict[str, Any]] = None, # NEU für Orchestrator
                        explicit_params_override: Optional[Dict[str, Any]] = None # NEU für Orchestrator
                        ):
    """Führt eine komplette Analyse mit einer definierten Strategie oder einem spezifischen Modell aus."""
    
    effective_model_strategy_config: Dict[str, Dict[str, Any]] = {} # Die tatsächlich verwendete Strategie-Konfiguration
    model_id_for_dir: str # Der Name des Modells/der Strategie für den Ausgabeordner
    
    # --- 1. Bestimme die effektive Modellstrategie ---
    if explicit_model_strategy_config:
        # Fall A: Orchestrator gibt explizite Strategie-Konfiguration vor
        logger.info(f"CLI RUNNER: Verwende Orchestrator-definierte Strategie für '{os.path.basename(image_path)}'.")
        effective_model_strategy_config = explicit_model_strategy_config
        # Den Namen für den Ausgabeordner aus dem Detailmodell der expliziten Strategie ableiten
        model_id_for_dir = effective_model_strategy_config.get('detail_model', {}).get('id', 'orchestrator_default').replace("google/", "").replace("vertex_ai/", "").replace("groq/", "").replace("ollama/", "")
    elif model_id:
        # Fall B: Einzelnes Modell für alle Phasen verwenden (z.B. vom Benchmark-Runner oder direkter CLI-Aufruf)
        logger.info(f"CLI RUNNER: Starte Analyse für '{os.path.basename(image_path)}' mit Ad-hoc-Strategie basierend auf Modell '{model_id}'.")
        model_info_from_config = config.get("models", {}).get(model_id) # Versuche Display-Name Match
        if not model_info_from_config: # Fallback: Suche nach ID im 'id' Feld der Modelle
            model_info_from_config = next((m for m_name, m in config.get("models", {}).items() if m.get("id") == model_id), None)
        
        if not model_info_from_config:
            logger.error(f"Modell-ID '{model_id}' nicht in der Konfiguration gefunden.")
            sys.exit(1)
        
        # Erstelle eine Ad-hoc-Strategie mit diesem Modell für alle Phasen
        temp_model_info = model_info_from_config.copy()
        if location: # CLI Location überschreibt Config-Location wenn angegeben
            temp_model_info['location'] = location
        # Füge generation_config hinzu, falls in der config.yaml für dieses Modell definiert
        if "generation_config" in model_info_from_config:
             temp_model_info["generation_config"] = model_info_from_config["generation_config"].copy()
        
        effective_model_strategy_config = {
            "meta_model": temp_model_info,
            "hotspot_model": temp_model_info,
            "detail_model": temp_model_info,
            "correction_model": temp_model_info
        }
        model_id_for_dir = model_id.replace("google/", "").replace("vertex_ai/", "").replace("groq/", "").replace("ollama/", "")
    elif strategy_name:
        # Fall C: Vordefinierte Strategie aus config.yaml verwenden (direkter CLI-Aufruf)
        logger.info(f"CLI RUNNER: Starte Analyse für '{os.path.basename(image_path)}' mit Strategie '{strategy_name}'.")
        strategy_def = config.get("strategies", {}).get(strategy_name)
        if not strategy_def:
            logger.error(f"Strategie '{strategy_name}' nicht in der Konfigurationsdatei gefunden.")
            sys.exit(1)
        
        for step, model_name_display in strategy_def.items(): # model_name_display ist der lesbare Name
            model_info = config.get("models", {}).get(model_name_display) # Hole Model-Info über Display-Name
            if not model_info:
                logger.error(f"Modell '{model_name_display}' aus Strategie '{strategy_name}' nicht in der Konfiguration gefunden.")
                sys.exit(1)
            effective_model_strategy_config[step] = {"id": model_info["id"], **model_info} # Speichere mit ID und allen Infos
        
        # Den Namen für den Ausgabeordner aus dem Detailmodell der gewählten Strategie ableiten
        model_id_for_dir = effective_model_strategy_config.get('detail_model', {}).get('id', 'default_strategy').replace("google/", "").replace("vertex_ai/", "").replace("groq/", "").replace("ollama/", "")
    else:
        logger.error("Keine Strategie oder Modell-ID/Location angegeben. Analyse kann nicht gestartet werden.")
        sys.exit(1)

    # --- 2. Bestimme die effektiven Logik-Parameter ---
    # Die Basis-Logic-Parameters kommen immer aus der App-Config.
    # explicit_params_override (vom Orchestrator) überschreibt diese, wenn sie vorhanden sind.
    final_logic_parameters = config.get("logic_parameters", {}).copy()
    if explicit_params_override:
        logger.info("CLI RUNNER: Wende Logik-Parameter-Overrides aus Orchestrator an.")
        final_logic_parameters.update(explicit_params_override)
    logger.debug(f"Effektive Logik-Parameter für diesen Lauf: {final_logic_parameters}")

    # --- 3. Erstelle Output-Verzeichnis und richte Logging ein ---
    output_dir = utils.create_output_directory(image_path, model_id_for_dir)
    if not output_dir:
        logger.error("Konnte kein Ausgabe-Verzeichnis erstellen. Analyse abgebrochen.")
        sys.exit(1) # Kritischer Fehler
    utils.setup_logging(output_dir) 
    
    logger.info(f"CLI RUNNER: Analyse für '{os.path.basename(image_path)}' mit Modell/Strategie '{model_id_for_dir}' gestartet.")
    start_time = time.time()
    
    # --- 4. Initialisiere Backend-Komponenten ---
    gcp_project_id = os.getenv("GCP_PROJECT_ID")
    gcp_location = os.getenv("GCP_LOCATION", "us-central1") # Standardwert, kann überschrieben werden

    if not gcp_project_id:
        logger.error("GCP_PROJECT_ID Umgebungsvariable nicht gesetzt. Breche Analyse ab.")
        sys.exit(1)
        
    llm_handler = LLM_Handler(gcp_project_id, gcp_location, config) # Volle Konfig an LLM_Handler
    knowledge_manager = KnowledgeManager(
        config["paths"]["element_type_list"], 
        config["paths"]["learning_db"], 
        llm_handler,
        config # Volle Konfig an KnowledgeManager
    )
    
    # --- 5. Initialisiere den Core_Processor und starte die Pipeline ---
    processor = Core_Processor(llm_handler, knowledge_manager, effective_model_strategy_config, config) # Volle Konfig an Core_Processor
    
    try:
        # progress_callback-Implementierung für CLI
        class CliProgressCallback(ProgressCallback):
            def __init__(self):
                self.last_value = 0
                
            def update_progress(self, value: int, message: str):
                # tqdm progress bar can be integrated here if needed.
                # For simple CLI, just log the change.
                if value > self.last_value:
                    logger.info(f"Fortschritt: {message} ({value}%)")
                    self.last_value = value

            def update_status_label(self, text: str):
                logger.info(f"Status: {text}")

        cli_callback = CliProgressCallback() # Instanz der Callback-Klasse
        
        final_results = processor.run_full_pipeline(
            image_path=image_path,
            progress_callback=cli_callback,
            output_dir=output_dir,
            analysis_params_override=final_logic_parameters
        )
        
        # --- 6. Speichere Analyse-Artefakte und Zusammenfassung ---
        end_time = time.time()
        logger.info(f"CLI RUNNER: Analyse für '{os.path.basename(image_path)}' abgeschlossen in {end_time - start_time:.2f} Sekunden.")

        # Füge Modell- und Strategie-Informationen zum final_results-Dictionary hinzu
        final_results["model_name_for_summary"] = model_id_for_dir
        final_results["model_strategy_used"] = {k: v.get('id', 'N/A') for k, v in effective_model_strategy_config.items()}

        save_analysis_artifacts(final_results, output_dir, image_path)
        logger.info(f"CLI RUNNER: Alle Artefakte für '{os.path.basename(image_path)}' in '{output_dir}' gespeichert.")

    except Exception as e:
        logger.critical(f"CLI RUNNER: Ein kritischer Fehler ist während der Analyse-Pipeline für '{os.path.basename(image_path)}' aufgetreten: {e}", exc_info=True)
        sys.exit(1)

# --- Einstiegspunkt ---

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s - %(levelname)s] %(message)s', datefmt='%H:%M:%S')

    parser = argparse.ArgumentParser(description="Führt eine P&ID-Analyse für ein Bild mit einer Strategie oder einem spezifischen Modell aus.")
    parser.add_argument("--image", required=True, help="Pfad zur zu analysierenden Bilddatei.")
    parser.add_argument("--strategy", help="Name der zu verwendenden Analyse-Strategie aus config.yaml (z.B. 'default_flash').")
    parser.add_argument("--model_id", help="Die ID eines spezifischen Modells, das für alle Phasen verwendet werden soll (überschreibt --strategy).")
    parser.add_argument("--location", help="Die GCP-Region für das Modell (nur relevant mit --model_id).")
    # NEU: Parameter für den Trainings-Orchestrator
    parser.add_argument("--config_path", help="Optionaler Pfad zu einer temporären Konfigurationsdatei, die geladen werden soll (vom Trainings-Orchestrator verwendet).")
    # Der cli_runner wird die Strategie und Parameter nun direkt aus der geladenen config_path ziehen.
    
    args = parser.parse_args()

    # Lade die Konfiguration: entweder die temporäre vom Orchestrator oder die Standard-Konfiguration
    current_config_path_loaded: Path
    if args.config_path:
        app_config = load_config(args.config_path)
        current_config_path_loaded = Path(args.config_path)
        logger.info(f"CLI RUNNER: Temporäre Konfiguration von '{args.config_path}' geladen.")
        logger.info(f"DEBUG: Geladener Konfigurationspfad: {current_config_path_loaded.resolve()}")
    else:
        app_config = load_config(CONFIG_PATH)
        current_config_path_loaded = Path(CONFIG_PATH)
        logger.info(f"CLI RUNNER: Standard-Konfiguration von '{CONFIG_PATH}' geladen.")

    # --- Debugging-Punkt für Prompts (integriert) ---
    if "prompts" in app_config:
        logger.info("DEBUG: 'prompts' Sektion in app_config gefunden.")
        prompts_section = app_config["prompts"]
        
        # Prüfe 'raster_analysis_user_prompt_template'
        if "raster_analysis_user_prompt_template" in prompts_section:
            logger.info("DEBUG: 'raster_analysis_user_prompt_template' gefunden in app_config['prompts'].")
        else:
            logger.error("DEBUG: 'raster_analysis_user_prompt_template' NICHT gefunden in app_config['prompts'].")
            logger.error("DEBUG: Verfügbare Prompts in Sektion: %s", list(prompts_section.keys()))
        
        # Prüfe 'cgm_code_generation_user_prompt'
        if "cgm_code_generation_user_prompt" in prompts_section:
            logger.info("DEBUG: 'cgm_code_generation_user_prompt' gefunden in app_config['prompts'].")
        else:
            logger.error("DEBUG: 'cgm_code_generation_user_prompt' NICHT gefunden in app_config['prompts'].")
            
    else:
        logger.error("DEBUG: 'prompts' Sektion NICHT gefunden in app_config.")
    # --- Ende Debugging-Punkt ---


    # Der Orchestrator setzt "_active_run_strategy" und "logic_parameters" in der temporären Konfig
    # für den aktuellen Lauf. Diese haben Vorrang.
    explicit_model_strategy_config_for_run = app_config.get("_active_run_strategy")
    explicit_params_override_for_run = app_config.get("logic_parameters")

    # Wenn der Orchestrator eine explizite Strategie gesetzt hat, verwenden wir diese
# Wenn der Orchestrator eine explizite Strategie gesetzt hat, verwenden wir diese
    if explicit_model_strategy_config_for_run:
        logger.info("CLI RUNNER: Strategie und Parameter-Overrides aus temporärer Konfiguration (Orchestrator) angewendet.")
        strategy_name_for_run = None
        model_id_for_run = None
        location_for_run = None
    else:
        # Dies ist der Pfad für manuelle CLI-Aufrufe ohne Orchestrator
        if not args.strategy and not args.model_id:
            parser.error("Entweder --strategy oder --model_id muss angegeben werden.")
        
        strategy_name_for_run = args.strategy
        model_id_for_run = args.model_id
        location_for_run = args.location
        # Im manuellen Modus gibt es keine explizite Strategie-Konfiguration
        explicit_model_strategy_config_for_run = None

    # --- HIER WIRD DIE run_single_analysis FUNKTION AUFGERUFEN ---
    # Achte darauf, dass die run_single_analysis Funktion in cli_runner.py
    # die neuen Parameter akzeptiert und korrekt verarbeitet.
    run_single_analysis(
        image_path=args.image,
        strategy_name=strategy_name_for_run,
        config=app_config,
        model_id=model_id_for_run,
        location=location_for_run,
        explicit_model_strategy_config=explicit_model_strategy_config_for_run,
        explicit_params_override=explicit_params_override_for_run
    )
