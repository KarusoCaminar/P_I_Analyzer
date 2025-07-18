# 📦 Standardbibliotheken
import os
from pathlib import Path
import json
import sys
import threading
import time
import queue
import shutil
import logging
import traceback
import subprocess
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Union, cast, Dict, Any, List, Optional

# 🧪 Drittanbieter-Bibliotheken
import fitz  # PyMuPDF
import dotenv
from google.cloud.aiplatform import log_params
import yaml  # Für .yaml Konfiguration
from PIL import Image  # falls du später was mit Bildern machst (nicht aktiv im Code gezeigt)
import diskcache 

# 🔄 GUI (tkinter)
import tkinter as tk
from tkinter import (
    ttk, filedialog, messagebox, scrolledtext,
    Listbox, MULTIPLE, END, simpledialog
)

# 🌐 Projekt-Module
from llm_handler import LLM_Handler
from knowledge_bases import KnowledgeManager
from core_processor import Core_Processor, ProgressCallback
import utils  # z. B. für Logging, Hilfsfunktionen

# 🌱 .env initialisieren
dotenv.load_dotenv()
class GuiLogHandler(logging.Handler):
    """Ein benutzerdefinierter Logging-Handler, der Log-Records in eine Tkinter-Queue legt."""
    def __init__(self, queue_instance):
        super().__init__()
        self.queue = queue_instance

    def emit(self, record):
        self.queue.put(('LOG_MESSAGE', self.format(record)))

# ==============================================================================
# === ZENTRALE KONFIGURATION LADEN (NEUE, SAUBERE VERSION) ===
# ==============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_SCRIPT_DIR_PATH = Path(os.path.dirname(os.path.abspath(__file__)))
_CONFIG_PATH_OBJ = _SCRIPT_DIR_PATH / "config.yaml"

def load_config(path: Union[str, Path]) -> Optional[Dict[str, Any]]: # <- NEU: Path kann auch String sein
    """Lädt die zentrale YAML-Konfigurationsdatei."""
    try:
        # Konvertiere explizit zu Path, falls es ein String ist
        _path_obj = Path(path) if isinstance(path, str) else path
        with _path_obj.open('r', encoding='utf-8') as f: # <- Nutzt die .open() Methode des Path-Objekts
            return yaml.safe_load(f)
    except FileNotFoundError:
        logging.critical(f"FATAL: Konfigurationsdatei '{path}' nicht gefunden.")
        messagebox.showerror("Kritischer Fehler", f"Konfigurationsdatei '{path}' nicht gefunden!")
        return None
    except yaml.YAMLError as e:
        logging.critical(f"FATAL: Fehler beim Parsen der Konfigurationsdatei '{path}': {e}")
        messagebox.showerror("Kritischer Fehler", f"Fehler in der Konfigurationsdatei '{path}':\n{e}")
        return None
    except Exception as e:
        logging.critical(f"FATAL: Ein unerwarteter Fehler ist beim Laden der Konfiguration aufgetreten: {e}", exc_info=True)
        messagebox.showerror("Kritischer Fehler", f"Unerwarteter Fehler beim Laden der Konfiguration:\n{e}")
        return None

# Lade die gesamte App-Konfiguration beim Start
# KORRIGIERT: Übergibt das explizite Path-Objekt
APP_CONFIG = load_config(path=_CONFIG_PATH_OBJ)
if not APP_CONFIG:
    sys.exit("Anwendung kann ohne gültige Konfiguration nicht starten.")

# Die Pfade und Modelle werden jetzt aus der geladenen Konfiguration bezogen
ELEMENT_TYPE_LIST_PATH = os.path.join(SCRIPT_DIR, APP_CONFIG["paths"]["element_type_list"])
LEARNING_DB_PATH = os.path.join(SCRIPT_DIR, APP_CONFIG["paths"]["learning_db"])
AVAILABLE_MODELS = APP_CONFIG["models"]
AVAILABLE_MODELS_DISPLAY_NAMES = list(AVAILABLE_MODELS.keys())

# Umgebungsvariablen bleiben für sensitive Daten
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID")
GCP_LOCATION = os.getenv("GCP_LOCATION", "us-central1")

# NEU: "Fail-Fast"-Prüfung für kritische Konfiguration
if not GCP_PROJECT_ID:
    # Logge einen kritischen Fehler und zeige eine unmissverständliche Fehlermeldung an
    logging.critical("FATAL: GCP_PROJECT_ID nicht in der .env-Datei gesetzt. Anwendung kann nicht starten.")
    # Zeige die Fehlermeldung in einer eigenen, einfachen Tkinter-Box an, falls die Haupt-GUI noch nicht voll da ist.
    root = tk.Tk()
    root.withdraw() # Verstecke das leere Hauptfenster
    messagebox.showerror("Kritischer Konfigurationsfehler", "GCP_PROJECT_ID wurde nicht in der .env-Datei gefunden.\n\nBitte fügen Sie die Variable hinzu, um die Anwendung zu starten.")
    sys.exit(1) # Beende das Programm mit einem Fehlercode

# KORRIGIERT: Implementierung des ProgressCallback Interfaces
class CoreProgressCallback(ProgressCallback):
    def __init__(self, gui_update_callback_func, progress_bar_update_func, status_label_update_func):
        self.gui_update_callback_func = gui_update_callback_func
        self.progress_bar_update_func = progress_bar_update_func
        self.status_label_update_func = status_label_update_func

    def update_progress(self, value: int, message: str):
        self.gui_update_callback_func(self.progress_bar_update_func, value=value)
        self.gui_update_callback_func(self.status_label_update_func, text=message) # Nachricht für Statusleiste
    
    def update_status_label(self, text: str):
        self.gui_update_callback_func(self.status_label_update_func, text=text)

# =============================================================================
#  HAUPTANWENDUNG
# =============================================================================
class IntegratedSuite(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("P&I Diagram Analyse Suite v2.2")
        self.geometry("950x850")

        # Setup Logging
        self.gui_queue = queue.Queue()
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        for handler in list(root_logger.handlers):
            root_logger.removeHandler(handler)
        
        console_handler = logging.StreamHandler(sys.stdout)
        gui_formatter = logging.Formatter('[%(asctime)s - %(levelname)s] %(message)s', datefmt='%H:%M:%S')
        console_handler.setFormatter(gui_formatter)
        root_logger.addHandler(console_handler)
        
        gui_handler = GuiLogHandler(self.gui_queue)
        gui_handler.setFormatter(gui_formatter)
        root_logger.addHandler(gui_handler)
        
        logging.info("GUI-Logging System initialisiert.") 

        # Setup UI
        style = ttk.Style(self)
        style.theme_use('clam') 
        
        style.configure("TButton", padding=6, font=('Helvetica', 9))
        style.configure("Accent.TButton", 
                        font=('Helvetica', 10, 'bold'), 
                        padding=8, 
                        background="#4CAF50", 
                        foreground="white", 
                        focusthickness=3, 
                        focuscolor="none") 
        style.map("Accent.TButton", 
                  background=[('active', '#66BB6A')], 
                  foreground=[('active', 'white')])

        style.configure("TLabelframe", background="#f0f0f0") 
        style.configure("TLabelframe.Label", font=('Helvetica', 10, 'bold'), foreground="#333333") 

        style.configure("TCombobox", fieldbackground="white")
        style.map("TCombobox", fieldbackground=[('readonly', 'white')])

        style.configure("TProgressbar", background="#4CAF50", troughcolor="#e0e0e0", bordercolor="#bbb")     

        main_pane = ttk.PanedWindow(self, orient=tk.VERTICAL)
        status_frame = ttk.Frame(self, padding=(10, 0, 10, 10))
        status_frame.pack(side="top", fill="x", expand=False)

        self.status_label = ttk.Label(status_frame, text="Bereit.")
        self.status_label.pack(side="left", fill="x", expand=True)
        
        self.progress_bar = ttk.Progressbar(status_frame, orient='horizontal', mode='determinate', length=200)
        self.progress_bar.pack(side="right")

        main_pane.pack(expand=True, fill=tk.BOTH, padx=10, pady=(0, 10))

        self.notebook = ttk.Notebook(main_pane)
        main_pane.add(self.notebook, weight=3)

        log_frame = ttk.LabelFrame(main_pane, text="Globales Log", padding=10)
        self.log_text = scrolledtext.ScrolledText(log_frame, wrap=tk.WORD, state='disabled', font=("Courier New", 9), bg='#2e2e2e', fg='#ffffff')
        self.log_text.pack(expand=True, fill=tk.BOTH)
        main_pane.add(log_frame, weight=1)

        self.log_text.tag_config('INFO', foreground='#a0a0a0')  
        self.log_text.tag_config('WARNING', foreground='#FFA500') 
        self.log_text.tag_config('ERROR', foreground='#FF0000')   
        self.log_text.tag_config('CRITICAL', foreground='#FF0000', font=("Courier New", 9, 'bold')) 
        self.log_text.tag_config('SUBPROCESS_LOG', foreground='#00BFFF') # NEU: Für Subprozess-Logs

        logging.info("Initialisiere Backend-Module...")
        
        assert GCP_PROJECT_ID is not None, "Kritischer Fehler: GCP_PROJECT_ID nicht gefunden."

        # 1. Weise die Konfiguration SOFORT einer Instanz-Variable zu.
        #    Ab hier weiß Pylance, dass self.app_config immer ein Dictionary ist.
        self.app_config = APP_CONFIG

        # 2. Erstelle jetzt alle Backend-Module und übergebe IMMER die sichere Instanz-Variable.
        # KORRIGIERT: Explicitly cast self.app_config to Dict[str, Any] to satisfy Pylance,
        # as load_config can return None but sys.exit() handles that case.
        # This tells Pylance that self.app_config will indeed be a Dict[str, Any] here.
        self.llm_handler = LLM_Handler(GCP_PROJECT_ID, GCP_LOCATION, cast(Dict[str, Any], self.app_config))
        self.knowledge_manager = KnowledgeManager(ELEMENT_TYPE_LIST_PATH, LEARNING_DB_PATH, self.llm_handler, cast(Dict[str, Any], self.app_config))
        
        self.core_processor = Core_Processor(
            self.llm_handler, 
            self.knowledge_manager, 
            {}, 
            cast(Dict[str, Any], self.app_config)
        )
        
        logging.info("Backend-Module initialisiert. Anwendung bereit.")
        
        self.analysis_tab = AnalysisTab(self.notebook, self)
        self.converter_tab = ConverterTab(self.notebook, self)
        self.benchmark_tab = BenchmarkTab(self.notebook, self)
        self.pre_training_tab = PreTrainingTab(self.notebook, self)
        self.maintenance_tab = MaintenanceTab(self.notebook, self) # NEU
        
        self.notebook.add(self.analysis_tab, text=" Einzelbild-Analyse ")
        self.notebook.add(self.converter_tab, text=" PDF Konverter ")
        self.notebook.add(self.benchmark_tab, text=" Benchmark Suite ")
        self.notebook.add(self.pre_training_tab, text=" Symbol-Vortraining ")
        self.notebook.add(self.maintenance_tab, text=" Wartung ") # NEU
        
        self.process_queue()

    def queue_gui_update(self, func, *args, **kwargs):
        self.gui_queue.put((func, args, kwargs))

    def process_queue(self):
        try:
            while True:
                task_tuple = self.gui_queue.get_nowait()
                task_type = task_tuple[0]
                if task_type == 'LOG_MESSAGE':
                    formatted_message = task_tuple[1]
                    log_level = 'INFO' 
                    if ' - WARNING' in formatted_message:
                        log_level = 'WARNING'
                    elif ' - ERROR' in formatted_message:
                        log_level = 'ERROR'
                    elif ' - CRITICAL' in formatted_message:
                        log_level = 'CRITICAL'

                    if self.log_text.winfo_exists():
                        self.log_text.configure(state='normal')
                        self.log_text.insert(tk.END, formatted_message + "\n", log_level) 
                        self.log_text.see(tk.END)
                        self.log_text.configure(state='disabled')
                else: # Für andere GUI-Updates
                    func, args, kwargs = task_type, task_tuple[1], task_tuple[2] # Korrigiert: task_tuple[1] und task_tuple[2] sind args und kwargs
                    func(*args, **kwargs)
        except queue.Empty:
            pass
        finally:
            self.after(100, self.process_queue)

    def clear_gui_log(self):
        """Löscht den Inhalt des Log-Fensters in der GUI."""
        if messagebox.askyesno("Bestätigung", "Möchten Sie das Log-Fenster wirklich leeren?"):
            self.log_text.config(state=tk.NORMAL)
            self.log_text.delete('1.0', tk.END)
            self.log_text.config(state=tk.DISABLED)
            logging.info("GUI-Log-Fenster wurde manuell geleert.")

    def clear_llm_cache(self):
        """Löscht den LLM-Disk-Cache."""
        if not self.app_config: return

        if messagebox.askyesno("Bestätigung", "Sind Sie sicher, dass Sie den gesamten LLM-Cache (.pni_analyzer_cache) löschen wollen?\n\nAlle zwischengespeicherten KI-Antworten gehen verloren."):
            logging.info("Lösche LLM-Cache...")
            try:
                cache_dir_name = self.app_config.get('paths', {}).get('llm_cache_dir', '.pni_analyzer_cache')
                cache_path = Path(cache_dir_name)
                if cache_path.exists():
                    # KORREKTUR: Zuerst die Datenbankverbindung schließen
                    self.llm_handler.disk_cache.close()
                    # Dann den Ordner löschen
                    shutil.rmtree(cache_path)
                    # Und danach den Cache für die weitere Nutzung neu initialisieren
                    cache_size_gb = self.app_config.get('logic_parameters', {}).get('llm_cache_size_gb', 1)
                    self.llm_handler.disk_cache = diskcache.Cache(cache_path, size_limit=int(cache_size_gb * 1024 * 1024 * 1024))
                    logging.info(f"LLM-Cache '{cache_path}' erfolgreich gelöscht und neu initialisiert.")
                    messagebox.showinfo("Erfolg", "Der LLM-Cache wurde erfolgreich gelöscht.")
                else:
                    logging.info("LLM-Cache existiert nicht, nichts zu löschen.")
                    messagebox.showinfo("Info", "Der LLM-Cache existiert nicht.")
            except Exception as e:
                logging.error(f"Fehler beim Löschen des LLM-Cache: {e}")
                messagebox.showerror("Fehler", f"Konnte den LLM-Cache nicht löschen:\n{e}")

    def clear_learned_symbols(self):
        """Löscht die gelernten Symbole (Bilder und Datenbankeinträge)."""
        if not self.app_config: return # Sicherheitsprüfung

        if messagebox.askyesno("Bestätigung", "Sind Sie sicher, dass Sie alle gelernten Symbole löschen wollen?\n\nDies betrifft den Ordner 'learned_symbols_images' und die 'symbol_library' in der learning_db.json."):
            logging.info("Lösche gelernte Symbole...")
            try:
                base_knowledge_dir = Path(self.app_config["paths"]["learning_db"]).parent
                symbol_images_dir = base_knowledge_dir / "learned_symbols_images"
                if symbol_images_dir.exists():
                    shutil.rmtree(symbol_images_dir)
                    logging.info(f"Ordner '{symbol_images_dir}' erfolgreich gelöscht.")

                self.knowledge_manager.learning_database["symbol_library"] = {}
                self.knowledge_manager.save_learning_database()
                self.knowledge_manager._build_vector_index()
                logging.info("Symbol-Bibliothek in der learning_db.json wurde geleert.")
                messagebox.showinfo("Erfolg", "Alle gelernten Symbole wurden erfolgreich gelöscht.")
            except Exception as e:
                logging.error(f"Fehler beim Löschen der gelernten Symbole: {e}")
                messagebox.showerror("Fehler", f"Konnte die gelernten Symbole nicht löschen:\n{e}")

# =============================================================================
#  TAB 1: Einzelbild-Analyse
# =============================================================================
class AnalysisTab(ttk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent, padding="15")
        self.controller = controller
        self.selected_image_paths = []
        self.model_comboboxes = {} 
        self.model_desc_labels = {} 
        
        # KORRIGIERT: Tkinter-Variablen für BBox-Verfeinerungsparameter initialisieren
        # Die Werte werden aus der geladenen APP_CONFIG des Controllers geholt.
        self.min_bbox_quality_var = tk.DoubleVar(value=controller.app_config.get('logic_parameters', {}).get('min_quality_to_keep_bbox', 0.5))
        self.max_bbox_iter_var = tk.IntVar(value=controller.app_config.get('logic_parameters', {}).get('max_bbox_refinement_iterations', 3))
        self.min_visual_match_score_var = tk.DoubleVar(value=controller.app_config.get('logic_parameters', {}).get('min_visual_match_score', 0.70))

        file_frame = ttk.LabelFrame(self, text="1. P&ID Bild(er) auswählen", padding="10")
        file_frame.pack(fill="both", pady=5, expand=True) # KORRIGIERT: fill="both" für LabelFrame

        file_selection_row = ttk.Frame(file_frame)
        file_selection_row.pack(fill="x", expand=True, pady=2) # KORRIGIERT: pady hinzugefügt
        ttk.Button(file_selection_row, text="Bilder auswählen...", command=self.select_files).pack(side="left", padx=(0,10))
        self.files_label = ttk.Label(file_selection_row, text="Keine Dateien ausgewählt", foreground="gray")
        self.files_label.pack(side="left", expand=True, fill="x")

        strategy_frame = ttk.LabelFrame(self, text="2. Analyse-Strategie definieren (Modelle pro Phase)", padding="10")
        strategy_frame.pack(fill="both", pady=5, expand=True) # KORRIGIERT: fill="both" für LabelFrame
        
        self.strategy_vars = {}
        tasks = {
            "meta_model": "Phase 1 (Metadaten & Legende)",
            "hotspot_model": "Phase 2.1 (Hotspot-Findung)",
            "detail_model": "Phase 2.2 (Detail-Analyse)",
            "correction_model": "Phase 3 (Selbstkorrektur)"
        }
        
        for task_key, task_name in tasks.items():
            row = ttk.Frame(strategy_frame)
            row.pack(fill="x", pady=2, expand=True)
            
            ttk.Label(row, text=f"{task_name}:", width=25).pack(side="left")
            
            var = tk.StringVar()
            combobox = ttk.Combobox(row, textvariable=var, values=AVAILABLE_MODELS_DISPLAY_NAMES, state="readonly")
            combobox.pack(side="left", fill="x", expand=True, padx=(0, 5))
            self.strategy_vars[task_key] = var
            self.model_comboboxes[task_key] = combobox

            desc_label = ttk.Label(row, text="", foreground="gray", font=('Helvetica', 8))
            desc_label.pack(side="right", padx=5, anchor="e")
            self.model_desc_labels[task_key] = desc_label
            
            var.trace_add("write", lambda name, index, mode, v=var, dl=desc_label, tk=task_key: self._update_model_description(v, dl, tk))

        self.strategy_vars["meta_model"].set("Google Gemini 2.5 Flash")
        self.strategy_vars["hotspot_model"].set("Google Gemini 2.5 Flash")
        self.strategy_vars["detail_model"].set("Google Gemini 2.5 Flash")
        self.strategy_vars["correction_model"].set("Google Gemini 2.5 Flash")
        
        for task_key, var in self.strategy_vars.items():
            if task_key in self.model_desc_labels:
                self._update_model_description(var, self.model_desc_labels[task_key], task_key)

        bbox_params_frame = ttk.LabelFrame(self, text="3. BBox-Verfeinerung & Visuelle Prüfung", padding="10")
        bbox_params_frame.pack(fill="both", pady=5, expand=True) 
        bbox_params_frame.grid_columnconfigure(1, weight=1) 

        ttk.Label(bbox_params_frame, text="Minimale BBox-Qualität (0.0-1.0):").grid(row=0, column=0, sticky="w", pady=2, padx=5)
        ttk.Scale(bbox_params_frame, from_=0.0, to=1.0, orient="horizontal", variable=self.min_bbox_quality_var, length=200).grid(row=0, column=1, sticky="ew", pady=2, padx=5)
        ttk.Label(bbox_params_frame, textvariable=self.min_bbox_quality_var, width=5).grid(row=0, column=2, sticky="w", pady=2, padx=5)

        ttk.Label(bbox_params_frame, text="Max. BBox-Verfeinerung Iterationen:").grid(row=1, column=0, sticky="w", pady=2, padx=5)
        ttk.Spinbox(bbox_params_frame, from_=1, to=10, textvariable=self.max_bbox_iter_var, width=5).grid(row=1, column=1, sticky="w", pady=2, padx=5)

        ttk.Label(bbox_params_frame, text="Min. vis. Symbol-Match-Score (0.0-1.0):").grid(row=2, column=0, sticky="w", pady=2, padx=5)
        ttk.Scale(bbox_params_frame, from_=0.0, to=1.0, orient="horizontal", variable=self.min_visual_match_score_var, length=200).grid(row=2, column=1, sticky="ew", pady=2, padx=5)
        ttk.Label(bbox_params_frame, textvariable=self.min_visual_match_score_var, width=5).grid(row=2, column=2, sticky="w", pady=2, padx=5)

        action_frame = ttk.Frame(self)
        action_frame.pack(fill="both", pady=10, expand=True) # KORRIGIERT: fill="both" und expand=True für action_frame
        
        self.start_button = ttk.Button(action_frame, text="Analyse starten", command=self.start_analysis, style="Accent.TButton", state=tk.DISABLED)
        self.start_button.pack(ipady=8, fill="x", expand=True)

    def _update_model_description(self, var: tk.StringVar, desc_label_widget: ttk.Label, task_key: str):
        selected_display_name = var.get()
        model_info = AVAILABLE_MODELS.get(selected_display_name)
        if model_info:
            desc_label_widget.config(text=model_info.get('description', ''))
        else:
            desc_label_widget.config(text="")

    def select_files(self):
        filetypes = [("Bilddateien", "*.png *.jpg *.jpeg"), ("Alle Dateien", "*.*")]
        paths = filedialog.askopenfilenames(filetypes=filetypes)
        if paths:
            self.selected_image_paths = paths
            self.files_label.config(text=f"{len(paths)} Datei(en) ausgewählt")
            self.start_button.config(state=tk.NORMAL)
        else:
            self.selected_image_paths = []
            self.files_label.config(text="Keine Dateien ausgewählt")
            self.start_button.config(state=tk.DISABLED)
            
    def start_analysis(self):
        if not self.selected_image_paths:
            messagebox.showerror("Fehler", "Bitte wählen Sie zuerst eine oder mehrere Bilddateien aus.")
            return
        
        strategy_config = {}
        try:
            for task_key, var in self.strategy_vars.items():
                display_name = var.get()
                if display_name in AVAILABLE_MODELS:
                    strategy_config[task_key] = AVAILABLE_MODELS[display_name]
                else:
                    messagebox.showerror("Fehler", f"Ausgewähltes Modell '{display_name}' nicht gefunden. Bitte prüfen Sie Ihre Konfiguration.")
                    return
        except KeyError:
            messagebox.showerror("Fehler", "Bitte für jeden Schritt ein gültiges Modell auswählen.")
            return

        # NEU: Lese die Werte aus der GUI und packe sie in ein Dictionary
        # Diese Werte überschreiben die Standardwerte aus der config.yaml im Core_Processor
        analysis_params_from_gui = {
            "min_quality_to_keep_bbox": self.min_bbox_quality_var.get(),
            "max_bbox_refinement_iterations": self.max_bbox_iter_var.get(),
            "min_visual_match_score": self.min_visual_match_score_var.get(),
            # Hier könnten weitere GUI-gesteuerte Parameter hinzugefügt werden
        }


        self.start_button.config(state=tk.DISABLED)
        # Deaktiviere alle Tabs während der Analyse
        for tab_id in self.controller.notebook.tabs():
            self.controller.notebook.tab(tab_id, state="disabled")
        
        for task_key in self.model_comboboxes: 
            combobox = self.model_comboboxes[task_key]
            combobox.config(state="disabled")
        
        core_progress_callback = CoreProgressCallback(
            self.controller.queue_gui_update,
            self.controller.progress_bar.config,
            self.controller.status_label.config
        )

        # KORRIGIERT: Übergabe von analysis_params_from_gui an den Worker-Thread
        thread = threading.Thread(target=self.run_analysis_worker, 
                                  args=(self.selected_image_paths, strategy_config, 
                                        core_progress_callback, analysis_params_from_gui)) 
        thread.daemon = True
        thread.start()

    def run_analysis_worker(self, image_paths, strategy_config, 
                            core_progress_callback: CoreProgressCallback, 
                            analysis_params_from_gui: Dict[str, Any]): # NEUER PARAMETER
        try:
            total_images = len(image_paths)
            for i, image_path in enumerate(image_paths):
                logging.info(f"Starte Analyse für Bild {i+1}/{total_images}: {os.path.basename(image_path)}")
                core_progress_callback.update_status_label(text=f"Analysiere Bild {i+1}/{total_images}...")
                start_time = time.time()
                
                try:
                    main_model_name_for_folder = strategy_config.get('detail_model', {}).get('id', 'unknown_model')
                    output_dir = utils.create_output_directory(image_path, main_model_name_for_folder)
                    if not output_dir:
                        logging.error(f"FATAL: Ausgabeordner für {image_path} konnte nicht erstellt werden. Überspringe.")
                        continue
                    
                    utils.setup_logging(output_dir)
                    logging.info(f"Log-Datei für diesen Lauf wird unter {os.path.join(output_dir, 'run_log.txt')} gespeichert.")

                    self.controller.core_processor.model_strategy = strategy_config
                    final_results = self.controller.core_processor.run_full_pipeline(
                        image_path=image_path,
                        progress_callback=core_progress_callback,
                        output_dir=output_dir,
                        analysis_params_override=analysis_params_from_gui
                    )
                    
                    if final_results:
                        logging.info(f"Pipeline für {os.path.basename(image_path)} erfolgreich abgeschlossen.")
                        # KORRIGIERT: Sicherstellen, dass das Speichern der Lern-Datenbank nur einmal nach dem letzten Bild erfolgt.
                        # self.controller.knowledge_manager.save_learning_database() # Entfernt von hier
                        core_progress_callback.update_progress(value=100, message="Analyse abgeschlossen.")
                        logging.info(f"Alle Artefakte wurden erfolgreich in {output_dir} gespeichert.")

                except Exception as e:
                    logging.error(f"FATALER FEHLER bei der Analyse von {image_path}: {e}", exc_info=True)
                    core_progress_callback.update_progress(value=100, message=f"Analyse FEHLGESCHLAGEN für {os.path.basename(image_path)}.")
        
        finally:
            logging.info("Analyse-Batch abgeschlossen. Gebe GUI wieder frei.")
            # KORRIGIERT: Lern-Datenbank nach Abschluss des gesamten Batches speichern
            try:
                self.controller.knowledge_manager.save_learning_database()
            except Exception as e:
                logging.error(f"Fehler beim Speichern der Lern-Datenbank nach Batch: {e}", exc_info=True)

            self.controller.queue_gui_update(self.start_button.config, state=tk.NORMAL)
            self.controller.queue_gui_update(self.controller.status_label.config, text="Analyse abgeschlossen. Bereit für nächsten Lauf.")
            for tab_id in self.controller.notebook.tabs():
                self.controller.queue_gui_update(self.controller.notebook.tab, tab_id, state="normal")
            for task_key in self.model_comboboxes:
                combobox = self.model_comboboxes[task_key]
                self.controller.queue_gui_update(combobox.config, state="readonly")

# =============================================================================
#  TAB 2: PDF Konverter
# =============================================================================
class ConverterTab(ttk.Frame):
    def __init__(self, parent, controller: 'IntegratedSuite'): 
        super().__init__(parent, padding="15")
        self.controller = controller
        
        self.pdf_path_var = tk.StringVar()
        self.output_dir_var = tk.StringVar()

        ttk.Button(self, text="PDF auswählen...", command=self.select_pdf).pack(pady=5)
        self.pdf_label = ttk.Label(self, text="Keine PDF ausgewählt")
        self.pdf_label.pack()
        ttk.Button(self, text="Ausgabeordner auswählen...", command=self.select_output_dir).pack(pady=5)
        self.output_dir_label = ttk.Label(self, text="Kein Ausgabeordner ausgewählt")
        self.output_dir_label.pack()
        ttk.Button(self, text="Konvertieren", command=self.convert, style="Accent.TButton").pack(pady=20)

    def select_pdf(self):
        path = filedialog.askopenfilename(filetypes=[("PDF", "*.pdf")])
        if path:
            self.pdf_path_var.set(path)
            self.pdf_label.config(text=os.path.basename(path))

    def select_output_dir(self):
        path = filedialog.askdirectory()
        if path:
            self.output_dir_var.set(path)
            # Korrektur der fehlenden Klammer:
            self.output_dir_label.config(text=path) 
        
    def convert(self):
        """Startet die Konvertierung, indem es das zentrale Werkzeug aus utils.py aufruft."""
        pdf_path_str = self.pdf_path_var.get()
        out_path_str = self.output_dir_var.get()
        
        if not pdf_path_str or not out_path_str:
            messagebox.showerror("Fehler", "Bitte eine PDF-Datei und einen Ausgabeordner auswählen.")
            return

        # Wir rufen jetzt einfach unser neues Werkzeug auf!
        # Wichtig: Wir wandeln die Pfade in Path-Objekte um, wie es die Funktion erwartet.
        created_files = utils.convert_pdf_to_images(Path(pdf_path_str), Path(out_path_str))

        if created_files:
            messagebox.showinfo("Erfolg", f"{len(created_files)} Seite(n) erfolgreich konvertiert!")
        else:
            messagebox.showerror("Fehler", "Die Konvertierung ist fehlgeschlagen. Bitte prüfen Sie das Log-Fenster für Details.")

# =============================================================================
#  TAB 3: Benchmark Suite (FINALE VERSION)
# =============================================================================
class BenchmarkTab(ttk.Frame):
    def __init__(self, parent, controller: 'IntegratedSuite'):
        super().__init__(parent, padding="15")
        self.controller = controller
        
        folder_frame = ttk.LabelFrame(self, text="1. Benchmark-Ordner auswählen (enthält P&ID Bilder)", padding="10")
        folder_frame.pack(fill="x", pady=5)
        self.folder_var = tk.StringVar()
        ttk.Entry(folder_frame, textvariable=self.folder_var).pack(side="left", fill="x", expand=True, padx=(0,5))
        ttk.Button(folder_frame, text="Durchsuchen...", command=self.select_folder).pack(side="left")

        model_frame = ttk.LabelFrame(self, text="2. Modelle für den Test auswählen", padding="10")
        model_frame.pack(fill="x", pady=5)
        self.model_listbox = Listbox(model_frame, selectmode=MULTIPLE, height=len(AVAILABLE_MODELS))
        self.model_listbox.pack(fill="x", expand=True)
        for name in AVAILABLE_MODELS.keys():
            self.model_listbox.insert(END, name)

        parallel_frame = ttk.Frame(self)
        parallel_frame.pack(fill="x", pady=10, anchor="w")
        ttk.Label(parallel_frame, text="Anzahl paralleler Prozesse:").pack(side="left")
        self.parallel_var = tk.IntVar(value=2)
        ttk.Spinbox(parallel_frame, from_=1, to=8, textvariable=self.parallel_var, width=5).pack(side="left", padx=5)

        self.start_button = ttk.Button(self, text="Benchmark starten", command=self.start_benchmark, style="Accent.TButton")
        self.start_button.pack(pady=20, ipady=5, fill="x")

    def select_folder(self):
        path = filedialog.askdirectory()
        if path:
            self.folder_var.set(path)
            
    def start_benchmark(self):
        folder_path = self.folder_var.get()
        selected_indices = self.model_listbox.curselection()
        if not folder_path or not selected_indices:
            messagebox.showerror("Fehler", "Bitte Ordner und mindestens ein Modell auswählen.")
            return

        self.start_button.config(state=tk.DISABLED)
        self.controller.queue_gui_update(self.controller.notebook.tab, self.controller.notebook.select(), state="disabled") 

        models_to_run = {}
        for i in selected_indices:
            model_display_name = self.model_listbox.get(i)
            if model_display_name in AVAILABLE_MODELS:
                models_to_run[model_display_name] = AVAILABLE_MODELS[model_display_name]
            else:
                logging.warning(f"Ausgewähltes Modell '{model_display_name}' nicht in der Konfiguration gefunden. Wird übersprungen.")
        max_workers = self.parallel_var.get()
        image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        if not image_files:
            messagebox.showerror("Fehler", "Keine Bilddateien im ausgewählten Ordner gefunden.")
            self.start_button.config(state=tk.NORMAL)
            self.controller.queue_gui_update(self.controller.notebook.tab, self.controller.notebook.select(), state="normal")
            return

        thread = threading.Thread(target=self.run_benchmark_worker, args=(folder_path, image_files, models_to_run, max_workers))
        thread.daemon = True
        thread.start()

    def run_benchmark_worker(self, folder_path, image_files, models_to_run, max_workers):
        logging.info(f"Starte Benchmark für {len(image_files)} Bilder und {len(models_to_run)} Modelle...")
        
        cli_script_path = os.path.join(SCRIPT_DIR, "cli_runner.py")
        tasks = []
        for image_file in image_files:
            for model_name, model_data in models_to_run.items():
                image_full_path = os.path.join(folder_path, image_file)
                # KORRIGIERT: Übergabe von model_id und location an den cli_runner
                tasks.append((model_name, model_data.get("id"), model_data.get("location"), image_full_path))
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self.run_single_subprocess, task, cli_script_path): task for task in tasks}
            for future in as_completed(futures):
                future.result() # Wartet auf das Ergebnis, um sicherzustellen

        logging.info("Alle Analysen abgeschlossen. Starte finale KPI-Auswertung...")
        kpi_script_path = os.path.join(SCRIPT_DIR, "evaluate_kpis.py")
        if os.path.exists(kpi_script_path):
            # KORRIGIERT: KPI-Lauf mit dem Ordner als direktes Argument
            self.run_kpi_subprocess(folder_path, kpi_script_path)
            logging.info("KPI-Report erfolgreich erstellt. Details im Ordner.")
        else:
            logging.error("FEHLER: evaluate_kpis.py nicht gefunden.")

        logging.info("Benchmark-Prozess vollständig abgeschlossen.")
        
        self.controller.queue_gui_update(self.start_button.config, state=tk.NORMAL)
        self.controller.queue_gui_update(self.controller.notebook.tab, self.controller.notebook.select(), state="normal")

    def run_single_subprocess(self, task, script_path):
        """Führt einen einzelnen Subprozess für eine Bildanalyse aus und loggt den Output."""
        model_name, model_id, location, image_full_path = task
        
        # KORRIGIERT: Befehl mit --model_id und --location
        command = [
            sys.executable, str(script_path),
            "--image", str(image_full_path),
            "--model_id", model_id,
            "--location", location
        ]
        log_source = f"{model_name}-{os.path.basename(image_full_path)}"

        try:
            process = subprocess.Popen(
                command, 
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
                text=True, encoding='utf-8', errors='ignore'
            )
            if process.stdout:
                for line in iter(process.stdout.readline, ''):
                    if line:
                        self.controller.queue_gui_update(self.controller.log_text.insert, tk.END, f"[{log_source}] {line}", ("SUBPROCESS_LOG",))
                        self.controller.queue_gui_update(self.controller.log_text.see, tk.END)

            process.wait()
            if process.returncode != 0:
                 logging.error(f"Subprozess '{log_source}' beendet mit Fehlercode {process.returncode}.")
            else:
                 logging.info(f"Subprozess '{log_source}' erfolgreich beendet.")
        except Exception as e:
            logging.error(f"Ein kritischer Fehler ist beim Starten des Subprozesses für '{log_source}' aufgetreten: {e}", exc_info=True)

    def run_kpi_subprocess(self, folder_path, script_path):
        """Führt den KPI-Auswertungssubprozess aus."""
        command = [sys.executable, str(script_path), str(folder_path)]
        log_source = "KPI-REPORT"
        
        try:
            process = subprocess.Popen(
                command, 
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
                text=True, encoding='utf-8', errors='ignore'
            )
            if process.stdout:
                for line in iter(process.stdout.readline, ''):
                    if line:
                        self.controller.queue_gui_update(self.controller.log_text.insert, tk.END, f"[{log_source}] {line}", ("SUBPROCESS_LOG",))
                        self.controller.queue_gui_update(self.controller.log_text.see, tk.END)

            process.wait()
            if process.returncode != 0:
                 logging.error(f"KPI-Report-Subprozess beendet mit Fehlercode {process.returncode}.")
            else:
                 logging.info(f"KPI-Report-Subprozess erfolgreich beendet.")
        except Exception as e:
            logging.error(f"Ein kritischer Fehler ist beim Starten des KPI-Report-Subprozesses aufgetreten: {e}", exc_info=True)

# =============================================================================
#  TAB 4: Vortraining & Wissensbasis (FINALE, VERBESSERTE VERSION)
# =============================================================================
class PreTrainingTab(ttk.Frame):
    def __init__(self, parent, controller: 'IntegratedSuite'):
        super().__init__(parent, padding="15")
        self.controller = controller
        # Konvertieren Sie SCRIPT_DIR in ein Path-Objekt, bevor Sie den / Operator verwenden.
        self.pre_training_path = Path(SCRIPT_DIR) / "pretraining_symbols"
        self.report_path = self.pre_training_path / "pretraining_report.json"

        # --- UI-Layout mit zwei Spalten ---
        main_pane = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        main_pane.pack(fill=tk.BOTH, expand=True)

        # Linke Spalte: Steuerung
        left_frame = ttk.LabelFrame(main_pane, text="Steuerung", padding="10")
        main_pane.add(left_frame, weight=1)

        # Rechte Spalte: Informationsanzeige
        right_frame = ttk.LabelFrame(main_pane, text="Letzter Trainingslauf", padding="10")
        main_pane.add(right_frame, weight=1)

        # --- Linke Spalte: Inhalte füllen ---
        # 1. Modellauswahl
        model_frame = ttk.LabelFrame(left_frame, text="1. Modell für Symbolerkennung auswählen", padding=10)
        model_frame.pack(fill="x", pady=5)
        
        self.model_var = tk.StringVar()
        # WICHTIG: Wir benutzen die zentrale Liste deiner verfügbaren Modelle
        model_combo = ttk.Combobox(model_frame, textvariable=self.model_var, values=AVAILABLE_MODELS_DISPLAY_NAMES, state="readonly")
        model_combo.pack(fill="x")
        # Setze einen sinnvollen Standardwert, z.B. das Flash-Modell
        model_combo.set("Google Gemini 2.5 Flash") 

        # 2. Aktion starten
        action_frame = ttk.LabelFrame(left_frame, text="2. Training durchführen", padding=10)
        action_frame.pack(fill="x", pady=15)
        
        info_text = "Startet den vollautomatischen Prozess:\n1. Dateien auswählen (Bilder/PDFs)\n2. Trainingsordner wird bereinigt\n3. Dateien werden vorbereitet & kopiert\n4. KI-Training wird ausgeführt"
        ttk.Label(action_frame, text=info_text, justify="left").pack(fill="x", pady=(0, 10))

        self.run_button = ttk.Button(action_frame, text="Vortraining starten...", command=self.start_pretraining_workflow, style="Accent.TButton")
        self.run_button.pack(ipady=8, fill="x", expand=True)
        
        self.status_label = ttk.Label(left_frame, text="Bereit.", justify="left")
        self.status_label.pack(pady=10, fill="x", side="bottom")

        # --- Rechte Spalte: Inhalte füllen ---
        ttk.Label(right_frame, text="Zuletzt verarbeitete Dateien:").pack(anchor="w")
        
        self.tree = ttk.Treeview(right_frame, columns=("status", "source"), show="headings", height=15)
        self.tree.heading("status", text="Status")
        self.tree.heading("source", text="Quelldatei")
        self.tree.column("status", width=80, anchor="w")
        self.tree.column("source", width=220, anchor="w")
        self.tree.pack(fill="both", expand=True, pady=(5,0))
        
        self._populate_last_run_info()

    def _populate_last_run_info(self):
        """Liest den letzten Report und füllt die Dateiliste."""
        for i in self.tree.get_children():
            self.tree.delete(i)
            
        if not self.report_path.exists():
            self.tree.insert("", "end", text="Info", values=("N/A", "Noch kein Training durchgeführt."))
            return
            
        try:
            with open(self.report_path, 'r', encoding='utf-8') as f:
                report_data = json.load(f)
            
            unique_sources = sorted(list(set([item.get('source', 'Unbekannt') for item in report_data])))
            if not unique_sources:
                 self.tree.insert("", "end", text="Info", values=("Leer", "Keine Symbole gefunden."))
            
            for source_file in unique_sources:
                self.tree.insert("", "end", values=("Gelernt", source_file))
        except (json.JSONDecodeError, IOError) as e:
            self.tree.insert("", "end", text="Fehler", values=("Fehler", "Report-Datei konnte nicht gelesen werden."))
            logging.error(f"Fehler beim Lesen des Pre-Training-Reports: {e}")

    def _clean_pretraining_folder(self):
        logging.info(f"Räume den Ordner '{self.pre_training_path}' für neue Trainingsdaten auf...")
        for item in self.pre_training_path.iterdir():
            if item.is_file():
                item.unlink()
            elif item.is_dir() and item.name != "old_learning":
                shutil.rmtree(item)

    def start_pretraining_workflow(self):
        selected_model_name = self.model_var.get()
        if not selected_model_name:
            messagebox.showerror("Fehler", "Bitte wählen Sie zuerst ein Modell für die Symbolerkennung aus.")
            return
        model_info = AVAILABLE_MODELS[selected_model_name]

        file_paths = filedialog.askopenfilenames(
            title="Wähle Bilder oder PDFs für das Vortraining aus",
            filetypes=[("Alle Trainingsdateien", "*.png *.jpg *.jpeg *.pdf")]
        )
        if not file_paths: return

        self.run_button.config(state=tk.DISABLED)
        self.controller.queue_gui_update(self.status_label.config, text="Bereite Trainingsdaten vor...")

        try:
            self._clean_pretraining_folder()
            for path_str in file_paths:
                path = Path(path_str)
                if path.suffix.lower() == ".pdf":
                    utils.convert_pdf_to_images(path, self.pre_training_path)
                elif path.suffix.lower() in [".png", ".jpg", ".jpeg"]:
                    shutil.copy(path, self.pre_training_path)
        except Exception as e:
            messagebox.showerror("Fehler", f"Die Trainingsdaten konnten nicht vorbereitet werden:\n{e}")
            self.run_button.config(state=tk.NORMAL)
            return
            
        thread = threading.Thread(target=self.run_pre_training_worker, args=(model_info,))
        thread.daemon = True
        thread.start()

    def run_pre_training_worker(self, model_info: dict):
        try:
            report: List[Dict] = self.controller.core_processor.run_pretraining(self.pre_training_path, model_info)

            with open(self.report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=4)
            
            final_msg = f"Vortraining abgeschlossen!\n\n{len(report)} Symbole verarbeitet.\nEin detaillierter Report wurde gespeichert."
            self.controller.queue_gui_update(messagebox.showinfo, "Erfolg", final_msg)
            
        except Exception as e:
            self.controller.queue_gui_update(messagebox.showerror, "Fehler", f"Fehler beim Vortraining: {e}")
            logging.error(f"Error during pre-training: {e}", exc_info=True)
        finally:
            self.controller.queue_gui_update(self.run_button.config, state=tk.NORMAL)
            self.controller.queue_gui_update(self.status_label.config, text="Bereit für das nächste Vortraining.")
            self.controller.queue_gui_update(self._populate_last_run_info)

# =============================================================================
#  TAB 5: Wartung & Werkzeuge
# =============================================================================
class MaintenanceTab(ttk.Frame):
    def __init__(self, parent, controller: 'IntegratedSuite'):
        super().__init__(parent, padding="15")
        self.controller = controller

        # Frame für Cache-Management
        cache_frame = ttk.LabelFrame(self, text="Cache & temporäre Daten", padding="10")
        cache_frame.pack(fill="x", pady=5)
        
        ttk.Label(cache_frame, text="Löscht zwischengespeicherte KI-Antworten, um neue Antworten zu erzwingen.").pack(anchor="w")
        ttk.Button(cache_frame, text="LLM-Cache (.pni_analyzer_cache) löschen", command=self.controller.clear_llm_cache).pack(pady=5, fill="x")

        # Frame für Wissensbasis-Management
        kb_frame = ttk.LabelFrame(self, text="Wissensbasis & Training", padding="10")
        kb_frame.pack(fill="x", pady=5)

        ttk.Label(kb_frame, text="Setzt die visuelle Symbol-Bibliothek zurück (Vortraining).").pack(anchor="w")
        ttk.Button(kb_frame, text="Gelernte Symbole (learned_symbols_images) löschen", command=self.controller.clear_learned_symbols).pack(pady=5, fill="x")

        # Frame für GUI-Management
        gui_frame = ttk.LabelFrame(self, text="Oberfläche", padding="10")
        gui_frame.pack(fill="x", pady=5)

        ttk.Label(gui_frame, text="Leert das Log-Fenster, ohne die Anwendung neu zu starten.").pack(anchor="w")
        ttk.Button(gui_frame, text="GUI-Log leeren", command=self.controller.clear_gui_log).pack(pady=5, fill="x")

if __name__ == "__main__":
    try:
        logging.info("DIAGNOSE: Starte Hauptanwendung...")
        app = IntegratedSuite()
        logging.info("DIAGNOSE: App-Objekt erstellt. Rufe mainloop() auf...")
        app.mainloop()
        logging.info("DIAGNOSE: mainloop() beendet.")
    except Exception as e:
        # Wenn irgendetwas beim Start schiefgeht, wird dieser Block ausgeführt
        logging.error("Ein fataler Fehler hat den Start der Anwendung verhindert.", exc_info=True)
        messagebox.showerror(
            "Fataler Startfehler",
            f"Die Anwendung konnte nicht gestartet werden.\n\nFehler: {e}\n\nSchau in die Log-Datei für Details."
        )