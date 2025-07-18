# knowledge_bases.py - FINALE, OPTIMIERTE & LOGISCH KONSISTENTE VERSION

# 📦 Standardbibliotheken
import datetime
import hashlib
import json
import logging
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
from typing import Dict, Any, List, Optional, Tuple, cast # Hinzugefügt Tuple
from pathlib import Path
from io import BytesIO
import base64
import uuid
import threading
from filelock import FileLock, Timeout

# 🖼️ Bildverarbeitung
from PIL import Image


logger = logging.getLogger(__name__)

class KnowledgeManager:
    """
    Verwaltet statisches und dynamisches Wissen. Diese Version integriert
    detailliertes Fehler-Tracking, das Lernen aus Korrekturen und ist für
    Skalierbarkeit bei der Vektorsuche optimiert.
    """

    def __init__(self, element_type_list_path: str, learning_db_path: str, llm_handler: Any, config: Dict[str, Any]):
        self.element_type_list_path = element_type_list_path
        self.learning_db_path = learning_db_path
        self.llm_handler = llm_handler
        self.config = config
        
        self.config_library: List[Dict[str, Any]] = []
        self.learning_database: Dict[str, Any] = {}
        
        self._type_name_to_id: Dict[str, str] = {}
        self._type_id_to_data: Dict[str, Dict] = {}
        
        # OPTIMIERUNG: In-Memory Vektor-Index für schnelle Suche
        self._symbol_vector_index: Optional[np.ndarray] = None 
        self._symbol_vector_data: List[Dict[str, Any]] = [] # Speichert die Symbol-Daten, nicht nur Lösungen
        self._symbol_vector_ids: List[str] = [] # Die IDs der Symbole im Index

        # Der alte Index für learned_solutions bleibt getrennt, jetzt basierend auf Text-Embeddings
        self._solution_vector_index: Optional[np.ndarray] = None
        self._solution_vector_keys: List[str] = [] # Problem-Beschreibungen (Text) als JSON-Strings der Embeddings
        self._solution_vector_solutions: List[Dict[str, Any]] = [] # Korrekturen (Lösungen)


        # NEU: Prozesssichere Sperre für Schreibzugriffe auf die Datenbank
        self.db_path = learning_db_path
        file_lock_timeout = self.config.get('logic_parameters', {}).get('db_file_lock_timeout', 15) 
        self.lock_path = self.db_path + ".lock"
        self.db_process_lock = FileLock(self.lock_path, timeout=file_lock_timeout)
        self.db_thread_lock = threading.Lock() # Thread-Sperre für In-Process-Zugriffe

        self._load_config_library()
        self._load_learning_database()

    def _load_config_library(self):
        """Lädt die Basis-Typenliste und baut schnelle Such-Indizes auf."""
        logging.info(f"Lade Basis-Wissensbasis von: {self.element_type_list_path}")
        try:
            with open(self.element_type_list_path, 'r', encoding='utf-8') as f:
                self.config_library = json.load(f)
            
            for element_type in self.config_library:
                type_id = element_type.get('id')
                type_name = element_type.get('name', '')
                if type_id and type_name:
                    # Normalisiere den Namen beim Speichern im Index
                    self._type_name_to_id[self._normalize_label(type_name)] = type_id
                    self._type_id_to_data[type_id] = element_type
            
            logging.info(f"Erfolgreich {len(self.config_library)} Basis-Typen verarbeitet.")
        except Exception as e:
            logging.error(f"FATAL: Konnte Basis-Wissensbasis nicht laden: {e}", exc_info=True)

    def _load_learning_database(self):
        """Lädt die Lern-Datenbank und baut den In-Memory Vektor-Index auf."""
        logging.info(f"Lade Lern-Datenbank von: {self.learning_db_path}")
        try:
            with open(self.learning_db_path, 'r', encoding='utf-8') as f:
                self.learning_database = json.load(f)
            logging.info("Lern-Datenbank erfolgreich geladen.")
        except FileNotFoundError:
            logging.warning(f"Lern-Datenbank nicht gefunden. Erstelle neue Datenbank.")
            self.learning_database = {
                "knowledge_extensions": {"type_aliases": {}},
                "successful_patterns": {},
                "error_stats": {},
                "learned_solutions": {},
                "symbol_library": {}, # Visuelle Symbolbibliothek
                "learned_visual_corrections": {} # Für visuelle Fehler-Muster
            }
        except Exception as e:
            logging.error(f"Fehler beim Laden der Lern-Datenbank: {e}", exc_info=True)
        
        # Sicherstellen, dass alle Top-Level-Keys existieren
        self.learning_database.setdefault("knowledge_extensions", {"type_aliases": {}})
        self.learning_database["knowledge_extensions"].setdefault("type_aliases", {})
        self.learning_database.setdefault("successful_patterns", {})
        self.learning_database.setdefault("error_stats", {})
        self.learning_database.setdefault("learned_solutions", {})
        self.learning_database.setdefault("symbol_library", {})
        self.learning_database.setdefault("learned_visual_corrections", {})
        
        self._build_vector_index() # Index nach dem Laden bauen

    def save_learning_database(self):
            """Speichert den aktuellen Stand der Lern-Datenbank prozesssicher und lesbar."""
            logging.info(f"Speichere Lern-Datenbank nach: {self.db_path}")
            try:
                with self.db_process_lock:
                    with self.db_thread_lock:
                        # Temporär in einen String schreiben, um Fehler abzufangen
                        # bevor die eigentliche Datei überschrieben wird.
                        temp_json_string = json.dumps(
                            self.learning_database, 
                            ensure_ascii=False, 
                            indent=4
                        )
                        with open(self.db_path, 'w', encoding='utf-8') as f:
                            f.write(temp_json_string)
                        logging.info("Lern-Datenbank erfolgreich gespeichert.")
            except Timeout:
                logging.error(f"Konnte die Sperre für {self.lock_path} nicht erhalten. Speichern übersprungen.")
            except Exception as e:
                logging.error(f"Fehler beim Speichern der Lern-Datenbank: {e}", exc_info=True)

    def _build_vector_index(self):
        """
        Erstellt aus den gespeicherten Vektoren schnelle In-Memory Indizes.
        Liest Vektoren nun aus dem korrekten Feld "problem_embedding".
        """
        logging.info("Baue In-Memory Vektor-Indizes auf...")
        
        # --- Index für gelernte Lösungen (Text-basierte Problem-Embeddings) ---
        learned_solutions = self.learning_database.get("learned_solutions", {})
        vectors_as_lists = []
        solution_keys_filtered = []
        solution_values_filtered = []

        if learned_solutions:
            for key, solution_data in learned_solutions.items():
                # KORREKTUR: Lese den Vektor aus dem 'problem_embedding'-Feld, nicht aus dem Key.
                embedding = solution_data.get("problem_embedding")
                if isinstance(embedding, list) and all(isinstance(x, (int, float)) for x in embedding):
                    vectors_as_lists.append(embedding)
                    solution_keys_filtered.append(key) # Der Key ist der Hash
                    solution_values_filtered.append(solution_data)
                else:
                    logging.warning(f"Kein gültiges Embedding im Lösungs-Eintrag für Schlüssel '{key[:50]}...' gefunden. Übersprungen.")
            
            if vectors_as_lists:
                self._solution_vector_index = np.array(vectors_as_lists, dtype=np.float32)
                self._solution_vector_keys = solution_keys_filtered
                self._solution_vector_solutions = solution_values_filtered
                logging.info(f"Lösungs-Vektor-Index für Text-Probleme mit {len(self._solution_vector_index)} Einträgen gebaut.")
            else:
                self._solution_vector_index = None
                self._solution_vector_keys = []
                self._solution_vector_solutions = []
                logging.warning("Keine gültigen Text-Embeddings für Lösungs-Index gefunden.")
        else:
            self._solution_vector_index = None
            self._solution_vector_keys = []
            self._solution_vector_solutions = []
            logging.info("Keine gelernten Lösungen vorhanden, um Lösungs-Index zu bauen.")

        # --- Index für Symbol-Bibliothek (Visuelle Bild-Embeddings) ---
        symbol_library = self.learning_database.get("symbol_library", {})
        if symbol_library:
            valid_symbols_data = [data for data in symbol_library.values() if isinstance(data, dict) and isinstance(data.get("visual_embedding"), list)]
            if valid_symbols_data:
                self._symbol_vector_ids = [sym_id for sym_id, data in symbol_library.items() if isinstance(data, dict) and isinstance(data.get("visual_embedding"), list)]
                self._symbol_vector_data = valid_symbols_data
                vectors_as_lists_symbols = [data["visual_embedding"] for data in valid_symbols_data]
                self._symbol_vector_index = np.array(vectors_as_lists_symbols, dtype=np.float32)
                logging.info(f"Symbol-Vektor-Index für visuelle Symbole mit {len(self._symbol_vector_index)} Einträgen gebaut.")
            else:
                self._symbol_vector_index = None
                self._symbol_vector_ids = []
                self._symbol_vector_data = []
                logging.warning("Keine gültigen visuellen Embeddings in der Symbol-Bibliothek gefunden, um Index zu bauen.")
        else:
            self._symbol_vector_index = None
            self._symbol_vector_ids = []
            self._symbol_vector_data = []
            logging.info("Keine Symbole in Bibliothek vorhanden, um Symbol-Index zu bauen.")

    def cleanup_learning_database(self):
            """
            Führt eine automatische Bereinigung der Lern-Datenbank durch.
            Diese Funktion bereinigt NUR alte, qualitativ schlechte
            'successful_patterns' und prüft auf Alias-Konflikte.
            Die 'symbol_library' und 'learned_visual_corrections' werden hierbei NICHT verändert oder gelöscht,
            da sie als dauerhafte Wissenspeicher dienen.
            """
            logging.info("Starte automatische Bereinigung der Lern-Datenbank...")
            was_modified = False
            
            # 1. Bereinige qualitativ schlechte "successful_patterns" (für Few-Shot Learning)
            minimum_score_for_few_shot_pattern = self.config.get('logic_parameters', {}).get('min_score_for_few_shot_pattern', 85)
            
            patterns = self.learning_database.get("successful_patterns", {})
            if patterns:
                # Erstelle eine Kopie der Keys, da wir das Dictionary während der Iteration verändern
                patterns_to_check = list(patterns.keys())
                for key in patterns_to_check:
                    pattern_data = patterns.get(key)
                    # Stelle sicher, dass pattern_data ein Dict ist und relevante Schlüssel hat
                    if isinstance(pattern_data, dict) and "final_ai_data" in pattern_data:
                        final_data = pattern_data["final_ai_data"]
                        score = final_data.get("quality_score", 0) if isinstance(final_data, dict) else 0

                        if score < minimum_score_for_few_shot_pattern:
                            del self.learning_database["successful_patterns"][key]
                            logging.warning(f"Entferne qualitativ schlechtes Lernbeispiel '{key}' (Score: {score}) aus der Datenbank.")
                            was_modified = True
                    else:
                        logging.warning(f"Ungültiges Muster-Datenformat für Schlüssel '{key}' in 'successful_patterns'. Wird entfernt.")
                        del self.learning_database["successful_patterns"][key]
                        was_modified = True

            # 2. Finde und melde potenzielle Konflikte in den Aliasen (löscht nicht automatisch)
            aliases = self.get_all_aliases()
            alias_to_canonical_map: Dict[str, List[str]] = {}
            for canonical, alias_list in aliases.items():
                for alias in alias_list:
                    alias_lower = alias.lower()
                    if alias_lower in alias_to_canonical_map:
                        alias_to_canonical_map[alias_lower].append(canonical)
                    else:
                        alias_to_canonical_map[alias_lower] = [canonical]

            for alias, canonicals in alias_to_canonical_map.items():
                if len(canonicals) > 1:
                    logging.warning(f"Alias-Konflikt entdeckt: Der Alias '{alias}' ist für mehrere kanonische Typen registriert: {canonicals}. Bitte manuell in learning_db.json prüfen.")
                
                # Prüfe, ob ein Alias selbst ein kanonischer Typ ist
                if self.find_element_type_by_name(alias): # Nutzt find_element_type_by_name, um Kanonizität zu prüfen
                    logging.warning(f"Alias-Konflikt entdeckt: Der Alias '{alias}' ist auch ein offizieller kanonischer Typ. Dies kann zu Fehlern führen. Bitte manuell prüfen.")

            if was_modified:
                logging.info("Lern-Datenbank wurde bereinigt. Speichere Änderungen.")
                self.save_learning_database()
            else:
                logging.info("Keine Bereinigung der Lern-Datenbank notwendig.")
    
    def merge_old_databases(self, old_learning_path: Path):
        """Sucht nach alten learning_db.json Dateien und mergt deren Wissen."""
        logging.info(f"Suche nach alten Lern-Datenbanken in: {old_learning_path}")
        if not old_learning_path.exists():
            return

        for old_db_path in old_learning_path.glob("*.json"):
            try:
                with open(old_db_path, 'r', encoding='utf-8') as f:
                    old_db = json.load(f)
                logging.info(f"Verarbeite alte Datenbank: '{old_db_path.name}'...")
                
                # Mische alte Aliase mit den aktuellen
                old_aliases = old_db.get("knowledge_extensions", {}).get("type_aliases", {})
                if old_aliases:
                    current_aliases = self.learning_database.get("knowledge_extensions", {}).get("type_aliases", {})
                    for canonical, alias_list in old_aliases.items():
                        # Sicherstellen, dass canonical_name in _type_name_to_id gefunden wird
                        valid_canonical_id = self._type_name_to_id.get(self._normalize_label(canonical))
                        if valid_canonical_id:
                            if canonical not in current_aliases:
                                current_aliases[canonical] = []
                            for alias in alias_list:
                                if alias not in current_aliases[canonical]: # Vermeide Duplikate
                                    current_aliases[canonical].append(alias)
                    self.learning_database["knowledge_extensions"]["type_aliases"] = current_aliases # Aktualisiere im Haupt-DB
                    logging.info(f"-> Aliase aus '{old_db_path.name}' erfolgreich gemischt.")

                # Optional: Merge successful_patterns, error_stats, learned_solutions, symbol_library
                # Diese Merges können komplexer sein, um Duplikate zu handhaben und nur die besten/neuesten zu behalten
                # Für jetzt nur Aliase gemerged, da das am unkompliziertesten ist.

            except Exception as e:
                logging.error(f"Fehler beim Mischen der alten Datenbank '{old_db_path.name}': {e}")
        
        self.save_learning_database()
        self._build_vector_index() # Rebuild Index nach dem Mergen

    def integrate_symbol_library(self, symbols_to_process: List[Tuple[str, Image.Image, str]]) -> List[Dict]:
        """
        Verarbeitet extrahierte Symbole, prüft auf Duplikate via Vektorsuche,
        speichert Bilder separat und referenziert nur ihren Pfad.
        """
        report_data = []
        logging.info(f"Beginne Integration von {len(symbols_to_process)} neuen Symbolkandidaten...")
        
        symbol_library = self.learning_database.setdefault("symbol_library", {})
        
        # Bestimme den Basispfad, relativ zu dem die Bildpfade gespeichert werden.
        base_knowledge_dir = Path(self.learning_db_path).parent
        learned_symbol_images_dir = base_knowledge_dir / "learned_symbols_images"
        learned_symbol_images_dir.mkdir(parents=True, exist_ok=True)
        logging.debug(f"Symbolbilder werden gespeichert in: {learned_symbol_images_dir}")

        for label, cropped_img, source_name in symbols_to_process:
            # Hole das visuelle Embedding für das neue Symbol
            new_visual_embedding = self.llm_handler.get_image_embedding(cropped_img)

            if not new_visual_embedding:
                logging.warning(f"Konnte kein visuelles Embedding für Symbol '{label}' erstellen. Überspringe.")
                continue
            
            new_visual_embedding_np = np.array([new_visual_embedding], dtype=np.float32)

            status = "neu"
            similar_to = None
            target_id = f"sym_{uuid.uuid4().hex[:12]}" # Standardmäßig eine neue ID

            # Führe einen visuellen Duplikat-Check durch
            visual_threshold = self.config.get('logic_parameters', {}).get('visual_symbol_similarity_threshold', 0.85)
            if self._symbol_vector_index is not None and self._symbol_vector_index.size > 0:
                similarities = cosine_similarity(new_visual_embedding_np, self._symbol_vector_index)[0]
                best_match_idx = np.argmax(similarities)
                
                if similarities[best_match_idx] >= visual_threshold:
                    # Ein Duplikat wurde gefunden. Wir verwenden dessen ID.
                    target_id = self._symbol_vector_ids[best_match_idx]
                    status = "aktualisiert (visuell)"
                    
                    matched_symbol_data = self.learning_database['symbol_library'].get(target_id, {})
                    matched_symbol_name = matched_symbol_data.get('name', 'N/A')
                    similar_to = f"{matched_symbol_name} (ID: {target_id[:6]}...), Visuelle Ähnlichkeit: {similarities[best_match_idx]:.2f}"
                    logging.info(f"Symbol '{label}' als Duplikat von '{matched_symbol_name}' erkannt.")
            
            # Speichere das Bild und hole den relativen Pfad
            symbol_image_filename = f"{target_id}.png"
            symbol_image_path_full = learned_symbol_images_dir / symbol_image_filename
            symbol_image_relative_path = None
            try:
                cropped_img.save(symbol_image_path_full)
                symbol_image_relative_path = str(symbol_image_path_full.relative_to(base_knowledge_dir))
            except Exception as e:
                logging.error(f"Konnte Symbolbild {symbol_image_path_full} nicht speichern: {e}", exc_info=True)

            # Füge die neue oder aktualisierte Lernkarte hinzu/überschreibe sie.
            symbol_library[target_id] = {
                "name": label,
                "description": f"Standard P&ID Symbol für '{label}'.",
                "source": source_name,
                "visual_embedding": new_visual_embedding, # Das visuelle Embedding speichern
                "image_path": symbol_image_relative_path, # Nur der Pfad wird gespeichert
                "timestamp": datetime.datetime.now().isoformat()
            }
            
            report_data.append({"symbol_id": target_id, "name": label, "source": source_name, "status": status, "similar_to": similar_to})
            
        self.save_learning_database()
        self._build_vector_index() # Index nach Hinzufügen neuer Symbole neu aufbauen
        logging.info(f"-> {len(report_data)} Symbole verarbeitet und in die Wissensbasis integriert.")
        return report_data


    def add_learning_pattern(self, analysis_result: Dict[str, Any], truth_data: Dict[str, Any], error_matrix: Dict[str, Any]):
        """
        Konsistente Lernfunktion. Lernt aus Erfolgen (für Few-Shot)
        UND aus Fehlern (wenn 'truth_data' als Korrektur bereitsteht).
        """
        logging.info("Aktualisiere Lern-Datenbank basierend auf dem letzten Lauf...")
        image_name = analysis_result.get("metadata", {}).get("image_name", "unknown")
        was_modified = False

        # 1. Aus Fehlern lernen, wenn eine Korrektur (truth_data) vorhanden ist
        if truth_data:
            kpis = analysis_result.get('final_kpis', {})
            error_details = kpis.get('error_details', {})
            
            # Lerne nur, wenn es tatsächlich Fehler gab
            if error_details and any(error_details.values()):
                logging.info(f"Fehler für Bild '{image_name}' entdeckt. Lerne aus Korrektur.")
                
                self._update_error_statistics(error_details)
                
                problem_description = f"Analysefehler für Bild '{image_name}': {json.dumps(error_details)}"
                correction_data = {"corrected_data": truth_data}
                
                problem_to_learn = {
                    "type": "CORRECTION_EVENT",
                    "description": problem_description,
                }
                # Rufe die korrigierte learn_from_correction auf, die mit Hashes arbeitet
                self.learn_from_correction(problem_to_learn, correction_data)
                was_modified = True

        # 2. Aus allgemeinen Erfolgen lernen (für Few-Shot-Beispiele)
        score = analysis_result.get('quality_score', 0)
        min_score = self.config.get('logic_parameters', {}).get('min_score_for_few_shot_pattern', 85)
        if score >= min_score:
            logging.info(f"Hoher Quality Score ({score}). Speichere als erfolgreiches Muster.")
            
            self.learning_database["successful_patterns"][image_name] = {
                "timestamp": datetime.datetime.now().isoformat(),
                "quality_score": score,
                "final_ai_data": analysis_result
            }
            was_modified = True
        
        if was_modified:
            self.save_learning_database()

    def _update_error_statistics(self, error_details: Dict):
        """
        Verarbeitet strukturierte Fehlerobjekte und aktualisiert Statistiken.
        Dies speichert jetzt auch visuelle Fehler, wenn möglich, und ist für TYPING_ERRORs und HALLUCINATED_ELEMENTs ausgelegt.
        """
        stats = self.learning_database.get("error_stats", {})

        # Verarbeite Typisierungsfehler
        for error in error_details.get("typing_errors", []):
            ai_type = error.get("ai_type")
            truth_type = error.get("truth_type")
            element_id = error.get("element_id") 
            
            if not all([ai_type, truth_type, element_id]): 
                logging.warning(f"Unvollständige Typisierungsfehler-Details: {error}. Überspringe.")
                continue
            
            # Lerne Alias, wenn der Wahrheitstyp bekannt ist
            if self.find_element_type_by_name(truth_type):
                if self.find_element_type_by_name(ai_type) is None: # Nur lernen, wenn ai_type selbst kein kanonischer Typ ist
                    self.add_type_alias(truth_type, ai_type)
                else:
                    logging.debug(f"Lerne Alias '{ai_type}' für '{truth_type}' nicht, da '{ai_type}' bereits ein bekannter kanonischer Typ ist.")
            
            # Statistiken aktualisieren (z.B. TYPING_ERROR:Valve->Sensor)
            key = f"TYPING_ERROR:{truth_type}->{ai_type}"
            stats[key] = stats.get(key, 0) + 1

        # Verarbeite halluzinierte Elemente (von der KI erfundene Elemente)
        for element_id in error_details.get("hallucinated_elements", []):
            key = f"HALLUCINATED_ELEMENT:{element_id}" # Speichern der ID des halluzinierten Elements
            stats[key] = stats.get(key, 0) + 1

        # Hier können weitere Fehlertypen hinzugefügt werden (z.B. missed_elements)
        # For example:
        # for element_id in error_details.get("missed_elements", []):
        #     key = f"MISSED_ELEMENT:{element_id}"
        #     stats[key] = stats.get(key, 0) + 1

        self.learning_database["error_stats"] = stats
        logging.info("Fehler-Statistiken aktualisiert.")

    def _get_problem_hash(self, problem_description: str) -> str:
        """Erzeugt einen eindeutigen, kurzen Hash aus einer Problembeschreibung."""
        return hashlib.sha256(problem_description.encode('utf-8')).hexdigest()

    def find_solution_for_problem(self, problem: Dict) -> Optional[Dict]:
            """Sucht nach Lösungen, indem es den Hash der Problembeschreibung vergleicht."""
            logging.info("Suche in der Wissensbasis nach einer Lösung für das Problem (Hash-basiert)...")
            
            problem_description = problem.get("description")
            if not problem_description:
                logging.warning("Keine Problembeschreibung für die Suche vorhanden.")
                return None

            problem_hash = self._get_problem_hash(problem_description)
            
            learned_solutions = self.learning_database.get("learned_solutions", {})
            
            if problem_hash in learned_solutions:
                found_solution_data = learned_solutions[problem_hash]
                found_correction = found_solution_data.get("correction_applied")
                logging.info(f"GEFUNDEN: Exaktes Problem (Hash: {problem_hash[:8]}...) in der DB. Schlage gelernte Lösung vor.")
                return cast(Dict, found_correction)

            logging.info("Kein exakt passendes, gelöstes Problem in der Wissensbasis gefunden.")
            return None

    def learn_from_correction(self, problem: Dict, correction: Dict):
        """
        Speichert ein Problem-Korrektur-Paar unter Verwendung eines kurzen Hashes als Schlüssel.
        """
        problem_description = problem.get("description")
        if not problem_description or not correction:
            logging.warning("Keine vollständige Problem- oder Korrekturinformation zum Lernen verfügbar.")
            return

        problem_hash = self._get_problem_hash(problem_description)
        
        # Lerne nur, wenn dieser exakte Fehler noch nicht bekannt ist.
        if problem_hash not in self.learning_database["learned_solutions"]:
            # Das Embedding wird jetzt im Eintrag gespeichert, nicht mehr als Schlüssel.
            problem_vector = self.llm_handler.get_text_embedding(problem_description)
            
            self.learning_database["learned_solutions"][problem_hash] = {
                "problem_type": problem.get("type"),
                "problem_description": problem_description,
                "problem_embedding": problem_vector, # Vektor sauber im JSON speichern
                "correction_applied": correction,
                "timestamp": datetime.datetime.now().isoformat()
            }
            logging.info(f"GELERNT: Neue Lösung für Problem (Hash: {problem_hash[:8]}...) gespeichert.")
            self.save_learning_database()
        else:
            logging.info("Problem/Lösung-Kombination bereits bekannt. Kein erneutes Lernen notwendig.")

    def add_type_alias(self, canonical_name: str, new_alias: str):
        """Fügt einen neuen Alias zu einem bekannten Typ hinzu."""
        aliases = self.learning_database.get("knowledge_extensions", {}).get("type_aliases", {})
        
        if canonical_name not in aliases:
            aliases[canonical_name] = []
        
        # Vermeide Duplikate und Normalisiere für den Vergleich
        if self._normalize_label(new_alias) not in [self._normalize_label(a) for a in aliases[canonical_name]]:
            aliases[canonical_name].append(new_alias)
            logging.info(f"GELERNT: Neuer Alias '{new_alias}' für den Typ '{canonical_name}' hinzugefügt.")
            self.learning_database["knowledge_extensions"]["type_aliases"] = aliases
            self.save_learning_database()  # Speichert nach Alias-Lernen sofort die Datenbank  

    def find_parent_type_by_heuristic(self, unknown_type_name: str) -> Optional[Dict]:
        """
        Versucht, einen unbekannten Typen heuristisch einem bekannten übergeordneten
        Typ zuzuordnen, indem es prüft, ob ein kanonischer Name im unbekannten Namen enthalten ist.
        """
        if not unknown_type_name:
            return None
            
        normalized_unknown_name = self._normalize_label(unknown_type_name)
        best_match = None
        max_len = 0

        for type_data in self.config_library: # Iteriere über die geladene Konfigurationsbibliothek
            canonical_name = type_data.get('name')
            if canonical_name:
                normalized_canonical_name = self._normalize_label(canonical_name)
                # Prüfe, ob der kanonische Name (oder dessen Teil) im unbekannten Namen enthalten ist
                if normalized_canonical_name in normalized_unknown_name:
                    if len(normalized_canonical_name) > max_len: # Finde den längsten passenden Namen
                        max_len = len(normalized_canonical_name)
                        best_match = type_data
        
        if best_match:
            logging.info(f"HEURISTIK: Unbekannter Typ '{unknown_type_name}' wurde dem bekannten übergeordneten Typ '{best_match['name']}' zugeordnet.")
            return best_match
            
        return None      

    def get_known_types(self) -> List[str]:
        """Gibt eine Liste aller kanonischen Namen der Basis-Typen zurück."""
        # Sicherstellen, dass _type_id_to_data gefüllt ist
        if not self._type_id_to_data:
            self._load_config_library() # Versuche, die Bibliothek zu laden, falls leer
            if not self._type_id_to_data:
                logging.warning("Keine bekannten Typen in der Konfigurationsbibliothek gefunden.")
                return []

        return [type_data.get('name', '') for type_data in self._type_id_to_data.values() if type_data.get('name')]

    def get_all_aliases(self) -> Dict[str, List[str]]:
        """Gibt alle bekannten Aliase aus der Lern-Datenbank zurück."""
        return self.learning_database.get("knowledge_extensions", {}).get("type_aliases", {})
    
    def get_few_shot_examples(self, image_name: str, k: int = 1) -> List[Dict[str, Any]]:
        """
        Holt die besten letzten erfolgreichen Analysen als Lernbeispiele.
        Diese Version filtert das aktuell analysierte Bild aus den Beispielen heraus
        und gibt nur die relevanten 'elements' und 'connections' zurück.
        """
        logging.info(f"Rufe {k} 'Few-Shot'-Beispiel(e) ab.")

        successful_patterns = self.learning_database.get("successful_patterns", {})
        if not successful_patterns:
            return []

        # Filtere Muster, um das aktuelle Bild auszuschließen
        other_patterns = [v for key, v in successful_patterns.items() if key != image_name]

        # Robustere Sortierung mit Absicherung
        filtered_patterns = [
            x for x in other_patterns 
            if isinstance(x.get('timestamp'), str)
        ]

        # Sortiere die verbleibenden Muster nach Zeitstempel, um die neuesten zu erhalten
        sorted_patterns = sorted(
            filtered_patterns,
            key=lambda x: x.get('timestamp'),
            reverse=True
        )

        # Extrahiere nur die 'elements' und 'connections' aus final_ai_data
        lean_examples = []
        # Nutze min_score_for_few_shot_pattern aus der Konfiguration
        min_score_for_few_shot_pattern = self.config.get('logic_parameters', {}).get('min_score_for_few_shot_pattern', 85)

        for pattern in sorted_patterns: # Iteriere über alle sortierten Muster
            if pattern.get('quality_score', 0) >= min_score_for_few_shot_pattern: # Nur Muster mit ausreichendem Score
                final_ai_data = pattern.get('final_ai_data')
                if final_ai_data and isinstance(final_ai_data, dict):
                    lean_examples.append({
                        "elements": final_ai_data.get('elements', []),
                        "connections": final_ai_data.get('connections', [])
                    })
            if len(lean_examples) >= k: # Nur so viele Beispiele nehmen wie angefordert
                break

        return lean_examples

    # DIES IST DIE LETZTE ZEILE DES KNOWLEDGE_MANAGER KLASSEN-CODE-BLOCKS.
    # NÄCHSTER CODE GEHÖRT WARSCHEINLICH NICHT MEHR ZUR KLASSE.
    # Der Rest deines knowledge_bases.py-Codes muss entsprechend außerhalb
    # der Klasse liegen oder als neue Methode innerhalb der Klasse eingerückt sein.
    # Prüfe hier die Einrückung des restlichen Codes nach get_few_shot_examples!    

    def find_element_type_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Sucht einen Elementtyp zuerst nach kanonischem Namen, dann nach Aliaseinträgen.
        Verwendet normalisierte Namen für den Vergleich.
        """
        if not isinstance(name, str):
            logging.warning(f"WARNUNG: 'find_element_type_by_name' wurde mit ungültigem Typ aufgerufen: {type(name)} – Wert: {str(name)[:50]}...")
            return None

        normalized_name = self._normalize_label(name) 

        # 1. Zuerst nach kanonischem Namen suchen (normalisierter Schlüssel)
        type_id = self._type_name_to_id.get(normalized_name)
        if type_id:
            return self._type_id_to_data.get(type_id)

        # 2. Dann nach Alias suchen
        aliases = self.get_all_aliases()
        for canonical_name, alias_list in aliases.items():
            # Normalisiere den kanonischen Namen für den Vergleich mit _type_name_to_id
            normalized_canonical_for_lookup = self._normalize_label(canonical_name)
            
            # Prüfe, ob der normalisierte Eingabename in der normalisierten Aliasliste ist
            if normalized_name in [self._normalize_label(a) for a in alias_list]:
                canonical_id = self._type_name_to_id.get(normalized_canonical_for_lookup) 
                if canonical_id:
                    return self._type_id_to_data.get(canonical_id)
        return None

    @staticmethod
    def _normalize_label(label: Optional[str]) -> str:
        """
        Normalisiert einen Label-String (Kleinschreibung, Leerzeichen/Sonderzeichen entfernen, trimmen).
        """
        if label is None: return ""
        return str(label).lower().replace(" ", "").replace("-", "").replace("_", "").strip()

    def find_symbol_type_by_visual_similarity(self, image_path: str) -> Optional[str]:
        """
        Sucht in der Symbolbibliothek nach dem Typ des visuell ähnlichsten Symbols.
        Nutzt den In-Memory Vektor-Index, der visuelle Embeddings enthält.
        """
        logging.info(f"Suche visuell ähnlichen Symboltyp für Bild: {image_path}")

        # Prüfe, ob der Vektor-Index für Symbole überhaupt existiert und Daten enthält
        if self._symbol_vector_index is None or self._symbol_vector_index.size == 0:
            logging.warning("Symbol-Vektor-Index ist leer. Kein visueller Abgleich möglich. Hast du ein Vortraining durchgeführt?")
            return None

        # 1. Embedding für das gegebene Bild erstellen (nutzt LLM_Handler Workaround)
        query_embedding = self.llm_handler.get_image_embedding(image_path)
        if query_embedding is None:
            logging.error(f"Konnte kein Embedding für das Abfragebild {image_path} erstellen. Visueller Abgleich nicht möglich.")
            return None

        # Wandle das Abfrage-Embedding in ein Numpy-Array um, bereit für den Vergleich
        query_embedding_np = np.array([query_embedding], dtype=np.float32)

        # 2. Kosinus-Ähnlichkeit mit allen gespeicherten Symbol-Embeddings berechnen
        try:
            # cosine_similarity erwartet 2D-Arrays: (n_samples, n_features)
            similarities = cosine_similarity(query_embedding_np, self._symbol_vector_index)[0]
        except ValueError as e:
            logging.error(f"Fehler bei der Berechnung der Kosinus-Ähnlichkeit für visuellen Abgleich. Formen der Vektoren inkompatibel: {e}")
            logging.debug(f"Query embedding shape: {query_embedding_np.shape}, Index shape: {self._symbol_vector_index.shape}")
            return None
        except Exception as e:
            logging.error(f"Unerwarteter Fehler bei Kosinus-Ähnlichkeit für visuellen Abgleich: {e}", exc_info=True)
            return None


        # 3. Besten Treffer finden (den Index des höchsten Ähnlichkeitswerts)
        best_match_idx = np.argmax(similarities)
        highest_similarity = similarities[best_match_idx]

        # 4. Schwellenwert prüfen (aus Konfiguration)
        threshold = self.config.get('logic_parameters', {}).get('min_visual_match_score', 0.70)
        
        if highest_similarity >= threshold:
            # Den originalen Symboltyp ('name') des am besten passenden Symbols aus unserer Datenliste zurückgeben
            matched_symbol_data = self._symbol_vector_data[best_match_idx]
            matched_type = matched_symbol_data.get('name') 
            logging.info(f"Visueller Treffer gefunden: '{matched_type}' (Ähnlichkeit: {highest_similarity:.2f} >= Schwelle: {threshold:.2f}) für '{image_path}'.")
            return matched_type
        else:
            logging.info(f"Kein ausreichend ähnlicher visueller Treffer gefunden (beste Ähnlichkeit: {highest_similarity:.2f} < Schwelle: {threshold:.2f}).")
            return None