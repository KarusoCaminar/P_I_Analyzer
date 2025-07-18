# file: core_processor.py
# core_processor.py - FINALE, BEREINIGTE & KORRIGIERTE VERSION

# 📦 Standardbibliotheken
import logging
import os
import json
import time
import uuid
import shutil
from pathlib import Path
from typing import Optional, Callable, Any, Dict, List, Tuple, Union, cast, Set 
import concurrent.futures 
from concurrent.futures import ThreadPoolExecutor, as_completed

# 🧪 Drittanbieter-Bibliotheken
from PIL import Image
Image.MAX_IMAGE_PIXELS = None 
import networkx as nx 

# 🌐 Projektinterne Module
import utils
from utils import GraphSynthesizer, BBox, Element
from llm_handler import LLM_Handler
from knowledge_bases import KnowledgeManager
from utils import Connection, Element, BBox, TileResult, SynthesizerConfig, _normalize_bbox_from_pixels # Korrektes Importieren der TypedDicts/Dataclasses und der Hilfsfunktion

# Configure module-level logger
logger = logging.getLogger(__name__)

# NEU: Definition des abstrakten Callbacks
class ProgressCallback:
    def update_progress(self, value: int, message: str):
        """Aktualisiert den Fortschrittsbalken und Status."""
        raise NotImplementedError

    def update_status_label(self, text: str):
        """Aktualisiert nur den Status-Text."""
        raise NotImplementedError

class Core_Processor:
    def __init__(self, llm_handler: LLM_Handler, knowledge_manager: KnowledgeManager, model_strategy: Dict[str, Any], config: Dict[str, Any]):
        self.llm_handler = llm_handler
        self.knowledge_manager = knowledge_manager
        self.model_strategy = model_strategy
        self.config = config
        self._reset_state()
        self.synthesizer: Optional[utils.GraphSynthesizer] = None 

        self.progress_callback: Optional[ProgressCallback] = None
        self.start_time = 0

    def _reset_state(self):
        """Setzt den internen Zustand für einen neuen Analyse-Lauf zurück."""
        self._global_knowledge_repo: Dict[str, Any] = {}
        self._excluded_zones: List[Dict[str, float]] = []
        self._analysis_results: Dict[str, Any] = {
            "metadata": {}, "elements": [], "connections": []
        }
        self.current_image_path: Optional[str] = None
        self.active_logic_parameters: Dict[str, Any] = {}

    def _update_status_and_progress_internal(self, progress: int, message: str):
        """
        Interne Methode zur Aktualisierung von Fortschritt und Status über das Callback.
        """
        if self.progress_callback:
            elapsed_time = time.time() - self.start_time
            
            # Vermeide Division durch Null und Inf-Werte
            total_time_estimate = (elapsed_time / progress) * 100.0 if progress > 0 else 0
            
            time_str = f"Verstrichen: {elapsed_time:.1f}s"
            if 0 < total_time_estimate < float('inf'):
                time_str += f", Geschätzt Gesamt: {total_time_estimate:.1f}s"
            
            full_message = f"{message} ({progress}%) {time_str}"
            self.progress_callback.update_status_label(text=full_message)
            self.progress_callback.update_progress(value=progress, message=full_message) 
                                                                                        
    def run_pretraining(self, pretraining_path: Path, model_info: Dict[str, Any], progress_callback: Optional[ProgressCallback] = None) -> List[Dict]:
        """Orchestriert den gesamten Pre-Training-Workflow."""
        logging.info("===== Starte Symbol-Vortrainings-Pipeline =====")
        self.progress_callback = progress_callback
        if self.progress_callback: self.progress_callback.update_status_label(text="Starte Vortraining...")

        self.knowledge_manager.merge_old_databases(pretraining_path / "old_learning")

        all_image_paths = list(pretraining_path.glob("*.[pP][nN][gG]"))
        all_image_paths.extend(list(pretraining_path.glob("*.[jJ][pP]*[gG]")))
        
        if not all_image_paths:
            logging.warning("Keine Bilder im Vortrainings-Ordner gefunden.")
            if self.progress_callback: self.progress_callback.update_status_label(text="Keine Bilder gefunden.")
            return []

        logging.info(f"Starte Symbol-Extraktion für {len(all_image_paths)} Bilder...")
        all_extracted_symbols: List[Tuple[str, Image.Image, str]] = []
        
        max_workers_pretraining = self.config.get('logic_parameters', {}).get('pretraining_executor_max_workers', 5)
        
        # Sicherstellen, dass das Modell für die Symbolerkennung verfügbar ist
        symbol_detection_model_info = model_info # KORREKTUR: Nutze das direkt übergebene Modell
        if not symbol_detection_model_info:
            logging.error("Kein 'detail_model' in der Strategie für Symbol-Vortraining definiert. Symbol-Extraktion wird übersprungen.")
            if self.progress_callback: self.progress_callback.update_status_label(text="Fehler: Modell fehlt.")
            return [] # Nichts zum Extrahieren

        with ThreadPoolExecutor(max_workers=max_workers_pretraining) as executor:
            future_to_image = {
                executor.submit(self._extract_symbols_from_image, img_path, symbol_detection_model_info): img_path
                for img_path in all_image_paths
            }
            for i, future in enumerate(as_completed(future_to_image)):
                image_path = future_to_image[future]
                if self.progress_callback: self.progress_callback.update_status_label(text=f"Extrahiere Symbole... ({i+1}/{len(all_image_paths)})")
                try:
                    symbols = future.result()
                    all_extracted_symbols.extend([(label, img, image_path.name) for label, img in symbols])
                except Exception as exc:
                    logging.error(f"'{image_path.name}' hat eine Ausnahme während der Extraktion erzeugt: {exc}")
        
        logging.info(f"Extraktion abgeschlossen. {len(all_extracted_symbols)} Symbole gefunden. Starte Integration...")
        if self.progress_callback: self.progress_callback.update_status_label(text="Integriere Symbole in Wissensbasis...")
        report = self.knowledge_manager.integrate_symbol_library(all_extracted_symbols)
        
        logging.info("===== Symbol-Vortrainings-Pipeline erfolgreich abgeschlossen =====")
        if self.progress_callback: self.progress_callback.update_status_label(text="Vortraining abgeschlossen.")
        return report
    

    def _process_image_with_ai(self, image_path: Path, model_info: Dict[str, Any]) -> List[Tuple[str, Image.Image, BBox]]: # <- Ändere den Rückgabetyp
            """
            Nimmt EIN Bild (oder eine Kachel), schickt es an die KI mit dem symbol_detection_user_prompt,
            und gibt die ausgeschnittenen Symbole mit ihren Labels zurück.
            DIESE VERSION IST ROBUSTER und versteht alternative Schlüsselnamen vom LLM.
            """
            system_prompt_symbol_detection = self.config.get('prompts', {}).get('general_system_prompt', 'You are a P&ID analysis expert.')
            user_prompt_symbol_detection = self.config.get('prompts', {}).get('symbol_detection_user_prompt', 'Identify symbols and their bboxes.')
            
            response = self.llm_handler.call_llm(
                model_info, 
                system_prompt_symbol_detection, 
                user_prompt_symbol_detection, 
                str(image_path), 
                use_cache=True, 
                expected_json_keys=["symbols"] 
            )
            
            extracted_symbols: List[Tuple[str, Image.Image, BBox]] = [] # <- Ändere den Typ hier
            symbol_data_list: List[Dict[str, Any]] = []

            if response is None:
                logging.warning(f"LLM response for symbol detection from '{image_path.name}' was None.")
                return extracted_symbols

            # Flexible Extraktion der Symbol-Liste
            if isinstance(response, dict) and "symbols" in response and isinstance(response["symbols"], list):
                symbol_data_list = response["symbols"]
            elif isinstance(response, list):
                symbol_data_list = response
            else:
                logging.warning(f"Unexpected structure for symbol detection from '{image_path.name}'. Response: {response}. Cannot extract symbols.")
                return extracted_symbols 

            # Weiterverarbeitung mit flexiblen Schlüsseln
            if not symbol_data_list:
                return extracted_symbols

            try:
                with Image.open(image_path) as img:
                    img_width, img_height = img.size
                    
                    for symbol_data_item in symbol_data_list:
                        if not isinstance(symbol_data_item, dict):
                            logging.warning(f"Symbol-Datenobjekt ist kein Dictionary: {symbol_data_item}. Skipping.")
                            continue

                        label_candidates = []
                        bbox_data = None

                        # Helper function to find label/bbox in a nested dictionary (more robust)
                        def find_label_bbox_recursive(data: Any) -> Tuple[List[str], Optional[Any]]:
                            found_labels = []
                            found_bbox = None
                            if not isinstance(data, dict):
                                return found_labels, found_bbox
                            
                            # Prioritize direct, common keys
                            direct_label = data.get("name") or data.get("label") or data.get("text") or data.get("symbol_name") or data.get("symbol_description") or data.get("description") or data.get("value") or data.get("content") or data.get("symbol_label")
                            direct_bbox = data.get("box_2d") or data.get("bbox") or data.get("bbox_2d") or data.get("symbol_bbox") or data.get("coordinates") or data.get("box")
                            
                            if direct_label:
                                found_labels.append(str(direct_label))
                            if direct_bbox:
                                found_bbox = direct_bbox

                            # Check for label in 'value' or 'content' if not found yet
                            if not found_labels:
                                if data.get("value"): found_labels.append(str(data["value"]))
                                if data.get("content"): found_labels.append(str(data["content"]))
                                if data.get("symbol_label"): found_labels.append(str(data["symbol_label"]))

                            # Iterate through values for nested dictionaries or lists of dicts
                            for val in data.values():
                                if isinstance(val, dict):
                                    nested_labels, nested_bbox = find_label_bbox_recursive(val)
                                    found_labels.extend(nested_labels)
                                    if not found_bbox:
                                        found_bbox = nested_bbox
                                elif isinstance(val, list):
                                    for item in val:
                                        if isinstance(item, dict):
                                            nested_labels, nested_bbox = find_label_bbox_recursive(item)
                                            found_labels.extend(nested_labels)
                                            if not found_bbox:
                                                found_bbox = nested_bbox
                            return found_labels, found_bbox

                        # Attempt to find label and bbox data using the recursive helper
                        label_candidates, bbox_data = find_label_bbox_recursive(symbol_data_item)

                        # Choose the best label (longest or first non-empty)
                        label = max(label_candidates, key=len) if label_candidates else None
                        if label and len(label) > 100:
                            label = label[:100] + "..."

                        # Final check if label or bbox_data is missing
                        if not label or not bbox_data:
                            logging.warning(f"Symbol data missing required keys (label/text or bbox) after comprehensive recursive search: {symbol_data_item}. Skipping.")
                            continue

                        x1_px, y1_px, x2_px, y2_px = -1, -1, -1, -1 # Initialisiere mit ungültigen Werten

                        # Parsing der BBox-Daten (dieser Teil ist robust)
                        if isinstance(bbox_data, dict) and all(k in bbox_data for k in ['x', 'y', 'width', 'height']):
                            x1_px = int(bbox_data['x'] * img_width)
                            y1_px = int(bbox_data['y'] * img_height)
                            x2_px = x1_px + int(bbox_data['width'] * img_width)
                            y2_px = y1_px + int(bbox_data['height'] * img_height)
                        elif isinstance(bbox_data, list) and len(bbox_data) == 4 and all(isinstance(coord, (int, float)) for coord in bbox_data):
                            b0, b1, b2, b3 = [float(b) for b in bbox_data]
                            # Heuristik: Wenn die Werte klein sind (normalisiert), multipliziere mit Bildgröße
                            if all(val < 2.0 for val in [b0, b1, b2, b3]): # Annahme: normalisierte [x1, y1, x2, y2]
                                x1_px = int(b0 * img_width)
                                y1_px = int(b1 * img_height)
                                x2_px = int(b2 * img_width)
                                y2_px = int(b3 * img_height)
                            else: # Wahrscheinlich Pixelkoordinaten [x1_px, y1_px, x2_px, y2_px]
                                x1_px, y1_px, x2_px, y2_px = int(b0), int(b1), int(b2), int(b3)
                                x1_px, x2_px = sorted((x1_px, x2_px)) # Sicherstellen x1 < x2
                                y1_px, y2_px = sorted((y1_px, y2_px)) # Sicherstellen y1 < y2
                        elif isinstance(bbox_data, list): # Fallback für Listen > 4 Elemente oder verschachtelt
                            valid_bbox_coords = None # Dies wird gesetzt, wenn eine 4-Element-Liste gefunden wird
                            x_dict_found, y_dict_found, w_dict_found, h_dict_found = -1, -1, -1, -1 # Dies wird gesetzt, wenn ein Dict gefunden wird

                            # Zuerst nach einer 4-Elemente-Liste in der Liste suchen
                            for item in bbox_data:
                                if isinstance(item, list) and len(item) == 4 and all(isinstance(coord, (int, float)) for coord in item):
                                    valid_bbox_coords = [float(c) for c in item]
                                    logging.warning(f"Found nested BBox list for {label}, parsed first valid entry: {item}.")
                                    break
                            # Dann nach einem BBox-Dictionary in der Liste suchen (wenn noch keine 4-Element-Liste gefunden wurde)
                            if not valid_bbox_coords: 
                                for item in bbox_data:
                                    if isinstance(item, dict) and all(k in item for k in ['x', 'y', 'width', 'height']):
                                        x_dict_found = int(item['x'] * img_width)
                                        y_dict_found = int(item['y'] * img_height)
                                        w_dict_found = int(item['width'] * img_width)
                                        h_dict_found = int(item['height'] * img_height)
                                        logging.warning(f"Found nested BBox dict in list for {label}, parsed first valid entry: {item}.")
                                        break
                            
                            if valid_bbox_coords: # Wenn eine gültige 4-Element-Liste gefunden wurde
                                b0, b1, b2, b3 = valid_bbox_coords
                                if all(val < 2.0 for val in [b0, b1, b2, b3]): # Heuristik: Wahrscheinlich normalisiert
                                    x1_px = int(b0 * img_width)
                                    y1_px = int(b1 * img_height)
                                    x2_px = int(b2 * img_width)
                                    y2_px = int(b3 * img_height)
                                else: # Wahrscheinlich Pixelkoordinaten
                                    x1_px, y1_px, x2_px, y2_px = int(b0), int(b1), int(b2), int(b3)
                                    x1_px, x2_px = sorted((x1_px, x2_px))
                                    y1_px, y2_px = sorted((y1_px, y2_px))
                            elif x_dict_found != -1: # Wenn ein gültiges BBox-Dictionary gefunden wurde
                                x1_px = x_dict_found
                                y1_px = y_dict_found
                                x2_px = x_dict_found + w_dict_found
                                y2_px = y_dict_found + h_dict_found
                                x1_px, x2_px = sorted((x1_px, x2_px)) # Sicherstellen x1 < x2
                                y1_px, y2_px = sorted((y1_px, y2_px)) # Sicherstellen y1 < y2
                            else: # Wenn keine der oben genannten Strukturen gefunden wurde
                                logging.warning(f"Could not find valid bbox for {label} (raw: {bbox_data}). Skipping.")
                                continue
                        else: # Fängt alle anderen unerwarteten BBox-Formate ab
                            logging.warning(f"Unexpected or unparsable bbox format for symbol {label} (raw: {bbox_data}). Skipping.")
                            continue
                        
                        # BBox-Bereinigung und Validierung
                        x1_px = max(0, x1_px)
                        y1_px = max(0, y1_px)
                        x2_px = min(img_width, x2_px)
                        y2_px = min(img_height, y2_px)

                        if x1_px >= x2_px or y1_px >= y2_px:
                            logging.warning(f"Gereinigte BBox für Symbol {label} ist ungültig ({x1_px},{y1_px},{x2_px},{y2_px}). Überspringe.")
                            continue

                        cropped_img = img.crop((x1_px, y1_px, x2_px, y2_px))
                        # NEU: Speichere die normalisierte Original-BBox zusammen mit dem zugeschnittenen Bild
                        normalized_original_bbox = _normalize_bbox_from_pixels(x1_px, y1_px, x2_px, y2_px, img.size)
                        extracted_symbols.append((label, cropped_img, normalized_original_bbox))

                    # Post-Processing: Symbol-Text-Paare zusammenführen
                    merged_symbols: List[Tuple[str, Image.Image, BBox]] = []
                    processed_labels: Set[str] = set()

                    for i, (label1, img1, bbox1_orig) in enumerate(extracted_symbols):
                        if label1 in processed_labels:
                            continue

                        found_match = False
                        for j, (label2, img2, bbox2_orig) in enumerate(extracted_symbols):
                            if i == j or label2 in processed_labels:
                                continue
                            
                            # Verwende die originalen normalisierten BBoxes für die Distanzberechnung
                            distance = ((bbox1_orig['x'] + bbox1_orig['width']/2 - (bbox2_orig['x'] + bbox2_orig['width']/2)))**2 +

                            if (utils.string_similarity_ratio(label1.lower(), label2.lower()) > 0.7 or
                                label1.lower() in label2.lower() or label2.lower() in label1.lower()):
                                
                                distance = ((bbox1['x'] + bbox1['width']/2 - (bbox2['x'] + bbox2['width']/2))**2 +
                                            (bbox1['y'] + bbox1['height']/2 - (bbox2['y'] + bbox2['height']/2))**2)**0.5
                                
                                if distance < 0.1: # Schwellenwert für Distanz (anpassen)
                                    merged_x1 = min(bbox1_orig['x'], bbox2_orig['x']) # <- Geändert
                                    merged_y1 = min(bbox1_orig['y'], bbox2_orig['y']) # <- Geändert
                                    merged_x2 = max(bbox1_orig['x'] + bbox1_orig['width'], bbox2_orig['x'] + bbox2_orig['width']) # <- Geändert
                                    merged_y2 = max(bbox1_orig['y'] + bbox1_orig['height'], bbox2_orig['y'] + bbox2_orig['height']) # <- Geändert

                                    final_label = label2 if len(label2) > len(label1) else label1
                                    if final_label in ["symbols", "text"]:
                                        final_label = label1 if label1 not in ["symbols", "text"] else label2
                                        if final_label in ["symbols", "text"]:
                                            final_label = "Unbekanntes Symbol"

                                    merged_x1_px = int(merged_x1 * img_width)
                                    merged_y1_px = int(merged_y1 * img_height)
                                    merged_x2_px = int(merged_x2 * img_width)
                                    merged_y2_px = int(merged_y2 * img_height)

                                    final_cropped_img = img.crop((merged_x1_px, merged_y1_px, merged_x2_px, merged_y2_px))
                                    
                                    # Speichere die gemergte normalisierte BBox
                                    merged_bbox_normalized = BBox(x=merged_x1, y=merged_y1, width=merged_x2-merged_x1, height=merged_y2-merged_y1)
                                    merged_symbols.append((label1, img1, bbox1_orig))
                                    processed_labels.add(label1)
                                    processed_labels.add(label2)
                                    found_match = True
                                    logging.info(f"Merged BBoxes for '{label1}' and '{label2}' into '{final_label}'.")
                                    break # Nur eine Paarung pro Symbol suchen

                        if not found_match and label1 not in processed_labels:
                            # Wenn kein Merge-Partner gefunden wurde, das Symbol einfach hinzufügen
                            merged_symbols.append((label1, img1, bbox1_orig)) # <- Hinzugefügt
                            processed_labels.add(label1)
                    
                    extracted_symbols = merged_symbols # Ersetze die Liste der Symbole mit den gemergten
                    
            except Exception as e:
                logging.error(f"Fehler beim Öffnen/Verarbeiten des Bildes für Symbol-Extraktion für Pfad {image_path}: {e}", exc_info=True)
                return extracted_symbols
            
            return extracted_symbols
        
    def _extract_symbols_from_image(self, image_path: Path, model_info: Dict[str, Any]) -> List[Tuple[str, Image.Image, BBox]]:
        """Manager: Prüft Bildgröße und delegiert an den Helfer."""
        logging.info(f"Verarbeite Bild zur Symbol-Extraktion: {image_path.name}")
        all_extracted_symbols: List[Tuple[str, Image.Image, BBox]] = []
        try:
            with Image.open(image_path) as img:
                img_width, img_height = img.size
                
                if img_width > 2048 or img_height > 2048:
                    logging.info(f"Großes Bild erkannt ({img_width}x{img_height}). Starte Raster-Scan...")
                    temp_tile_dir = image_path.parent / f"temp_tiles_{image_path.stem}"
                    temp_tile_dir.mkdir(exist_ok=True)
                    
                    # Hier wurde utils.generate_raster_grid mit str(image_path) aufgerufen, was korrekt ist
                    tiles = utils.generate_raster_grid(str(image_path), tile_size=1024, overlap=128, output_folder=temp_tile_dir)
                    
                    for tile_info in tiles:
                        # Hier wurde Path(tile_info['path']) verwendet, was korrekt ist
                        symbols_in_tile = self._process_image_with_ai(Path(tile_info['path']), model_info)
                        all_extracted_symbols.extend(symbols_in_tile)
                    
                    # shutil.rmtree erwartet einen String oder Path, Path(temp_tile_dir) ist korrekt
                    shutil.rmtree(temp_tile_dir)
                else:
                    logging.info("Kleines Bild erkannt. Verarbeite direkt...")
                    all_extracted_symbols = self._process_image_with_ai(image_path, model_info)
        except Exception as e:
            logging.error(f"Fehler bei der Symbol-Extraktion für {image_path.name}: {e}", exc_info=True)

        logging.info(f"-> {len(all_extracted_symbols)} Symbole aus {image_path.name} extrahiert.")
        return all_extracted_symbols

    def _refine_single_bbox_visually(self, image_path: str, bbox: BBox) -> Optional[BBox]:
        """Sends a cropped image of a single bbox to the LLM for validation and refinement."""
        snippet_path = utils.crop_image_for_correction(image_path, cast(Dict, bbox), 0.05)
        if not snippet_path:
            return None

        try:
            refinement_prompt = self.config.get('prompts', {}).get('bbox_refinement_user_prompt')
            if not refinement_prompt:
                logging.error("bbox_refinement_user_prompt not found in config.")
                return None

            model_info = self.model_strategy.get('hotspot_model', self.model_strategy.get('detail_model'))
            if not model_info:
                logging.error("No model available for visual bbox refinement.")
                return None

            response = self.llm_handler.call_llm(
                model_info,
                self.config.get('prompts', {}).get('general_system_prompt', ''),
                refinement_prompt,
                snippet_path
            )

            if response and isinstance(response, dict) and response.get('is_single_component'):
                refined_bbox_local = response.get('refined_bbox')
                if refined_bbox_local:
                    # Convert local snippet coordinates back to global image coordinates
                    global_x = bbox['x'] + (refined_bbox_local['x'] * bbox['width'])
                    global_y = bbox['y'] + (refined_bbox_local['y'] * bbox['height'])
                    global_w = refined_bbox_local['width'] * bbox['width']
                    global_h = refined_bbox_local['height'] * bbox['height']
                    return BBox(x=global_x, y=global_y, width=global_w, height=global_h)
            
            return None

        finally:
            if snippet_path and os.path.exists(snippet_path):
                os.remove(snippet_path)
        
    def _iteratively_refine_llm_output_bboxes(self, 
                                            raw_llm_response: Dict[str, Any], 
                                            image_path: str,
                                            max_refinement_iterations: int,
                                            min_quality_to_keep: float) -> Dict[str, Any]:
        """
        Versucht, die Bounding Boxen iterativ zu verfeinern, jetzt mit visueller
        Validierung und Verfeinerung durch einen spezialisierten LLM-Aufruf.
        """
        logging.info("Starte erweiterte iterative Bounding-Box-Verfeinerung (mit visuellem Feedback)...")

        current_elements = [el.copy() for el in raw_llm_response.get('elements', [])]

        try:
            with Image.open(image_path) as img:
                image_dims = img.size
        except Exception as e:
            logging.error(f"Konnte Bildgröße für BBox-Verfeinerung nicht lesen: {e}")
            return raw_llm_response

        bbox_params = self.active_logic_parameters

        for iteration in range(max_refinement_iterations):
            logging.info(f"BBox-Verfeinerung Iteration {iteration + 1}/{max_refinement_iterations}...")
            modified_in_this_iteration = False
            elements_for_next_iteration = []

            for el in current_elements:
                if not (original_bbox := el.get('bbox')):
                    continue

                initial_quality = utils._evaluate_bbox_quality(image_path, cast(Dict[str, float], original_bbox), image_dims, bbox_params, el.get('type', 'Unknown'))

                best_bbox = original_bbox
                best_quality = initial_quality

                # Versuche visuelle Verfeinerung
                visually_refined_bbox = self._refine_single_bbox_visually(image_path, original_bbox)
                if visually_refined_bbox:
                    quality_after_visual = utils._evaluate_bbox_quality(image_path, cast(Dict[str, float], visually_refined_bbox), image_dims, bbox_params, el.get('type', 'Unknown'))
                    if quality_after_visual > best_quality:
                        best_bbox = visually_refined_bbox
                        best_quality = quality_after_visual
                        logging.info(f"Visuelle Verfeinerung hat BBox für '{el.get('label')}' verbessert.")
                
                # Versuche heuristische Verfeinerung (falls visuell nicht besser war)
                if best_bbox == original_bbox:
                    adjusted_bbox_contour = cast(BBox, utils._refine_bbox_with_contours(image_path, cast(Dict[str, float], original_bbox), image_dims))
                    quality_after_contour = utils._evaluate_bbox_quality(image_path, cast(Dict[str, float], adjusted_bbox_contour), image_dims, bbox_params, el.get('type', 'Unknown'))
                    if quality_after_contour > best_quality:
                        best_bbox = adjusted_bbox_contour
                        best_quality = quality_after_contour

                if best_quality > initial_quality:
                    el['bbox'] = best_bbox
                    modified_in_this_iteration = True

                if best_quality >= min_quality_to_keep:
                    elements_for_next_iteration.append(el)
                else:
                    logging.warning(f"Element '{el.get('label')}' nach Verfeinerung unter Qualitätslimit. Wird entfernt.")
            
            current_elements = elements_for_next_iteration
            if not modified_in_this_iteration and iteration > 0:
                logging.info("Keine weiteren BBox-Verbesserungen in dieser Iteration.")
                break

        logging.info(f"Finale BBox-Verfeinerung abgeschlossen. {len(current_elements)} Elemente verbleiben.")
        return {
            'elements': current_elements,
            'connections': raw_llm_response.get('connections', [])
        }

    def run_full_pipeline(self,
                          image_path: str,
                          progress_callback: Optional[ProgressCallback] = None,
                          output_dir: Optional[str] = None,
                          analysis_mode: str = 'hierarchical',
                          analysis_params_override: Optional[Dict[str, Any]] = None
                          ) -> Dict[str, Any]:
        """
        Führt die gesamte, nun iterative und harmonisierte Analyse-Pipeline aus.
        Behält alle ursprünglichen Phasen bei und bettet sie in eine
        selbstkorrigierende Schleife ein.
        """
        # --- INITIALISIERUNG ---
        self._initialize_run(image_path, progress_callback, analysis_params_override)
        logging.info(f"======== Full Analysis Pipeline started for {os.path.basename(image_path)} ========")

        truth_data = self._load_truth_data(image_path)
        
        # --- ITERATIVE SCHLEIFE ---
        max_iterations = self.active_logic_parameters.get('max_self_correction_iterations', 3)
        best_result: Dict[str, Any] = {"quality_score": -1.0, "final_ai_data": {}, "error_matrix": {}, "coarse_data": None, "hotspots": None}
        feedback_for_next_iteration: Optional[Dict[str, Any]] = None

        for i in range(1, max_iterations + 1):
            logging.info(f"--- Starting Analysis Iteration {i}/{max_iterations} ---")
            self._update_status_and_progress_internal(5 + int((i-1)/max_iterations * 80), f"Iteration {i}/{max_iterations}: Starte Analyse...")
            
            self._run_phase_1_pre_analysis(image_path)
            
            result_phase2 = self._run_phase_2_core_analysis(image_path, output_dir, analysis_mode, feedback=feedback_for_next_iteration)
            
            if result_phase2 is None or not self._analysis_results.get("elements"):
                logging.error(f"Iteration {i}: Keine Elemente nach Phase 2 gefunden. Breche Iterationen ab.")
                if i > 1: break 
                continue
            coarse_data, hotspots = result_phase2

            self._run_phase_2d_predictive_completion()

            current_score, current_errors = self._run_phase_3_validation_and_critic(truth_data)
            logging.info(f"Iteration {i} abgeschlossen mit Score: {current_score:.2f}")

            self._run_phase_3_5_adaptive_reanalysis(current_score)

            if current_score > best_result["quality_score"]:
                logging.info(f"Neues bestes Ergebnis in Iteration {i} gefunden! Score: {current_score:.2f}")
                best_result = {
                    "quality_score": current_score,
                    "final_ai_data": self._analysis_results.copy(),
                    "error_matrix": current_errors,
                    "coarse_data": coarse_data,
                    "hotspots": hotspots
                }
            
            if current_score >= self.active_logic_parameters.get('target_quality_score', 98.0):
                logging.info(f"Ziel-Score erreicht ({current_score:.2f}). Beende Iterationen frühzeitig.")
                break

            feedback_for_next_iteration = {
                "previous_data": self._analysis_results.copy(),
                "error_feedback": current_errors
            }
        
        if not best_result.get("final_ai_data"):
             logging.error("Nach allen Iterationen konnte kein valides Ergebnis erzeugt werden.")
             return {"error": "Pipeline failed to produce a valid result."}

        # --- FINALE AUSWERTUNG ---
        self._analysis_results = best_result["final_ai_data"]
        final_kpis = self._run_phase_4_kpi_and_cgm_generation(output_dir, truth_data)

        summary = {
            "image_name": os.path.basename(image_path),
            "quality_score": best_result["quality_score"],
            "final_kpis": final_kpis,
            **self._analysis_results
        }
        
        self._save_artifacts_and_learn(output_dir, image_path, summary, best_result["error_matrix"], best_result["coarse_data"], best_result["hotspots"])
        logging.info(f"======== Full Analysis Pipeline for {os.path.basename(image_path)} finished ========")
        return summary

    # --- NEUE HILFSFUNKTIONEN (ersetzen die Pylance-Fehler) ---

    def _initialize_run(self, image_path: str, progress_callback: Optional[ProgressCallback], analysis_params_override: Optional[Dict[str, Any]]):
        """Initialisiert den Zustand für einen neuen Analyse-Durchlauf."""
        self._reset_state()
        self.current_image_path = image_path
        self.progress_callback = progress_callback
        self.start_time = time.time()
        self.active_logic_parameters = self.config.get('logic_parameters', {}).copy()
        if analysis_params_override:
            self.active_logic_parameters.update(analysis_params_override)
            logging.info(f"Using overridden logic parameters: {analysis_params_override}")

    def _load_truth_data(self, image_path: str) -> Optional[Dict[str, Any]]:
        """Lädt die zugehörige Ground-Truth-Datei, falls vorhanden."""
        if not image_path: return None
        base_image_name_no_ext = os.path.splitext(os.path.basename(image_path))[0]
        truth_file_name = f"{base_image_name_no_ext}_truth.json"
        truth_path = os.path.join(os.path.dirname(image_path), truth_file_name)
        
        if os.path.exists(truth_path):
            logging.info(f"Ground-Truth-Datei '{truth_file_name}' gefunden.")
            try:
                with open(truth_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logging.error(f"Fehler beim Laden der Truth-Datei: {e}", exc_info=True)
        return None

    def _run_phase_2d_predictive_completion(self):
        """Führt die prädiktive Graph-Vervollständigung durch."""
        self._update_status_and_progress_internal(58, "Phase 2d: Schließe heuristische Lücken...")
        original_connection_count = len(self._analysis_results.get("connections", []))
        
        predicted_connections = utils.predict_and_complete_graph(
            elements=self._analysis_results.get("elements", []),
            connections=self._analysis_results.get("connections", []),
            logger=logging.getLogger(__name__)
        )
        
        if len(predicted_connections) > original_connection_count:
            self._analysis_results["connections"] = predicted_connections
            logging.info(f"{len(predicted_connections) - original_connection_count} Verbindungslücken prädiktiv geschlossen.")

    def _run_phase_3_validation_and_critic(self, truth_data: Optional[Dict[str, Any]]) -> Tuple[float, Dict[str, Any]]:
        """Führt die Validierung und die Analyst-Kritiker-Schleife durch."""
        self._update_status_and_progress_internal(60, "Phase 3: Validiere Ergebnis...")
        
        error_matrix: Dict[str, Any] = {}
        quality_score: float = 0.0

        if truth_data:
            kpi_results = utils.calculate_raw_kpis_with_truth(self._analysis_results, truth_data, self.knowledge_manager.get_all_aliases())
            self._analysis_results['final_kpis'] = kpi_results
            quality_score = utils.calculate_final_quality_score(kpi_results, use_truth_kpis=True)
            error_matrix = kpi_results.get("error_matrix", {})
        else:
            internal_kpis = utils.calculate_connectivity_kpis(
                cast(List[Dict], self._analysis_results.get("elements", [])),
                cast(List[Dict], self._analysis_results.get("connections", []))
            )
            self._analysis_results['final_kpis'] = internal_kpis
            quality_score = utils.calculate_final_quality_score(internal_kpis, use_truth_kpis=False)
            error_matrix = {"internal_connectivity_issues": internal_kpis.get("issues", [])}

        self._analysis_results['quality_score'] = quality_score
        
        self._update_status_and_progress_internal(70, "Phase 3: Starte Analyst-Kritiker-Schleife...")
        self._run_analyst_critic_loop(truth_data)
        
        return quality_score, error_matrix

    def _run_phase_4_kpi_and_cgm_generation(self, output_dir: Optional[str], truth_data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Generiert finale KPIs, den CGM-Graphen und den Python-Code."""
        self._update_status_and_progress_internal(85, "Phase 4: Erstelle finale Artefakte...")
        
        if not self.current_image_path:
             logging.error("Kann CGM nicht erstellen, da kein Bildpfad gesetzt ist.")
             return {}
        
        base_image_name_no_ext = os.path.splitext(os.path.basename(self.current_image_path))[0]
        current_output_dir = output_dir or self._get_log_directory() or os.getcwd()

        cgm_main_components = set(self.active_logic_parameters.get('cgm_main_components', []))
        cgm_network_data = utils.abstract_network_for_cgm(
            data=self._analysis_results,
            logger=logging.getLogger(__name__),
            type_aliases=self.knowledge_manager.get_all_aliases(),
            relevant_types=cgm_main_components
        )
        
        cgm_data_path = os.path.join(current_output_dir, f"{base_image_name_no_ext}_cgm_data.json")
        utils.save_json_output(cgm_network_data, cgm_data_path)

        cgm_graph_path = os.path.join(current_output_dir, f"{base_image_name_no_ext}_cgm_graph.png")
        utils.draw_cgm_graph(cgm_network_data, cgm_graph_path, logging.getLogger(__name__), config_params=self.active_logic_parameters)

        self._generate_cgm_python_code(cgm_network_data, current_output_dir, base_image_name_no_ext)
        
        return self._analysis_results.get('final_kpis', {})

    def _save_artifacts_and_learn(self, output_dir: Optional[str], image_path: str, summary: Dict[str, Any], error_matrix: Dict[str, Any], coarse_data: Optional[Dict], hotspots: Optional[List]):
        """Speichert das Debug-Bild und aktualisiert die Wissensbasis."""
        self._update_status_and_progress_internal(95, "Finalisiere: Speichere Artefakte & lerne...")
        
        base_image_name_no_ext = os.path.splitext(os.path.basename(image_path))[0]
        current_output_dir = output_dir or self._get_log_directory() or os.getcwd()

        full_debug_image_path = os.path.join(current_output_dir, f"{base_image_name_no_ext}_FULL_ANALYSIS_DEBUG.png")
        if coarse_data and hotspots:
            utils.export_full_analysis_debug_image(
                original_image_path=image_path,
                output_path=full_debug_image_path,
                analysis_data=self._analysis_results,
                coarse_graph_data=coarse_data,
                hotspot_zones=hotspots,
                excluded_zones=self._excluded_zones,
                metadata=self._analysis_results.get("metadata", {}),
                logger=logging.getLogger(__name__),
                config_params=self.active_logic_parameters
            )

        truth_data_for_learning = self._load_truth_data(image_path)
        if truth_data_for_learning:
            self.knowledge_manager.add_learning_pattern(summary, truth_data_for_learning, error_matrix)
    
    def _get_log_directory(self) -> Optional[str]:
        """Hilfsfunktion, um das aktuelle Log-Verzeichnis zu finden."""
        for handler in logging.getLogger().handlers:
            if isinstance(handler, logging.FileHandler):
                # Extrahiert den Verzeichnispfad aus dem Handler
                log_file_path = handler.baseFilename
                return os.path.dirname(log_file_path)
        return None

    def _generate_cgm_python_code(self, cgm_network_data: Dict, output_dir: str, base_name: str):
        """Kapselt die Logik zur Generierung des CGM Python-Codes."""
        cgm_code_gen_prompt = self.config.get('prompts', {}).get('cgm_code_generation_user_prompt')
        system_prompt_general = self.config.get('prompts', {}).get('general_system_prompt')
        
        if not cgm_code_gen_prompt or not system_prompt_general:
            logging.error("Prompts für CGM-Code-Generierung unvollständig in config.yaml.")
            return

        code_gen_model_info = self.model_strategy.get('code_gen_model')
        if not code_gen_model_info:
            logging.warning("Kein 'code_gen_model' in der Strategie definiert. Verwende 'detail_model' als Fallback.")
            code_gen_model_info = self.model_strategy.get('detail_model')

        # Explizite Prüfung, ob nach dem Fallback ein Modell vorhanden ist.
        if code_gen_model_info:
            formatted_prompt = cgm_code_gen_prompt.format(json_data=json.dumps(cgm_network_data, indent=2))
            
            code_response = self.llm_handler.call_llm(
                model_info=code_gen_model_info,
                system_prompt=system_prompt_general,
                user_prompt=formatted_prompt,
                image_path=None,
                use_cache=False,
                expected_json_keys=None
            )

            if code_response:
                code_text = str(code_response)
                if "```python" in code_text:
                    code_text = code_text.split("```python", 1)[1]
                if "```" in code_text:
                    code_text = code_text.split("```")[0]
                
                cgm_py_path = os.path.join(output_dir, f"{base_name}_cgm_network_generated.py")
                try:
                    with open(cgm_py_path, 'w', encoding='utf-8') as f:
                        f.write(code_text.strip())
                    logging.info(f"CGM Python-Code gespeichert: {cgm_py_path}")
                except IOError as e:
                    logging.error(f"Fehler beim Speichern des CGM Python-Codes: {e}")
            else:
                logging.warning("Kein valider Python-Code vom LLM für CGM erhalten.")
        else:
            logging.error("Kein Modell für die CGM-Code-Generierung verfügbar.")

    def _find_bbox_for_component(self, component_id: str) -> Optional[Dict[str, float]]:
        """Eine Hilfsfunktion, um die Bounding-Box einer Komponente zu finden."""
        for element in self._analysis_results.get("elements", []):
            if element.get("id") == component_id:
                return cast(Optional[Dict[str, float]], element.get("bbox"))
        return None
    

    def _run_phase_1_pre_analysis(self, image_path: str):
        """
        Führt die Vor-Analyse durch und stellt eine korrekte Validierung
        der Legendensymbole sicher.
        """
        logging.info("--- Running Phase 1: Global Pre-Analysis ---")
        system_prompt = self.config.get('prompts', {}).get('general_system_prompt', 'You are an expert in analyzing technical diagrams.')
        metadata_prompt = self.config.get('prompts', {}).get('metadata_extraction_user_prompt')
        if not metadata_prompt:
            logging.error("Metadata extraction prompt fehlt in config.yaml. Breche Phase 1 ab.")
            return None
        legend_prompt = self.config.get('prompts', {}).get('legend_extraction_user_prompt')
        if not legend_prompt:
            logging.error("Legend extraction prompt fehlt in config.yaml. Breche Phase 1 ab.")
            return None
        
        model_info = self.model_strategy.get('meta_model')
        if not model_info:
            logging.error("Meta model not defined in strategy. Aborting Phase 1.")
            return

        metadata_response = self.llm_handler.call_llm(
            model_info, system_prompt, metadata_prompt, image_path,
            expected_json_keys=["project", "title", "version", "date", "metadata_bbox"]
        )
        
        # NEU/GEÄNDERT: Explizite Initialisierung von metadata_dict und Prüfung des Antworttyps
        metadata_dict: Optional[Dict[str, Any]] = None
        if metadata_response and isinstance(metadata_response, dict):
            metadata_dict = metadata_response
            self._analysis_results['metadata'] = metadata_dict
            logging.info(f"Successfully extracted metadata: {metadata_dict}")
        else:
            logging.warning(f"LLM response for metadata was not a dict or was None: {metadata_response}. Skipping metadata extraction.")
            self._analysis_results['metadata'] = {} # Setze leeres Dict für Konsistenz

        # NEU/GEÄNDERT: Robustere Verarbeitung der Metadaten-BBox
        # Zugriff auf metadata_dict.get() ist jetzt sicher.
        raw_metadata_bbox = metadata_dict.get("metadata_bbox") if metadata_dict else None
        parsed_metadata_bbox: Optional[Dict[str, float]] = None

        # Temporäre Image-Größen-Initialisierung, falls `image_path` Probleme macht
        # Dieser Block war zuvor schon in Ihrer Funktion, ich kopiere ihn hier nur zur Kontextualisierung
        img_width, img_height = 1, 1
        try:
            with Image.open(image_path) as img:
                img_width, img_height = img.size
        except Exception as e:
            logging.error(f"Konnte Bildgröße für Metadaten-BBox-Parsing nicht lesen: {e}", exc_info=True)
            # Setze img_width/height auf Fallback, um Division by Zero zu vermeiden
            img_width, img_height = 1000, 1000 # Fallback-Werte

        # Versuche, die BBox zu parsen und zu normalisieren
        if isinstance(raw_metadata_bbox, dict) and all(k in raw_metadata_bbox for k in ['x', 'y', 'width', 'height']):
            parsed_metadata_bbox = cast(Dict[str, float], raw_metadata_bbox) # already in correct dict format
        elif isinstance(raw_metadata_bbox, list) and len(raw_metadata_bbox) == 4 and all(isinstance(coord, (int, float)) for coord in raw_metadata_bbox):
            try:
                x, y, width, height = raw_metadata_bbox
                parsed_metadata_bbox = {'x': float(x), 'y': float(y), 'width': float(width), 'height': float(height)}
                logging.warning(f"Metadata bbox received in list format: {raw_metadata_bbox}. Converted to dict.")
            except (ValueError, IndexError):
                logging.error(f"Could not convert metadata bbox list {raw_metadata_bbox} to dict format.", exc_info=True)
        
        # Wenn eine gültige BBox geparst wurde, füge sie zu den ausgeschlossenen Zonen hinzu
        if parsed_metadata_bbox:
            self._excluded_zones.append(parsed_metadata_bbox)
            logging.info(f"Identified metadata area to be excluded: {parsed_metadata_bbox}")
        else:
            logging.warning(f"Metadata bbox is malformed or unparsable: {raw_metadata_bbox}. Skipping exclusion for this zone.")

        # NEU/GEÄNDERT: Robustere Verarbeitung der Legenden-BBox
        legend_response = self.llm_handler.call_llm(
            model_info, system_prompt, legend_prompt, image_path,
            expected_json_keys=["symbol_map", "legend_bbox"]
        )
        
        # NEU/GEÄNDERT: Explizite Initialisierung von legend_dict und Prüfung des Antworttyps
        legend_dict: Optional[Dict[str, Any]] = None
        if legend_response and isinstance(legend_response, dict):
            legend_dict = legend_response
        else:
            logging.warning(f"LLM response for legend was not a dict or was None: {legend_response}. Skipping legend extraction.")
        
        # NEU/GEÄNDERT: Robustere Verarbeitung der Legenden-BBox und Symbol-Map
        # Zugriff auf legend_dict.get() ist jetzt sicher.
        legend_bbox_raw = legend_dict.get("legend_bbox") if legend_dict else None
        legend_bbox: Optional[Dict[str, float]] = None

        if isinstance(legend_bbox_raw, dict) and all(k in legend_bbox_raw for k in ['x', 'y', 'width', 'height']):
            legend_bbox = legend_bbox_raw
        elif isinstance(legend_bbox_raw, list) and len(legend_bbox_raw) == 4 and all(isinstance(coord, (int, float)) for coord in legend_bbox_raw):
            try:
                x, y, width, height = legend_bbox_raw
                legend_bbox = {'x': float(x), 'y': float(y), 'width': float(width), 'height': float(height)}
                logging.warning(f"Legend bbox received in list format: {legend_bbox_raw}. Converted to dict.")
            except (ValueError, IndexError):
                logging.error(f"Could not convert legend bbox list {legend_bbox_raw} to dict format.", exc_info=True)
            
        if legend_bbox: # Überprüft, ob die BBox erfolgreich geparst wurde
            self._excluded_zones.append(legend_bbox)
            logging.info(f"Identified legend area to be excluded: {legend_bbox}")
        else:
            logging.warning(f"Legend bbox is malformed or unparsable: {legend_bbox_raw}. Skipping exclusion for this zone.")

        # NEU/GEÄNDERT: symbol_map Zugriff ist jetzt sicher, da legend_dict geprüft wird
        symbol_map = legend_dict.get("symbol_map", {}) if legend_dict else {}
        
        validated_symbol_map = {}
        if isinstance(symbol_map, dict):
            for key, value in symbol_map.items():
                symbol_type_name = None
                if isinstance(value, str):
                    symbol_type_name = value
                elif isinstance(value, dict) and 'type' in value:
                    symbol_type_name = value['type']
                elif isinstance(value, list) and len(value) >= 4 and all(isinstance(x, (int, float)) for x in value):
                    logging.warning(f"Malformed symbol_map value for key '{key}'. Looks like a bbox, not a type name: {value}. Skipping this entry.")
                    continue
                else:
                    logging.warning(f"Unexpected type for symbol_map value '{value}' (Key: '{key}'). Expected string or dict with 'type', got {type(value)}. Skipping validation for this entry.")
                    continue

                if symbol_type_name:
                    element_type_data = self.knowledge_manager.find_element_type_by_name(symbol_type_name) 
                    
                    if not element_type_data:
                        logging.warning(f"Typ '{symbol_type_name}' nicht im Alias-Verzeichnis gefunden. Versuche Parent-Heuristik...")
                        element_type_data = self.knowledge_manager.find_parent_type_by_heuristic(symbol_type_name)

                    if element_type_data:
                        validated_symbol_map[key] = element_type_data['name']
                    else:
                        logging.error(f"Typ '{symbol_type_name}' aus der Legende konnte keinem bekannten Typ zugeordnet werden. Wird als neuer, unbekannter Typ behandelt.")
                        validated_symbol_map[key] = symbol_type_name
        else:
            logging.warning(f"Malformed 'symbol_map' in LLM response. Expected a dictionary, got {type(symbol_map)}. Skipping symbol map validation.")

        self._global_knowledge_repo['symbol_map'] = validated_symbol_map
        logging.info(f"Built knowledge repository with {len(validated_symbol_map)} validated symbol mappings.")

        # NEU: Verarbeite die extrahierte Linien-Semantik
        line_map = legend_dict.get("line_map", {}) if legend_dict else {}
        if line_map:
            self._global_knowledge_repo['line_map'] = line_map
            logging.info(f"Extracted {len(line_map)} line semantic rules from legend.")

        logging.info("Phase 1 completed.")

    def _run_phase_2_core_analysis(self, image_path: str, output_dir: Optional[str] = None, analysis_mode: str = 'hierarchical', feedback: Optional[Dict] = None) -> Optional[Tuple[Dict, List]]:
        """
        Führt eine Analyse-Strategie aus, die nun eine Inventar-Voranalyse beinhaltet.
        'hierarchical': Nutzt Inventar-Analyse, Grobanalyse, Hotspot-Detailanalyse, globale Detail-Analyse und Fusion.
        'vertical_strips': Nutzt eine Analyse mit vertikalen Streifen als alternative Methode oder Fallback.
        """
        logging.info(f"--- Running Phase 2 (Mode: {analysis_mode.upper()}): Analysis ---")

        # --- 1. Vorbereitung und saubere Initialisierung ---
        coarse_graph_data: Dict[str, Any] = {"elements": [], "connections": []}
        inventory_list: List[Dict[str, str]] = []
        hotspot_zones: List[Dict] = []
        all_fine_grained_tiles: List[Dict] = []
        use_fusion = False

        try:
            with Image.open(image_path) as img:
                img_width, img_height = img.size
        except Exception as e:
            logging.error(f"Image not found or could not be opened: {e}")
            return None

        temp_dir_path = Path(output_dir or os.path.dirname(image_path)) / "temp_tiles"
        temp_dir_path.mkdir(exist_ok=True, parents=True)

        # --- NEU: Phase 2a: Inventar-Analyse ---
        self._update_status_and_progress_internal(20, "Phase 2a: Erstelle Inventar...")
        inventory_prompt = self.config.get('prompts', {}).get('inventory_extraction_user_prompt')
        system_prompt_inventory = self.config.get('prompts', {}).get('general_system_prompt')
        meta_model_info = self.model_strategy.get('meta_model')

        if inventory_prompt and system_prompt_inventory and meta_model_info:
            inventory_response = self.llm_handler.call_llm(
                meta_model_info, system_prompt_inventory, inventory_prompt, image_path,
                expected_json_keys=["inventory"]
            )
            if inventory_response and isinstance(inventory_response, dict) and isinstance(inventory_response.get('inventory'), list):
                inventory_list = inventory_response['inventory']
                logging.info(f"Inventar-Analyse erfolgreich: {len(inventory_list)} potenzielle Komponenten identifiziert.")
            else:
                logging.warning("Inventar-Analyse fehlgeschlagen oder gab leere Liste zurück.")
        else:
            logging.warning("Prompts oder Modell für Inventar-Analyse nicht in config.yaml gefunden. Schritt wird übersprungen.")

        inventory_json_for_prompt = json.dumps(inventory_list, indent=2)

        if analysis_mode == 'hierarchical':
            logging.info("STRATEGIE: Führe 'Hierarchical Fusion'-Analyse aus.")
            
            # STUFE 2b: Globale Analyse (Coarse Analysis)
            self._update_status_and_progress_internal(25, "Phase 2b: Globale Grobanalyse...")
            coarse_analysis_prompt_template = self.config.get('prompts', {}).get('coarse_analysis_user_prompt')
            if not coarse_analysis_prompt_template:
                logging.error("Coarse analysis prompt fehlt in config.yaml. Breche Phase 2 ab.")
                return None
            
            formatted_coarse_analysis_prompt = coarse_analysis_prompt_template.replace(
                "{excluded_zones}", json.dumps(self._excluded_zones)
            )
            
            system_prompt_coarse = self.config.get('prompts', {}).get('general_system_prompt')
            coarse_model_info = self.model_strategy.get('coarse_model', self.model_strategy.get('detail_model'))
            if not coarse_model_info:
                logging.error("Kein Modell für die Grobanalyse in der Strategie definiert. Wechsle zu Fallback-Strategie.")
                analysis_mode = 'vertical_strips'
                coarse_graph_data = {"elements": [], "connections": []}
            else:
                llm_response = self.llm_handler.call_llm(
                    coarse_model_info, system_prompt_coarse, formatted_coarse_analysis_prompt, image_path,
                    expected_json_keys=["elements", "connections"]
                )
                coarse_graph_data = llm_response if isinstance(llm_response, dict) else {"elements": [], "connections": []}

            min_elements_success = self.active_logic_parameters.get('min_elements_for_global_success', 4)

            if coarse_graph_data and len(coarse_graph_data.get('elements', [])) >= min_elements_success:
                logging.info(f"Globale Analyse erfolgreich. {len(coarse_graph_data['elements'])} Hauptkomponenten gefunden.")
                hotspot_zones = [el['bbox'] for el in coarse_graph_data.get('elements', []) if el.get('bbox')]
                hotspot_zones = utils.merge_overlapping_boxes(hotspot_zones)
                
                logging.info("STRATEGIE: Generiere Hotspot-Kacheln für Detail-Analyse.")
                hotspot_tiles = utils.generate_raster_grid_in_zones(
                    image_path, hotspot_zones,
                    tile_size=self.active_logic_parameters.get('hotspot_tile_size', 1024),
                    overlap=self.active_logic_parameters.get('hotspot_tile_overlap', 128),
                    output_folder=temp_dir_path
                )
                all_fine_grained_tiles.extend(hotspot_tiles)

                logging.info("STRATEGIE: Generiere globale feine Rasterkacheln für umfassende Detail-Analyse.")
                global_fine_tiles = utils.generate_raster_grid(
                    image_path,
                    tile_size=self.active_logic_parameters.get('global_fine_tile_size', 512),
                    overlap=self.active_logic_parameters.get('global_fine_tile_overlap', 64),
                    excluded_zones=self._excluded_zones,
                    output_folder=temp_dir_path
                )
                all_fine_grained_tiles.extend(global_fine_tiles)
                use_fusion = True
            else:
                logging.warning("Globale Analyse fehlgeschlagen. Wechsle zu Fallback-Strategie 'Vertical Strips'.")
                analysis_mode = 'vertical_strips'

        if analysis_mode == 'vertical_strips':
            if not use_fusion:
                logging.info("STRATEGIE: Führe Analyse mit 'Vertical Strips' aus.")
            
            vertical_strip_tiles = utils.generate_vertical_strips(
                image_path,
                num_strips=self.active_logic_parameters.get('fine_grained_strips_count', 7),
                overlap_percent=self.active_logic_parameters.get('vertical_strip_overlap_percent', 15),
                output_folder=temp_dir_path
            )
            all_fine_grained_tiles.extend(vertical_strip_tiles)
            hotspot_zones = []
            use_fusion = False

        if not all_fine_grained_tiles:
            logging.error("Keine Kacheln für Detail-Analyse generiert. Breche Phase 2 ab.")
            if temp_dir_path.exists(): shutil.rmtree(temp_dir_path)
            return coarse_graph_data, hotspot_zones

        # ... (Rest der Funktion bleibt gleich, verwendet aber `all_fine_grained_tiles`
        # und wird im nächsten Schritt angepasst, um `inventory_json_for_prompt` zu nutzen)
        
        # STUFE 2c: Parallele Verarbeitung aller Kacheln
        few_shot_examples = self.knowledge_manager.get_few_shot_examples(image_name=os.path.basename(image_path), k=1)
        few_shot_prompt_part_str = f"\n\nFEW-SHOT EXAMPLE (Follow this structure exactly):\n{json.dumps(few_shot_examples[0], indent=2)}" if few_shot_examples else ""
        
        knowledge_context = {
            "known_types": self.knowledge_manager.get_known_types(),
            "type_aliases": self.knowledge_manager.get_all_aliases(),
            "preliminary_inventory": inventory_list # NEU: Füge die Inventarliste hinzu
        }
        if line_map := self._global_knowledge_repo.get('line_map'):
            knowledge_context['legend_line_rules'] = line_map
        
        known_types_json = json.dumps(knowledge_context, indent=2)

        raster_analysis_user_prompt_template = self.config.get('prompts', {}).get('raster_analysis_user_prompt_template')
        system_prompt_fine = self.config.get('prompts', {}).get('general_system_prompt')
        detail_model_info = self.model_strategy.get('detail_model')
        if not detail_model_info:
            logging.error("Kein Modell für die Feinanalyse in der Strategie definiert. Breche Phase 2 ab.")
            if temp_dir_path.exists(): shutil.rmtree(temp_dir_path)
            return coarse_graph_data, hotspot_zones

        if not all([raster_analysis_user_prompt_template, system_prompt_fine, detail_model_info]):
            logging.error("Konfiguration für Detail-Analyse unvollständig. Breche Phase 2 ab.")
            if temp_dir_path.exists(): shutil.rmtree(temp_dir_path)
            return coarse_graph_data, hotspot_zones

        self._update_status_and_progress_internal(30, "Phase 2c: Parallele Detail-Analyse...")
        raw_results: List[TileResult] = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.active_logic_parameters.get('llm_timeout_executor_workers', 1)) as executor:
            # Vorab-Scan
            future_to_tile_prescan = {}
            hotspot_model_info = self.model_strategy.get('hotspot_model')
            if not hotspot_model_info:
                logging.error("Kein 'hotspot_model' für Vorab-Scan definiert. Breche ab.")
                return coarse_graph_data, hotspot_zones
            
            pre_scan_prompt = "Analysiere dieses Bildsegment. Gib NUR eine JSON-Liste der P&ID-Komponententypen zurück, die du siehst. Beispiel: [\"Pump\", \"Valve\"]"
            for tile in all_fine_grained_tiles:
                future = executor.submit(self.llm_handler.call_llm, hotspot_model_info, system_prompt_fine, pre_scan_prompt, tile['path'], True, 60, None)
                future_to_tile_prescan[future] = tile
            
            pre_scan_results = {}
            for future in concurrent.futures.as_completed(future_to_tile_prescan):
                tile = future_to_tile_prescan[future]
                pre_scan_results[tile['path']] = future.result() or []

            # Hauptanalyse
            future_to_tile_main = {}
            for tile in all_fine_grained_tiles:
                relevant_types = pre_scan_results.get(tile['path'], [])
                current_knowledge_context = knowledge_context.copy()
                if relevant_types:
                    current_knowledge_context["relevant_types_in_tile"] = relevant_types
                
                current_known_types_json = json.dumps(current_knowledge_context, indent=2)
                # NEU: Formatiere das Feedback für den Prompt, falls vorhanden
                feedback_text = ""
                if feedback and feedback.get("error_feedback"):
                    feedback_text = "\n**PREVIOUS ERRORS TO CORRECT:**\n" + json.dumps(feedback.get("error_feedback"), indent=2)

                optimized_raster_prompt = raster_analysis_user_prompt_template.format(
                    known_types_json=current_known_types_json,
                    few_shot_prompt_part=few_shot_prompt_part_str,
                    error_feedback=feedback_text # Füge das formatierte Feedback hinzu
                )
                
                future = executor.submit(self.llm_handler.call_llm, detail_model_info, system_prompt_fine, optimized_raster_prompt, tile['path'], True, self.active_logic_parameters.get('llm_default_timeout', 120), ["elements", "connections"])
                future_to_tile_main[future] = tile

            for i, future in enumerate(concurrent.futures.as_completed(future_to_tile_main)):
                tile = future_to_tile_main[future]
                try:
                    if response_data := future.result():
                        raw_results.append(TileResult(tile_coords=tile['coords'], tile_width=tile['tile_width'], tile_height=tile['tile_height'], data=cast(utils.TileData, response_data)))
                except Exception as exc:
                    logging.error(f"Fehler bei Haupt-LLM-Analyse für Kachel '{tile['path']}': {exc}", exc_info=True)
                
                self._update_status_and_progress_internal(30 + int((i + 1) / len(all_fine_grained_tiles) * 25), f"Analysiere Kacheln... ({i+1}/{len(all_fine_grained_tiles)})")

        logging.info(f"Detail-Analyse abgeschlossen. {len(raw_results)} Kachel-Ergebnisse erhalten.")
        
        # ... Rest der Funktion (Synthese, BBox-Verfeinerung, Fusion) bleibt identisch ...
        self._update_status_and_progress_internal(55, "Phase 2d: Synthese & Fusion...")
        
        synthesizer_config = utils.SynthesizerConfig(
            iou_match_threshold=self.active_logic_parameters.get('iou_match_threshold', 0.5),
            border_coords_tolerance_factor_x=self.active_logic_parameters.get('border_coords_tolerance_factor_x', 0.015),
            border_coords_tolerance_factor_y=self.active_logic_parameters.get('border_coords_tolerance_factor_y', 0.015)
        )
        
        self.synthesizer = utils.GraphSynthesizer(raw_results=raw_results, image_width=img_width, image_height=img_height, config=synthesizer_config)
        synthesized_fine_data = self.synthesizer.synthesize()

        logging.info(f"Synthese fand {len(synthesized_fine_data.get('elements', []))} Elemente und {len(synthesized_fine_data.get('connections', []))} Verbindungen.")

        min_elements_for_success = self.active_logic_parameters.get('min_elements_for_successful_synthesis', 3)
        if analysis_mode == 'hierarchical' and (not synthesized_fine_data.get("elements") or len(synthesized_fine_data["elements"]) < min_elements_for_success):
            logging.warning(f"Primäre Analyse lieferte nur {len(synthesized_fine_data.get('elements', []))} Elemente. Aktiviere Rettungsanker: 'Vertical Strips'...")
            if temp_dir_path.exists(): shutil.rmtree(temp_dir_path)
            return self._run_phase_2_core_analysis(image_path, output_dir, analysis_mode='vertical_strips')
        
        refined_data = self._iteratively_refine_llm_output_bboxes(
            cast(Dict[str, Any], synthesized_fine_data), image_path,
            max_refinement_iterations=self.active_logic_parameters.get('max_bbox_refinement_iterations', 3),
            min_quality_to_keep=self.active_logic_parameters.get('min_quality_to_keep_bbox', 0.5),
        )
        synthesized_fine_data['elements'] = refined_data['elements']

        if use_fusion and coarse_graph_data:
            logging.info("Führe Fusions-Logik durch...")
            final_elements_map = {el['id']: el for el in synthesized_fine_data.get('elements', [])}
            final_connections = list(synthesized_fine_data.get('connections', []))
            
            fine_grained_graph = nx.DiGraph()
            for el_id in final_elements_map: fine_grained_graph.add_node(el_id)
            for conn in final_connections:
                if conn.get('from_id') in final_elements_map and conn.get('to_id') in final_elements_map: 
                    fine_grained_graph.add_edge(conn['from_id'], conn['to_id'])

            num_gaps_filled = 0
            for coarse_conn in coarse_graph_data.get('connections', []):
                from_id, to_id = coarse_conn.get('from_id'), coarse_conn.get('to_id')
                if from_id in final_elements_map and to_id in final_elements_map:
                    if not fine_grained_graph.has_edge(from_id, to_id) and not nx.has_path(fine_grained_graph, from_id, to_id):
                        new_predicted_conn: Connection = {
                            "from_id": from_id, "to_id": to_id, "from_port_id": "", "to_port_id": "",
                            "color": None, "style": None, "predicted": True,
                            "status": "predicted_from_coarse_analysis", "original_border_coords": None
                        }
                        final_connections.append(new_predicted_conn)
                        fine_grained_graph.add_edge(from_id, to_id)
                        num_gaps_filled += 1
            
            for coarse_el in coarse_graph_data.get('elements', []):
                if coarse_el.get('id') and coarse_el['id'] not in final_elements_map:
                    final_elements_map[coarse_el['id']] = coarse_el
            
            if num_gaps_filled > 0: logging.info(f"Fusion hat {num_gaps_filled} Lücken gefüllt.")
            
            self._analysis_results["elements"] = list(final_elements_map.values())
            self._analysis_results["connections"] = final_connections
        else:
            logging.info("Keine Fusions-Logik angewendet.")
            self._analysis_results["elements"] = synthesized_fine_data.get("elements", [])
            self._analysis_results["connections"] = synthesized_fine_data.get("connections", [])
        
        if temp_dir_path.exists():
            try:
                shutil.rmtree(temp_dir_path)
            except Exception as e:
                logging.warning(f"Konnte Temp-Ordner nicht löschen: {e}")
                
        return coarse_graph_data, hotspot_zones

    def _apply_correction(self, correction_data: Dict[str, Any]) -> bool:
        """
        Wendet eine einzelne, strukturierte Korrektur-Aktion auf den Graphen an.
        Gibt True zurück, wenn der Graph modifiziert wurde.
        """
        corrections = correction_data.get("corrections", [])
        if not isinstance(corrections, list) or not corrections:
            logging.warning("Keine gültigen Korrektur-Aktionen im Response gefunden.")
            return False

        was_modified = False
        action = corrections[0] # Wende nur die erste, priorisierte Aktion an
        action_type = action.get("action")

        if action_type == "CHANGE_TYPE":
            element_id = action.get("element_id")
            new_type = action.get("new_type")
            element = next((el for el in self._analysis_results["elements"] if el.get("id") == element_id), None)
            if element and new_type:
                logging.info(f"AKTION: Ändere Typ von Element '{element.get('label')}' zu '{new_type}'.")
                element["type"] = new_type
                was_modified = True

        elif action_type == "ADD_CONNECTION":
            from_id, to_id = action.get("from_id"), action.get("to_id")
            if from_id and to_id and from_id in [e['id'] for e in self._analysis_results['elements']] and to_id in [e['id'] for e in self._analysis_results['elements']]:
                logging.info(f"AKTION: Füge Verbindung von '{from_id}' zu '{to_id}' hinzu.")
                new_conn: Connection = {
                                    "from_id": from_id,
                                    "to_id": to_id,
                                    "from_port_id": "",
                                    "to_port_id": "",
                                    "color": None, # NEU
                                    "style": None, # NEU
                                    "predicted": True,
                                    "status": "corrected_by_llm",
                                    "original_border_coords": None
                                }
                self._analysis_results["connections"].append(new_conn)
                was_modified = True

        elif action_type == "DELETE_ELEMENT":
            element_id = action.get("element_id")
            initial_count = len(self._analysis_results["elements"])
            self._analysis_results["elements"] = [el for el in self._analysis_results["elements"] if el.get("id") != element_id]
            # Entferne auch alle Verbindungen, die mit diesem Element zusammenhängen
            self._analysis_results["connections"] = [c for c in self._analysis_results["connections"] if c.get('from_id') != element_id and c.get('to_id') != element_id]
            if len(self._analysis_results["elements"]) < initial_count:
                logging.info(f"AKTION: Lösche Element '{element_id}' und zugehörige Verbindungen.")
                was_modified = True

        return was_modified

    def _trigger_llm_correction(self, problem: Dict) -> Tuple[Optional[Union[Dict, List]], Optional[str]]:
        """
        Bereitet den Kontext vor, ruft das LLM für eine Korrektur auf und gibt
        sowohl die Antwort als auch den Pfad zum visuellen Snippet zurück.
        """
        if self.current_image_path is None:
            logging.error("current_image_path ist None. Kann keine LLM-Korrektur auslösen.")
            return None, None

        correction_context_margin = self.active_logic_parameters.get('correction_context_margin', 0.15)
        problem_bbox = problem.get("location_bbox")
        
        cropped_image_path = None
        if problem_bbox:
            cropped_image_path = utils.crop_image_for_correction(
                self.current_image_path, problem_bbox, correction_context_margin
            )

        context_snippet = utils.get_local_graph_context(
            problem.get("context", {}).get("element_label", ""),
            self._analysis_results["elements"],
            self._analysis_results["connections"]
        )

        system_prompt = self.config.get('prompts', {}).get('correction_system_prompt')
        user_prompt_template = self.config.get('prompts', {}).get('correction_user_prompt_template')
        
        if not system_prompt or not user_prompt_template:
            logging.error("Korrektur-Prompts fehlen in der Konfiguration.")
            return None, cropped_image_path

        formatted_user_prompt = user_prompt_template.format(
            problem_description=problem.get("description"),
            context_snippet=context_snippet
        )

        correction_model_info = self.model_strategy.get('correction_model')
        if not correction_model_info:
            logging.error("Kein 'correction_model' in der Strategie definiert.")
            return None, cropped_image_path

        response = self.llm_handler.call_llm(
            model_info=correction_model_info,
            system_prompt=system_prompt,
            user_prompt=formatted_user_prompt,
            image_path=cropped_image_path,
            expected_json_keys=["corrections"]
        )

        return response, cropped_image_path

    def _run_phase_3_self_correction(self, truth_data: Optional[Dict] = None):
        """
        Führt eine iterative Selbstkorrektur durch.
        NEU: Wenn 'truth_data' vorhanden ist, wird der "Trainingsmodus" aktiviert,
        der die KI zwingt, sich direkt an der Wahrheit zu korrigieren.
        """
        logging.info("--- Running Phase 3: Iterative Self-Correction Loop ---")
        max_iterations = self.active_logic_parameters.get('max_correction_iterations', 5) # Erhöht für Trainingsmodus

        for i in range(max_iterations):
            logging.info(f"Correction Iteration {i + 1}/{max_iterations}...")
            was_graph_modified_in_iteration = False
            
            # NEU: problem_list initialisieren, damit sie immer definiert ist
            problem_list: Optional[List[Dict[str, Any]]] = None 

            # =================================================================
            # === TRAININGSMODUS: Direkte Korrektur mit Lösungsheft (Truth) ===
            # =================================================================
            if truth_data:
                logging.info("TRAINING MODE: Comparing current state against ground truth...")
                # Nutze die bestehende KPI-Funktion, um eine detaillierte Fehlerliste zu erhalten
                current_kpis = utils.calculate_raw_kpis_with_truth(
                    self._analysis_results, truth_data, self.knowledge_manager.get_all_aliases()
                )
                error_details = current_kpis.get("error_details", {})

                # Wenn keine Fehler mehr da sind, ist das Training für diesen Lauf beendet.
                if not any(error_details.values()):
                    logging.info("TRAINING MODE: No more deviations from truth found. Correction successful.")
                    break # Beendet die Korrekturphase

                # Korrigiere den ersten gefundenen Fehler (Priorität der Korrektur)
                # Diese Korrekturen finden direkt statt und benötigen kein LLM oder KB-Lookup
                if error_details.get("missed_elements"):
                    element_id_to_add = error_details["missed_elements"][0]
                    element_data = next((el for el in truth_data.get('elements', []) if el.get('id') == element_id_to_add), None)
                    if element_data:
                        logging.warning(f"TRAINING FIX: Element '{element_id_to_add}' fehlt. Füge es hinzu.")
                        self._analysis_results["elements"].append(element_data)
                        was_graph_modified_in_iteration = True
                elif error_details.get("hallucinated_elements"):
                    element_id_to_delete = error_details["hallucinated_elements"][0]
                    logging.warning(f"TRAINING FIX: Element '{element_id_to_delete}' ist eine Halluzination. Lösche es.")
                    correction = {"corrections": [{"action": "DELETE_ELEMENT", "element_id": element_id_to_delete}]}
                    was_graph_modified_in_iteration = self._apply_correction(correction)
                elif error_details.get("typing_errors"):
                    type_error = error_details["typing_errors"][0]
                    element_id = type_error.get("element_id")
                    correct_type = type_error.get("truth_type")
                    logging.warning(f"TRAINING FIX: Element '{element_id}' hat falschen Typ. Korrigiere zu '{correct_type}'.")
                    correction = {"corrections": [{"action": "CHANGE_TYPE", "element_id": element_id, "new_type": correct_type}]}
                    was_graph_modified_in_iteration = self._apply_correction(correction)
                elif error_details.get("missed_connections"):
                    conn_to_add = error_details["missed_connections"][0]
                    logging.warning(f"TRAINING FIX: Verbindung '{conn_to_add[0]}' -> '{conn_to_add[1]}' fehlt. Füge sie hinzu.")
                    correction = {"corrections": [{"action": "ADD_CONNECTION", "from_id": conn_to_add[0], "to_id": conn_to_add[1]}]}
                    was_graph_modified_in_iteration = self._apply_correction(correction)
                else:
                    logging.info("TRAINING MODE: No more easily fixable errors found via direct comparison.")
                    break # Keine einfach fixbaren Fehler mehr, beende den Trainingsmodus für diese Iteration

            # =================================================================
            # === KLAUSURMODUS: Interne Logik-Prüfung (wenn keine Truth da ist) ===
            # =================================================================
            else: # Dieser Block wird nur ausgeführt, wenn KEINE truth_data vorhanden ist
                internal_kpis = utils.calculate_connectivity_kpis(self._analysis_results["elements"], self._analysis_results["connections"])
                problem_list = self._identify_problems_from_kpis(internal_kpis)

                if not problem_list:
                    logging.info("No significant internal problems found. Concluding correction phase.")
                    break # Beendet die Korrekturphase, wenn keine Probleme mehr gefunden werden
                
                # Hier bleibt die alte Logik für die visuelle und LLM-basierte Korrektur
                problem_to_fix = problem_list[0] # problem_list ist hier definiert
                logging.warning(f"Highest priority internal issue: '{problem_to_fix['description']}'")

                # VERSUCH 1: VISUELLE KORREKTUR
                if problem_to_fix.get("type") == "TYPING_ERROR" and self.current_image_path:
                    element_id = problem_to_fix.get("context", {}).get("element_id")
                    bbox = self._find_bbox_for_component(element_id)
                    if bbox:
                        snippet_path = utils.crop_image_for_correction(self.current_image_path, bbox, 0.05)
                        if snippet_path:
                            visually_verified_type = self.knowledge_manager.find_symbol_type_by_visual_similarity(snippet_path)
                            if visually_verified_type and visually_verified_type != problem_to_fix.get("context", {}).get("ai_type"):
                                logging.info(f"VISUAL FIX: Correcting type for '{element_id}' to '{visually_verified_type}' based on visual match.")
                                correction_action = {"corrections": [{"action": "CHANGE_TYPE", "element_id": element_id, "new_type": visually_verified_type}]}
                                was_graph_modified_in_iteration = self._apply_correction(correction_action)
                            if os.path.exists(snippet_path):
                                os.remove(snippet_path)
                
                # VERSUCH 2: TEXTBASIERTE/LLM-KORREKTUR (falls visuell nicht erfolgreich oder nicht anwendbar)
                if not was_graph_modified_in_iteration:
                    logging.warning(f"EXAM MODE: Attempting to fix internal issue: '{problem_to_fix['description']}' with LLM/KB.")
                    solution = self.knowledge_manager.find_solution_for_problem(problem_to_fix)
                    if solution:
                        was_graph_modified_in_iteration = self._apply_correction(solution)
                    else:
                        correction_response, snippet_path = self._trigger_llm_correction(problem_to_fix)
                        if correction_response and isinstance(correction_response, dict):
                            was_graph_modified_in_iteration = self._apply_correction(correction_response)
                            if was_graph_modified_in_iteration:
                                problem_to_learn = problem_to_fix.copy()
                                # Sicherstellen, dass snippet_path ein String ist, wenn es an das Lernen übergeben wird
                                if snippet_path: # Füge nur hinzu, wenn ein Snippet erstellt wurde
                                    problem_to_learn["problem_snippet_path"] = str(snippet_path)
                                self.knowledge_manager.learn_from_correction(problem_to_learn, cast(Dict, correction_response))
                        if snippet_path and os.path.exists(snippet_path):
                            os.remove(snippet_path)

            # Überprüfung am Ende jeder Iteration
            if not was_graph_modified_in_iteration:
                logging.info("No corrections were applied in this iteration. Stopping loop.")
                break # Beendet die Korrekturschleife, wenn keine Änderungen vorgenommen wurden

        logging.info("Phase 3 completed.")

    def _identify_uncertain_zones(self) -> List[Dict[str, float]]:
        """
        Identifiziert "unsichere" Zonen basierend auf den Analyseergebnissen.
        Eine einfache Heuristik: Zonen um kleine, unverbundene Subgraphen.
        """
        logging.info("Identifiziere unsichere Zonen für adaptive Re-Analyse...")
        G = nx.Graph() # Undirected graph to find connected components
        elements_map = {el['id']: el for el in self._analysis_results.get('elements', []) if 'id' in el}
        G.add_nodes_from(elements_map.keys())

        for conn in self._analysis_results.get('connections', []):
            if conn.get('from_id') in G and conn.get('to_id') in G:
                G.add_edge(conn['from_id'], conn['to_id'])

        if not G.nodes:
            return []

        # Finde alle verbundenen Komponenten (Subgraphen)
        components = list(nx.connected_components(G))
        if not components:
            return []
            
        # Finde die größte Komponente
        largest_component = max(components, key=len)
        
        uncertain_zones_bboxes = []
        # Betrachte alle Komponenten außer der größten als "unsicher"
        for component in components:
            if component is not largest_component:
                for element_id in component:
                    element = elements_map.get(element_id)
                    if element and 'bbox' in element and isinstance(element['bbox'], dict):
                        uncertain_zones_bboxes.append(cast(Dict[str, float], element['bbox']))
        
        if not uncertain_zones_bboxes:
            logging.info("Keine spezifisch unsicheren Zonen gefunden (nur ein großer Graph).")
            return []

        # Verschmelze die Bounding Boxes der unsicheren Elemente zu größeren Zonen
        merged_zones = utils.merge_overlapping_boxes(uncertain_zones_bboxes)
        logging.info(f"{len(merged_zones)} unsichere Zonen für Re-Analyse identifiziert.")
        return merged_zones

    def _run_phase_3_5_adaptive_reanalysis(self, current_score: float):
        """
        Prüft die Gesamtqualität des Graphen. Ist sie zu niedrig, werden unsichere
        Zonen identifiziert und einer fokussierten, erneuten Analyse unterzogen.
        """
        logging.info("--- Running Phase 3.5: Adaptive Re-Analysis ---")
        
        # 1. Prüfe, ob eine Re-Analyse überhaupt nötig ist
        score_threshold = self.active_logic_parameters.get('adaptive_reanalysis_score_threshold', 65.0)
        current_score = self._analysis_results.get('quality_score', 100.0)

        if current_score >= score_threshold:
            logging.info(f"Qualitätsscore ({current_score:.2f}) ist über dem Schwellenwert ({score_threshold}). Keine adaptive Re-Analyse notwendig.")
            return

        logging.warning(f"Qualitätsscore ({current_score:.2f}) liegt unter dem Schwellenwert ({score_threshold}). Starte adaptive Re-Analyse.")

    def _run_critic_agent(self) -> Optional[str]:
        """Führt den Kritiker-Agenten aus, um die aktuelle Analyse zu bewerten."""
        logging.info("Kritiker-Agent prüft die aktuelle Analyse...")
        critic_prompt = self.config.get('prompts', {}).get('critic_user_prompt')
        if not critic_prompt or not self.current_image_path:
            return "PERFEKT" # Fallback, um Schleife zu beenden

        # Der Kritiker bekommt das Originalbild und das aktuelle JSON-Ergebnis
        context_for_critic = json.dumps(self._analysis_results, indent=2)
        full_critic_prompt = f"{critic_prompt}\n\n**AKTUELLE ANALYSE (JSON):**\n{context_for_critic}"

        model_info = self.model_strategy.get('correction_model', self.model_strategy.get('detail_model'))
        if not model_info:
            return "PERFEKT"

        feedback = self.llm_handler.call_llm(
            model_info,
            self.config.get('prompts', {}).get('general_system_prompt', ''),
            full_critic_prompt,
            self.current_image_path
        )
        return str(feedback) if feedback else "PERFEKT"

    def _run_analyst_critic_loop(self, truth_data: Optional[Dict] = None):
        """Implementiert die iterative Analyst-Kritiker-Schleife."""
        max_iterations = self.active_logic_parameters.get('max_correction_iterations', 3)

        for i in range(max_iterations):
            logging.info(f"Analyst-Kritiker-Iteration {i + 1}/{max_iterations}...")
            
            # Wenn Truth-Daten vorhanden sind, wird der Trainingsmodus priorisiert
            if truth_data:
                logging.info("TRAINING MODE: Führe direkte Korrektur mit Truth-Daten durch.")
                # (Hier bleibt die Logik aus der alten _run_phase_3_self_correction für den Trainingsmodus)
                current_kpis = utils.calculate_raw_kpis_with_truth(self._analysis_results, truth_data, self.knowledge_manager.get_all_aliases())
                error_details = current_kpis.get("error_details", {})
                if not any(error_details.values()):
                    logging.info("TRAINING MODE: Perfekte Übereinstimmung mit Truth-Daten erreicht.")
                    return # Perfektes Ergebnis, Schleife beenden
                
                # Führe EINE Korrektur durch
                # (Hier könnte man die Korrekturlogik aus der alten Funktion einfügen)
                logging.warning(f"TRAINING FIX: Korrigiere Abweichung: {list(error_details.keys())[0]}")
                # ... (Hier würde die Logik zum Hinzufügen/Löschen basierend auf error_details stehen) ...
                continue # Nächste Iteration im Trainingsmodus

            # KLAUSURMODUS: Der Kritiker-Agent wird aktiv
            feedback = self._run_critic_agent()

            if feedback and "PERFEKT" in feedback.upper():
                logging.info("Kritiker ist zufrieden. Analyse abgeschlossen.")
                return

            logging.warning(f"Kritiker fand Fehler: {feedback}")
            
            # Hier nutzen wir die bestehende Korrektur-Logik, um das Feedback des Kritikers umzusetzen
            problem_to_fix = {
                "type": "CRITIC_FEEDBACK",
                "description": f"Der Kritiker-Agent hat folgende Fehler gefunden, bitte beheben: {feedback}",
                "location_bbox": None # Wir haben keine spezifische BBox für allgemeines Feedback
            }
            
            correction_response, _ = self._trigger_llm_correction(problem_to_fix)
            if correction_response and self._apply_correction(cast(Dict, correction_response)):
                logging.info("Korrektur basierend auf Kritiker-Feedback angewendet.")
            else:
                logging.warning("Konnte Kritiker-Feedback nicht anwenden. Breche Schleife ab.")
                return
            
        # 2. Identifiziere die unsicheren Zonen
        uncertain_zones = self._identify_uncertain_zones()
        if not uncertain_zones or self.current_image_path is None:
            logging.info("Keine unsicheren Zonen gefunden oder Bildpfad fehlt. Beende adaptive Re-Analyse.")
            return

        # 3. Führe eine neue, feingranulare Analyse nur in diesen Zonen durch
        temp_dir_path = Path(os.path.dirname(self.current_image_path)) / "temp_reanalysis_tiles"
        temp_dir_path.mkdir(exist_ok=True, parents=True)

        reanalysis_tiles = utils.generate_raster_grid_in_zones(
            image_path=self.current_image_path,
            zones=uncertain_zones,
            tile_size=self.active_logic_parameters.get('adaptive_reanalysis_tile_size', 768),
            overlap=self.active_logic_parameters.get('adaptive_reanalysis_overlap', 96),
            output_folder=temp_dir_path
        )

        if not reanalysis_tiles:
            logging.info("In unsicheren Zonen konnten keine neuen Kacheln generiert werden.")
            if temp_dir_path.exists(): shutil.rmtree(temp_dir_path)
            return

        # Führe LLM-Aufrufe für die neuen Kacheln durch (ähnlich wie in Phase 2)
        # HINWEIS: Hier wird die Logik aus Phase 2b wiederverwendet
        detail_model_info = self.model_strategy.get('detail_model')
        system_prompt_fine = self.config.get('prompts', {}).get('general_system_prompt')
        raster_analysis_user_prompt_template = self.config.get('prompts', {}).get('raster_analysis_user_prompt_template')
        
        if not detail_model_info:
            logging.error("Kein 'detail_model' in der Strategie für adaptive Re-Analyse definiert. Breche ab.")
            if temp_dir_path.exists(): shutil.rmtree(temp_dir_path)
            return

        # Prompts vorbereiten (kopiert aus Phase 2)
        known_types_json = json.dumps({"known_types": self.knowledge_manager.get_known_types(), "type_aliases": self.knowledge_manager.get_all_aliases()}, indent=2)
        formatted_raster_analysis_user_prompt = raster_analysis_user_prompt_template.format(known_types_json=known_types_json, few_shot_prompt_part="")


        new_raw_results: List[TileResult] = []
        with ThreadPoolExecutor(max_workers=self.active_logic_parameters.get('llm_timeout_executor_workers', 1)) as executor:
            future_to_tile = {
                executor.submit(self.llm_handler.call_llm, detail_model_info, system_prompt_fine, formatted_raster_analysis_user_prompt, tile['path'], True, self.active_logic_parameters.get('llm_default_timeout', 120), ["elements", "connections"]): tile
                for tile in reanalysis_tiles
            }
            for future in as_completed(future_to_tile):
                tile = future_to_tile[future]
                try:
                    response_data = future.result()
                    if response_data:
                        new_raw_results.append(TileResult(tile_coords=tile['coords'], tile_width=tile['tile_width'], tile_height=tile['tile_height'], data=cast(utils.TileData, response_data)))
                except Exception as exc:
                    logging.error(f"Fehler bei der Re-Analyse für Kachel '{tile['path']}': {exc}", exc_info=True)

        if not new_raw_results:
            logging.warning("Re-Analyse der unsicheren Zonen ergab keine neuen Ergebnisse.")
            if temp_dir_path.exists(): shutil.rmtree(temp_dir_path)
            return

        # 4. Synthetisiere die NEUEN Ergebnisse mit den ALTEN
        logging.info("Führe Re-Synthese durch, um neue Ergebnisse mit bestehendem Graphen zu verschmelzen...")
        
        img_width, img_height = Image.open(self.current_image_path).size

        # Erstelle ein "raw_result" aus den bereits existierenden Daten, damit der Synthesizer sie berücksichtigen kann
        existing_data_as_raw_result = TileResult(
            tile_coords=(0, 0),
            tile_width=img_width,
            tile_height=img_height,
            data={'elements': self._analysis_results['elements'], 'connections': self._analysis_results['connections']}
        )
        
        # Kombiniere alte und neue Ergebnisse für die finale Synthese
        combined_raw_results = [existing_data_as_raw_result] + new_raw_results
        
        synthesizer_config = utils.SynthesizerConfig(
            iou_match_threshold=self.active_logic_parameters.get('iou_match_threshold', 0.5),
            border_coords_tolerance_factor_x=self.active_logic_parameters.get('border_coords_tolerance_factor_x', 0.015),
            border_coords_tolerance_factor_y=self.active_logic_parameters.get('border_coords_tolerance_factor_y', 0.015)
        )
        final_synthesizer = utils.GraphSynthesizer(combined_raw_results, img_width, img_height, config=synthesizer_config)
        merged_analysis_data = final_synthesizer.synthesize()

        logging.info(f"Re-Analyse abgeschlossen. Elementanzahl geändert von {len(self._analysis_results['elements'])} auf {len(merged_analysis_data['elements'])}.")
        
        # Überschreibe die alten Analyseergebnisse mit den neuen, gemergten Ergebnissen
        self._analysis_results['elements'] = merged_analysis_data['elements']
        self._analysis_results['connections'] = merged_analysis_data['connections']
        
        # Räume das temporäre Verzeichnis auf
        if temp_dir_path.exists():
            shutil.rmtree(temp_dir_path)

    def _identify_problems_from_kpis(self, kpis: Dict) -> List[Dict]:
        """
        Analyzes KPIs to create a list of specific, actionable problems.
        Each problem now includes the 'location_bbox' for visual feedback.
        """
        problems: List[Dict] = []
        
        # Problem 1: Isolated Elements (Elemente, die nirgends verbunden sind)
        if kpis.get("isolated_elements_count", 0) > 0:
            logging.info("Identifiziere isolierte Elemente für mögliche Korrektur.")
            isolated_node_distance_threshold = self.active_logic_parameters.get('graph_completion_isolated_node_distance', 0.05)

            # Explizite Typisierung für das networkx.Graph Objekt, um Pylance zu beruhigen
            temp_G_graph: nx.Graph = nx.Graph() 
            elements_map_for_graph = {el.get('id'): el for el in self._analysis_results["elements"] if el.get('id')}
            temp_G_graph.add_nodes_from(elements_map_for_graph.keys())
            
            for conn in self._analysis_results["connections"]:
                from_id = conn.get('from_id')
                to_id = conn.get('to_id')
                # Sicherstellen, dass from_id und to_id strings sind und als Knoten existieren
                if isinstance(from_id, str) and isinstance(to_id, str) and from_id in temp_G_graph and to_id in temp_G_graph:
                    temp_G_graph.add_edge(from_id, to_id)
            
            for el_id, element_data in elements_map_for_graph.items():
                if element_data.get("type") == "Process Port":
                    continue

                if isinstance(el_id, str) and el_id in temp_G_graph and temp_G_graph.degree[el_id] == 0: # type: ignore [reportAttributeAccessIssue]
                    bbox_el = element_data.get('bbox')
                    
                    if not bbox_el:
                        logging.warning(f"Isoliertes Element '{el_id}' ohne BBox gefunden. Kann kein visuelles Problem melden.")
                        continue 

                    is_truly_isolated = True 
                    for other_el_id, other_element_data in elements_map_for_graph.items():
                        if other_el_id == el_id: continue 
                        # Wenn das andere Element auch isoliert ist, ignorieren wir es hier,
                        # da es später selbst als Problem gemeldet wird.
                        if isinstance(other_el_id, str) and other_el_id in temp_G_graph and temp_G_graph.degree[other_el_id] == 0: continue # type: ignore [reportAttributeAccessIssue]

                        bbox_other = other_element_data.get('bbox')
                        if not bbox_other: continue

                        bbox_el_dict = cast(Dict[str, float], bbox_el)
                        bbox_other_dict = cast(Dict[str, float], bbox_other)

                        center_x_el = bbox_el_dict['x'] + bbox_el_dict['width'] / 2
                        center_y_el = bbox_el_dict['y'] + bbox_el_dict['height'] / 2
                        center_x_other = bbox_other_dict['x'] + bbox_other_dict['width'] / 2
                        center_y_other = bbox_other_dict['y'] + bbox_other_dict['height'] / 2
                        
                        distance = ((center_x_el - center_x_other)**2 + (center_y_el - center_y_other)**2)**0.5
                        
                        if distance < isolated_node_distance_threshold:
                            is_truly_isolated = False
                            logging.debug(f"Element '{el_id}' ist physisch nah an '{other_el_id}', gilt nicht als 'echt isoliert'.")
                            break

                    if is_truly_isolated:
                        problems.append({
                            "type": "ISOLATED_ELEMENT",
                            "description": f"The element '{element_data.get('label', el_id)}' (ID: {el_id}) is not connected to any other element and seems truly isolated.",
                            "context": {"element_label": element_data.get('label', el_id), "element_id": el_id, "element_type": element_data.get('type')},
                            "location_bbox": cast(Dict[str, float], bbox_el) 
                        })
        
        # Problem 2: Typing Errors
        if "final_kpis" in self._analysis_results and "error_details" in self._analysis_results["final_kpis"]:
            typing_errors = self._analysis_results["final_kpis"]["error_details"].get("typing_errors", [])
            if typing_errors:
                logging.info("Identifiziere Typisierungsfehler für mögliche Korrektur.")
                for error_detail in typing_errors:
                    element_id = error_detail.get("element_id")
                    if isinstance(element_id, str): # Sicherstellen, dass element_id ein String ist
                        element = next((el for el in self._analysis_results["elements"] if el.get("id") == element_id), None)
                        if element and element.get("bbox"):
                            problems.append({
                                "type": "TYPING_ERROR",
                                "description": f"The element '{element.get('label', element_id)}' (ID: {element_id}) was identified as type '{error_detail.get('ai_type')}' but should be '{error_detail.get('truth_type')}'.",
                                "context": {"element_label": element.get('label', element_id), "element_id": element_id, "ai_type": error_detail.get('ai_type'), "truth_type": error_detail.get('truth_type')},
                                "location_bbox": cast(Dict[str, float], element.get("bbox")) 
                            })
                        else:
                            logging.warning(f"Typisierungsfehler für '{element_id}' gemeldet, aber Element oder BBox nicht in aktuellen Analyseergebnissen gefunden.")
                    else:
                        logging.warning(f"Ungültige Element-ID im Typisierungsfehler: {element_id}. Überspringe.")

        # Problem 3: Hallucinated Elements
        if "final_kpis" in self._analysis_results and "error_details" in self._analysis_results["final_kpis"]:
            hallucinated_elements = self._analysis_results["final_kpis"]["error_details"].get("hallucinated_elements", [])
            if hallucinated_elements:
                logging.info("Identifiziere halluzinierte Elemente für mögliche Korrektur.")
                for element_id in hallucinated_elements:
                    if isinstance(element_id, str): # Sicherstellen, dass element_id ein String ist
                        element = next((el for el in self._analysis_results["elements"] if el.get("id") == element_id), None)
                        if element and element.get("bbox"):
                            problems.append({
                                "type": "HALLUCINATED_ELEMENT",
                                "description": f"The element '{element.get('label', element_id)}' (ID: {element_id}) was identified by the AI but does not exist in the ground truth. It should be deleted.",
                                "context": {"element_label": element.get('label', element_id), "element_id": element_id, "element_type": element.get('type')},
                                "location_bbox": cast(Dict[str, float], element.get("bbox")) 
                            })
                        else:
                            logging.warning(f"Halluziniertes Element '{element_id}' gemeldet, aber Element oder BBox nicht in aktuellen Analyseergebnissen gefunden.")
                    else:
                        logging.warning(f"Ungültige Element-ID im halluzinierten Element: {element_id}. Überspringe.")

        priority_order = {"TYPING_ERROR": 1, "HALLUCINATED_ELEMENT": 2, "ISOLATED_ELEMENT": 3}
        problems.sort(key=lambda p: priority_order.get(p.get("type", "UNKNOWN"), 99))

        return problems