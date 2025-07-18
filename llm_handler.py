# file: llm_handler.py
# 📦 Standardbibliotheken
import os
import re
import json
import time
import base64
import logging
import hashlib
import mimetypes # <--- HINZUGEFÜGT/SICHERGESTELLT
import traceback
from pathlib import Path
from typing import Dict, Any, Optional, List, cast, Union, Set
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from io import BytesIO
import uuid # <--- HINZUGEFÜGT/SICHERGESTELLT

# 🧪 Drittanbieter-Bibliotheken (External Libraries)
import diskcache
import vertexai
from vertexai.generative_models import HarmCategory, HarmBlockThreshold
from vertexai.language_models import TextEmbeddingModel
from vertexai.generative_models import GenerativeModel, Part
# HINZUGEFÜGT: Für das direkte MultiModalEmbeddingModel
from vertexai.vision_models import MultiModalEmbeddingModel, Image as VertexImage
from PIL import Image # <--- Dies bleibt die PIL.Image Klasse
# (Der Kommentar zum nicht verwendeten Import kann bleiben oder entfernt werden)

logger = logging.getLogger(__name__)

class LLM_Handler:
    """
    Handles interactions with different LLM providers on Vertex AI.
    In dieser Version ist die Claude-Funktionalität deaktiviert.
    """

    def __init__(self, project_id: str, default_location: str, config: Dict[str, Any]):
        self.project_id = project_id
        self.default_location = default_location
        self.config = config
        
        # Initialisiere Vertex AI SDK einmal
        vertexai.init(project=self.project_id, location=self.default_location)

        self.gemini_clients: Dict[str, GenerativeModel] = {}
        self.text_embedding_model: Optional[TextEmbeddingModel] = None
        # Das MultiModal Embedding Model wird on-demand geladen und benötigt hier keine Instanz-Variable.
        self._initialized_locations: Set[str] = set()

        self._load_models() # Lade Modelle während der Initialisierung

        cache_dir_name = self.config.get('paths', {}).get('llm_cache_dir', '.pni_analyzer_cache')
        cache_path = Path(str(cache_dir_name)) # <- NEU: Expliziter Cast zu str, um Pylance zu beruhigen

        # Überprüfe, ob der Cache bei jedem Start geleert werden soll
        auto_clear_cache = self.config.get('logic_parameters', {}).get('auto_clear_llm_cache_on_start', False)
        if auto_clear_cache and cache_path.exists():
            try:
                for item in cache_path.iterdir():
                    if item.is_file():
                        item.unlink()
                    elif item.is_dir():
                        import shutil # Importiere shutil, falls nicht schon oben
                        shutil.rmtree(item)
                logging.info(f"Automatically cleared LLM disk cache at: {cache_path}")
            except Exception as e:
                logging.error(f"Error automatically clearing LLM disk cache at {cache_path}: {e}")
       
        cache_size_gb = self.config.get('logic_parameters', {}).get('llm_cache_size_gb', 1)
        # NEU: Expliziter Cast zu int für size_limit, um Pylance zu beruhigen
        self.disk_cache = diskcache.Cache(cache_path, size_limit=int(cache_size_gb * 1024 * 1024 * 1024))
        logging.info(f"Persistent cache initialized at: {cache_path}")

        max_workers_timeout = self.config.get('logic_parameters', {}).get('llm_timeout_executor_workers', 1)
        self.timeout_executor = ThreadPoolExecutor(max_workers=max_workers_timeout)

    def _try_llm_json_fix(self, malformed_text: str) -> Optional[str]:
        """
        Sends malformed text back to a fast LLM to attempt a syntax fix.
        """
        logging.info("Attempting to fix malformed JSON with a second LLM call...")
        
        fixer_prompt_template = self.config.get('prompts', {}).get('json_fixer_user_prompt')
        if not fixer_prompt_template:
            logging.error("json_fixer_user_prompt not found in config.yaml. Cannot attempt fix.")
            return None
            
        formatted_prompt = fixer_prompt_template.format(malformed_text=malformed_text)
        
        # Wähle ein schnelles, günstiges Modell für diese simple Aufgabe
        # Wir nehmen hier an, dass 'gemini-2.5-flash' in der config definiert ist.
        fixer_model_info = next((m for m in self.config.get('models', {}).values() if m['id'] == 'gemini-2.5-flash'), None)
        
        if not fixer_model_info:
            logging.error("Could not find 'gemini-2.5-flash' model in config for JSON fix. Aborting.")
            return None
            
        try:
            # Direkter Aufruf, ohne den vollen `call_llm`-Wrapper, um Zyklen zu vermeiden
            model = self.gemini_clients.get(fixer_model_info['id'])
            if not model:
                logging.error("Fixer model not loaded.")
                return None
                
            response = model.generate_content(
                formatted_prompt,
                generation_config={"response_mime_type": "application/json"}
            )
            logging.info("LLM-based JSON fix successful.")
            return response.text
        except Exception as e:
            logging.error(f"Error during LLM-based JSON fix: {e}")
            return None

    def _load_models(self):
        """
        Loads the necessary Vertex AI models from the configuration.
        The MultiModal Embedding Model is loaded on-demand in get_image_embedding.
        """
        for model_name_display, model_info in self.config.get("models", {}).items():
            model_id = model_info.get("id")
            if not model_id:
                logging.warning(f"Modell '{model_name_display}' in Konfiguration hat keine ID. Wird übersprungen.")
                continue

            # Lade nur Generative Models (Gemini-Modelle für Text/Vision)
            if model_info.get('access_method') == 'gemini':
                try:
                    if model_id not in self.gemini_clients:
                        self.gemini_clients[model_id] = GenerativeModel(model_id)
                        logging.info(f"Successfully loaded Gemini model: {model_id}")
                except Exception as e:
                    logging.error(f"Error loading Gemini model {model_id}: {e}", exc_info=True)

    def get_image_embedding(self, image_input: Union[str, Image.Image]) -> Optional[List[float]]:
        """
        Erstellt einen visuellen Vektor (Embedding) für ein gegebenes Bild.
        Akzeptiert entweder einen Bildpfad (str) oder ein PIL.Image-Objekt.
        Verwendet das Vertex AI MultiModalEmbeddingModel direkt.
        """
        log_identifier = "PIL.Image-Objekt"
        temp_path: Optional[Path] = None # Initialisiere temp_path
        
        try:
            image_bytes: bytes
            if isinstance(image_input, str):
                log_identifier = image_input
                if not os.path.exists(image_input):
                    logging.error(f"Bildpfad existiert nicht für Embedding: {image_input}")
                    return None
                with open(image_input, "rb") as f:
                    image_bytes = f.read()
            elif isinstance(image_input, Image.Image):
                # Für PIL.Image-Objekte müssen wir sie temporär speichern,
                # damit das MultiModalEmbeddingModel sie lesen kann.
                temp_dir = Path(self.config.get('paths', {}).get('temp_symbol_dir', 'temp_symbols_for_embeddings'))
                temp_dir.mkdir(exist_ok=True, parents=True)
                temp_path = temp_dir / f"temp_embedding_img_{uuid.uuid4().hex[:8]}.png"
                image_input.save(temp_path)
                with open(temp_path, "rb") as f:
                    image_bytes = f.read()
                log_identifier = str(temp_path) # Verwende den temporären Pfad als Identifikator
            else:
                logging.error(f"Ungültiger Input-Typ für get_image_embedding: {type(image_input)}. Erwartet str oder PIL.Image.")
                return None

            # Das Modell wird hier bei Bedarf geladen (nicht global im init)
            # Stellen Sie sicher, dass 'multimodalembedding@001' in Ihrer config.yaml korrekt definiert ist
            model = MultiModalEmbeddingModel.from_pretrained("multimodalembedding@001")
            vertex_image_obj = VertexImage(image_bytes)

            embeddings = model.get_embeddings(
                image=vertex_image_obj,
                contextual_text=None,
                dimension=1408 # Die Standard- und empfohlene Dimension
            )
            
            image_embedding = embeddings.image_embedding
            if image_embedding:
                logging.info(f"Successfully generated direct image embedding for '{log_identifier}' with dimension: {len(image_embedding)}.")
                return image_embedding
            else:
                logging.warning(f"API returned embeddings object for '{log_identifier}', but image_embedding was empty.")
                return None

        except Exception as e:
            logging.error(f"Error during direct image embedding generation for '{log_identifier}': {e}", exc_info=True)
            return None
        finally:
            if temp_path and temp_path.exists():
                try:
                    os.remove(temp_path)
                    # Versuche, das temp_dir zu entfernen, wenn es leer ist
                    if not any(temp_path.parent.iterdir()):
                        temp_path.parent.rmdir()
                except OSError as e:
                    logging.warning(f"Konnte temporäre Datei oder Verzeichnis {temp_path} nicht aufräumen: {e}")
     
    def _get_text_embedding_model(self) -> TextEmbeddingModel:
        """Initialisiert das Text Embedding-Modell bei der ersten Verwendung."""
        if not self.text_embedding_model:
            logging.info("Initializing Text Embedding Model (text-embedding-004)...")
            self.text_embedding_model = TextEmbeddingModel.from_pretrained("text-embedding-004")
        return self.text_embedding_model

    def get_text_embedding(self, text: str) -> Optional[List[float]]:
        """Erstellt einen semantischen Vektor (Embedding) für einen gegebenen Text."""
        try:
            model = self._get_text_embedding_model()
            if not model:
                logging.warning("TextEmbeddingModel nicht geladen. Kann kein Text-Embedding generieren.")
                return None
            embeddings = model.get_embeddings([text])
            return embeddings[0].values
        except Exception as e:
            logging.error(f"Error creating text embedding for text '{text[:30]}...': {e}")
            return None

    def _encode_image(self, image_path: str) -> Optional[str]:
        """Encodes an image to a base64 string."""
        try:
            with open(image_path, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        except IOError as e:
            logging.error(f"Error encoding image at {image_path}: {e}")
            return None

    def _generate_cache_key(self, model_id: str, system_prompt: str, user_prompt: str, image_path: Optional[str]) -> str:
        """Generates a unique hash key for a given request."""
        hasher = hashlib.sha256()
        image_hash = "no_image"
        if image_path and os.path.exists(image_path):
            with open(image_path, "rb") as f:
                image_hash = hashlib.sha256(f.read()).hexdigest()
        combined_input = f"{model_id}-{system_prompt}-{user_prompt}-{image_hash}"
        hasher.update(combined_input.encode('utf-8'))
        return hasher.hexdigest()
    
    def _parse_and_validate_json_response(self, response_text: str, image_path_for_rescue: Optional[str], expected_json_keys: Optional[List[str]] = None) -> Optional[Union[Dict[str, Any], List[Any]]]:
        """
        Parses a raw LLM response text into JSON, with aggressive, regex-based data rescue.
        This version can salvage data even from malformed JSON-like strings.
        """
        # 1. Erster Versuch: Direkte, saubere JSON-Konvertierung
        try:
            # Versuche, das JSON aus einem Markdown-Block zu extrahieren
            match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
            if match:
                json_str = match.group(1)
            else:
                # Finde das erste Vorkommen von '{' oder '[' als Startpunkt
                first_curly = response_text.find('{')
                first_square = response_text.find('[')
                
                if first_curly == -1 and first_square == -1:
                    json_str = response_text # Kein JSON-Start gefunden, übergebe gesamten Text an Rescue
                else:
                    start_index = min(first_curly, first_square) if first_curly != -1 and first_square != -1 else max(first_curly, first_square)
                    json_str = response_text[start_index:]
            
            parsed_json = json.loads(json_str)
            logging.info("Parser: Direkte JSON-Konvertierung erfolgreich.")
            # Hier könnte eine Validierung der Top-Level-Keys stattfinden, falls nötig
            return parsed_json
        except (json.JSONDecodeError, AttributeError):
            logging.warning("Parser: Direkte JSON-Konvertierung fehlgeschlagen. Starte Data-Rescue-Modus...")

        # 2. Zweiter Versuch: "Data Rescue" mit Regulären Ausdrücken
        rescued_symbols = []
        # Finde alle Blöcke, die wie ein Objekt aussehen (zwischen '{' und '}')
        object_candidates = re.findall(r'\{[^{}]*?\}', response_text.replace('\n', ''))
        
        for candidate in object_candidates:
            label = None
            x1, y1, x2, y2 = -1, -1, -1, -1

            try:
                # Erweitere Label-Schlüssel, um auch 'description', 'value', 'content', 'symbol_label' zu finden
                label_match = re.search(r'["\'](?:label|text|symbol_name|symbol_description|description|value|content|symbol_label|name)["\']\s*:\s*["\'](.*?)["\']', candidate)
                
                # Suche nach allen bekannten Variationen von Bounding-Box-Schlüsseln und -Formaten
                bbox_match_list = re.search(r'["\'](?:bbox|box_2d|bbox_2d|symbol_bbox|coordinates|box)["\']\s*:\s*\[\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*\]', candidate)
                bbox_match_dict = re.search(r'["\'](?:bbox|box_2d|bbox_2d|symbol_bbox|coordinates|box)["\']\s*:\s*\{\s*["\']x["\']:\s*([\d.]+)\s*,\s*["\']y["\']:\s*([\d.]+)\s*,\s*["\']width["\']:\s*([\d.]+)\s*,\s*["\']height["\']:\s*([\d.]+)\s*\}', candidate)

                if label_match and (bbox_match_list or bbox_match_dict):
                    label = label_match.group(1)
                    
                    if bbox_match_list:
                        x1, y1, x2, y2 = [float(v) for v in bbox_match_list.groups()]
                    elif bbox_match_dict:
                        x, y, w, h = [float(v) for v in bbox_match_dict.groups()]
                        x1, y1, x2, y2 = x, y, x+w, y+h

                    # Nur fortfahren, wenn BBox-Koordinaten erfolgreich geparst wurden
                    if x1 != -1: 
                        # Wenn wir den Bildpfad haben, können wir die Pixelwerte normalisieren
                        if image_path_for_rescue and Path(image_path_for_rescue).exists():
                            try:
                                # Import Image direkt aus PIL für diesen lokalen Kontext
                                from PIL import Image
                                with Image.open(image_path_for_rescue) as img:
                                    img_width, img_height = img.size
                                    if img_width > 0 and img_height > 0:
                                        # Annahme, dass die von Regex extrahierten Koordinaten normalisiert sind (0-1)
                                        width_norm = (x2 - x1) 
                                        height_norm = (y2 - y1)
                                        
                                        if width_norm > 0 and height_norm > 0 and width_norm <= 1.0 and height_norm <= 1.0:
                                            rescued_symbols.append({
                                                "label": label,
                                                "bbox": {
                                                    "x": x1, 
                                                    "y": y1, 
                                                    "width": width_norm,
                                                    "height": height_norm
                                                }
                                            })
                                        else: # Fallback, wenn normalisierte Werte ungültig erscheinen (dann sind es Pixel)
                                            width_norm_fallback = (x2 - x1) / img_width
                                            height_norm_fallback = (y2 - y1) / img_height
                                            if width_norm_fallback > 0 and height_norm_fallback > 0:
                                                rescued_symbols.append({
                                                    "label": label,
                                                    "bbox": {
                                                        "x": x1 / img_width,
                                                        "y": y1 / img_height,
                                                        "width": width_norm_fallback,
                                                        "height": height_norm_fallback
                                                    }
                                                })
                                            else:
                                                logging.warning(f"Rescued BBox for symbol {label} appears invalid after normalization attempt: {x1},{y1},{x2},{y2}. Skipping.")
                            except Exception as e: # Fängt Fehler beim Bildzugriff ab
                                logging.warning(f"Konnte Bildgröße für Data Rescue nicht lesen: {e}")
            except (ValueError, IndexError, AttributeError) as ve: # Fängt Parsing-Fehler innerhalb des 'try' ab
                logging.debug(f"Skipping candidate due to parsing error in Data Rescue: {ve}. Candidate: {candidate}")
                continue # Springt zum nächsten Kandidaten

        if rescued_symbols:
            logging.info(f"DATA RESCUE: Erfolgreich {len(rescued_symbols)} Symbole aus fehlerhaftem JSON gerettet.")
            # Gib die geretteten Daten im korrekten Format zurück
            return {"symbols": rescued_symbols}

        logging.error(f"Parser: Konnte auch mit Data Rescue keine validen Daten extrahieren. Antwort: '{response_text[:200]}...'")
        return None
        
    def call_llm(self, model_info: Dict[str, Any], system_prompt: str, user_prompt: str, image_path: Optional[str] = None, use_cache: bool = True, timeout: int = 120, expected_json_keys: Optional[List[str]] = None) -> Dict[str, Any] | List[Any] | None:
        model_id = model_info['id']
        access_method = model_info.get('access_method', 'gemini')
        location = model_info.get('location', self.default_location)
        cache_key = self._generate_cache_key(model_id, system_prompt, user_prompt, image_path)

        if use_cache and cache_key in self.disk_cache:
            logging.info(f"Returning cached response from DISK for model '{model_id}'.")
            return cast(Optional[Dict[str, Any]], self.disk_cache.get(cache_key))

        max_retries = self.config.get('logic_parameters', {}).get('llm_max_retries', 3)
        wait_time = self.config.get('logic_parameters', {}).get('llm_initial_wait_time', 2)
        
        default_config_timeout = self.config.get('logic_parameters', {}).get('llm_default_timeout', 120)
        timeout = timeout if timeout != 120 else default_config_timeout
        max_timeout_on_retry = self.config.get('logic_parameters', {}).get('llm_max_timeout_on_retry', 300)
 
        for i in range(max_retries):
            logging.info(f"Calling LLM API for model '{model_id}'... (Attempt {i + 1}/{max_retries})")
            
            try:
                if access_method == 'gemini':
                    model = self.gemini_clients.get(model_id)
                    if not model:
                        logging.error(f"Gemini model '{model_id}' not pre-loaded. Aborting call.")
                        return None
                    
                    prompt_parts = []
                    if system_prompt: prompt_parts.append(Part.from_text(system_prompt))
                    if image_path: 
                        image_data = self._encode_image(image_path)
                        if not image_data:
                            logging.error(f"Could not encode image {image_path} for LLM prompt.")
                            return None
                        prompt_parts.append(Part.from_data(data=base64.b64decode(image_data), mime_type=mimetypes.guess_type(image_path)[0] or 'application/octet-stream'))
                    prompt_parts.append(Part.from_text(user_prompt))

                    # Hier übergeben wir model_info, um die GenerationConfig Parameter zu nutzen
                    future = self.timeout_executor.submit(self._call_gemini, model_id, prompt_parts, model_info)
                elif access_method == 'vertex_ai_multimodal_embedding':
                    logging.error(f"Direct call_llm not supported for 'vertex_ai_multimodal_embedding' model type. Use get_image_embedding directly.")
                    return None
                else:
                    logging.error(f"Access method '{access_method}' is not supported.")
                    return None

                response_text = None
                try:
                    response_text = future.result(timeout=timeout)
                except TimeoutError as e:
                    logging.warning(f"LLM call for model '{model_id}' timed out after {timeout} seconds. (Attempt {i + 1}/{max_retries}). Retrying with increased timeout...")
                    timeout = min(timeout * 2, max_timeout_on_retry)
                    time.sleep(wait_time)
                    wait_time *= 2
                    continue
                except Exception as e:
                    logging.error(f"Error executing LLM call in separate thread: {e}", exc_info=True)
                    return None

                if not response_text:
                    logging.warning(f"LLM call for model '{model_id}' returned no text response.")
                    if i < max_retries - 1:
                        time.sleep(wait_time)
                        wait_time *= 2
                        continue
                    else:
                        return None
                
                parsed_json = self._parse_and_validate_json_response(response_text, image_path, expected_json_keys)

                # Wenn das Parsen fehlschlägt, versuche den LLM-Fixer
                if parsed_json is None:
                    fixed_json_text = self._try_llm_json_fix(response_text)
                    if fixed_json_text:
                        # Versuche erneut, die vom LLM korrigierte Antwort zu parsen
                        parsed_json = self._parse_and_validate_json_response(fixed_json_text, image_path, expected_json_keys)

                if parsed_json is None:
                    # Nur wenn auch der Fixer scheitert, machen wir einen Retry oder geben auf
                    if i < max_retries - 1:
                        logging.warning(f"JSON parsing and LLM-fix failed for model '{model_id}'. (Attempt {i + 1}/{max_retries}). Retrying...")
                        time.sleep(wait_time)
                        wait_time *= 2
                        continue
                    else:
                        logging.error(f"JSON parsing failed after all retries and fix attempts for model '{model_id}'.")
                        return None

                if use_cache:
                    self.disk_cache.set(cache_key, parsed_json)
                
                return parsed_json

            except Exception as e:
                error_message = str(e).lower()
                if "503" in error_message or "unavailable" in error_message or "end of tcp stream" in error_message or "connection" in error_message:
                    logging.warning(f"Attempt {i + 1} failed with a network error: {e}. Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                    wait_time *= 2  
                    continue 
                else:
                    logging.error(f"A non-retriable error occurred during LLM API call for model '{model_id}': {e}", exc_info=True)
                    return None
        
        logging.error(f"LLM call failed after {max_retries} attempts for model '{model_id}'.")
        return None
    
    def _call_gemini(self, model_id: str, prompt_parts: List[Part], model_info: Dict[str, Any]) -> Optional[str]: # <- Ändere den Rückgabetyp
        """
        Handles the specific API call structure for Google Gemini models,
        now strictly enforcing JSON output via generation_config and using parameters from config.
        """
        model = self.gemini_clients.get(model_id)
        if not model:
            raise ValueError(f"Gemini model '{model_id}' not found in loaded clients.")

        # Die GenerationConfig wird aus den model_info-Daten der Konfiguration gelesen.
        # 'response_mime_type' wird hier explizit gesetzt, um JSON zu erzwingen.
        generation_config = {
            "response_mime_type": "application/json",
            "temperature": model_info.get("generation_config", {}).get("temperature", 0.15), # Default 0.15
            "top_p": model_info.get("generation_config", {}).get("top_p", 0.95), # Default 0.95
            "top_k": model_info.get("generation_config", {}).get("top_k", -1), # Default -1 (unbegrenzt)
            "max_output_tokens": model_info.get("generation_config", {}).get("max_output_tokens", 8192) # Default 8192
        }
        # Entferne top_k, wenn es -1 ist (bedeutet "nicht setzen" für die API)
        if generation_config.get("top_k", -1) == -1: # Verwende .get() um sicher auf top_k zuzugreifen
            generation_config.pop("top_k", None) # Verwende pop mit None, um den Schlüssel sicher zu entfernen

        # Filtern der generation_config, um nur relevante Parameter zu übergeben
        generation_config = {k: v for k, v in generation_config.items() if v is not None}

        # NEU/GEÄNDERT: Safety Settings für technische Diagramme deaktivieren, um Blockaden zu vermeiden
        safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
        # Wichtig: Imports hinzufügen!
        # from vertexai.generative_models import HarmCategory, HarmBlockThreshold # Oben bei den Imports

        try:
            response = model.generate_content(
                prompt_parts, # type: ignore
                generation_config=generation_config,
                safety_settings=safety_settings # NEU: Safety settings anwenden
            )
            # Prüfe, ob response.text verfügbar ist, bevor es zurückgegeben wird
            if response.text:
                return response.text
            else:
                # Logge den Grund für das Fehlen des Textes, wahrscheinlich Safety-Filter oder Max-Tokens
                finish_reason = response.candidates[0].finish_reason if response.candidates else 'N/A'
                safety_ratings = response.candidates[0].safety_ratings if response.candidates else 'N/A'
                logging.warning(f"Gemini model '{model_id}' returned no text. Finish reason: {finish_reason}. Safety ratings: {safety_ratings}")
                return None # Gebe None zurück, um einen Fehler anzuzeigen
        except ValueError as e: # Fange den spezifischen ValueError vom Vertex AI SDK ab
            logging.error(f"Gemini API call failed for model '{model_id}' due to content blocking or other ValueError: {e}")
            return None
        except Exception as e:
            logging.error(f"An unexpected error occurred during Gemini API call for model '{model_id}': {e}", exc_info=True)
            return None

    # FIX 1: Funktionsdefinition wieder aktivieren, damit Claude-Calls funktionieren
    #def _call_claude(self, client: AnthropicVertex, model_id: str, system_prompt: str, user_prompt: str, image_path: Optional[str]) -> str:
        """Handles the specific API call structure for Anthropic Claude models."""
        messages = [{"role": "user", "content": []}]
        
        if image_path:
            image_data = self._encode_image(image_path)
            if image_data:
                messages[0]["content"].append({
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/png", "data": image_data},
                })
        messages[0]["content"].append({"type": "text", "text": user_prompt})
        
        response = client.messages.create(
            model=model_id,
            max_tokens=4096,
            system=system_prompt,
            messages=messages
        )
        return response.content[0].text