# utils.py - FINALE, VOLLSTÄNDIG KORRIGIERTE VERSION
# 📦 Standardbibliotheken
import os
import json
import uuid
import datetime
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Set, Any, TypedDict, Union, TypeVar, cast
from collections import defaultdict
from dataclasses import dataclass, field

# 🖼️ Bildverarbeitung & Darstellung
import cv2
import fitz  # PyMuPDF
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import textwrap # <- KORRIGIERT: textwrap hier importiert

# 🔗 Graphverarbeitung
import networkx as nx

# 🔠 Textähnlichkeit
from Levenshtein import ratio as string_similarity_ratio

logger = logging.getLogger(__name__)

# =============================================================================
#  Phase 0 & 1: Setup and File Management
# =============================================================================
def create_output_directory(base_image_path: str, model_name: str) -> Optional[str]:
    """
    Creates a standardized, timestamped output directory for an analysis run.
    """
    try:
        safe_model_name = model_name.replace("/", "_").replace(":", "_")
        image_name_without_ext = os.path.splitext(os.path.basename(base_image_path))[0]
        base_dir = os.path.dirname(base_image_path)
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        dir_name = f"{image_name_without_ext}_output_{safe_model_name}_{timestamp}"
        output_dir_path = os.path.join(base_dir, dir_name)
        os.makedirs(output_dir_path, exist_ok=True)
        return output_dir_path
    except OSError as e:
        logging.error(f"Error creating directory {output_dir_path}: {e}")
        return None

def setup_logging(log_dir: str) -> None:
    """
    Configures a FileHandler for the root logger to save logs to a run-specific file.
    """
    log_file_path = os.path.join(log_dir, "run_log.txt")
    root_logger = logging.getLogger()
    
    # Remove any existing FileHandler pointing to the same file to avoid duplicates
    for handler in root_logger.handlers[:]:
        if isinstance(handler, logging.FileHandler) and handler.baseFilename == os.path.abspath(log_file_path):
            root_logger.removeHandler(handler)
            
    file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
    file_formatter = logging.Formatter('[%(asctime)s - %(levelname)s - %(name)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)
    logging.info(f"File logging for current run initialized at: {log_file_path}")

def save_json_output(data: Dict[str, Any], file_path: str) -> None:
    """
    Saves a dictionary to a JSON file with pretty printing.
    """
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        logging.info(f"Successfully saved JSON to {file_path}")
    except IOError as e:
        logging.error(f"Error saving JSON to {file_path}: {e}")

# =============================================================================
#  Phase 2.1: Image Functions
# =============================================================================
def convert_pdf_to_images(pdf_path: Path, output_dir: Path, dpi: int = 300) -> list[Path]:
    """
    Konvertiert eine einzelne PDF-Datei in eine Serie von PNG-Bildern.
    Dies ist unser zentrales "Werkzeug" für die PDF-Konvertierung.
    """
    image_paths = []
    try:
        doc = fitz.open(pdf_path)
        logging.info(f"Konvertiere PDF '{pdf_path.name}' in Bilder...")
        for i, page in enumerate(doc):  # type: ignore[attr-defined]
            pix = page.get_pixmap(dpi=dpi)
            img_path = output_dir / f"{pdf_path.stem}_page_{i+1}.png"
            pix.save(str(img_path))
            image_paths.append(img_path)
        logging.info(f"-> Erfolgreich {len(doc)} Bilder erstellt in: {output_dir}")
    except Exception as e:
        logging.error(f"Fehler bei der Konvertierung von '{pdf_path.name}': {e}")
    return image_paths

def generate_vertical_strips(image_path: str, num_strips: int = 5, overlap_percent: int = 15, output_folder: Optional[Path] = None) -> List[Dict]:
    """
    Generiert eine Liste von überlappenden vertikalen Streifen aus einem Bild.
    """
    try:
        image = Image.open(image_path)
    except FileNotFoundError:
        logging.error(f"Cannot generate strips: Image file not found at {image_path}")
        return []

    img_width, img_height = image.size
    logging.info(f"Generating {num_strips} vertical strips for image of size {img_width}x{img_height}...")
    
    strips = []
    temp_dir = output_folder or Path(os.path.dirname(image_path)) / "temp_strips"
    os.makedirs(temp_dir, exist_ok=True)
    
    strip_width = img_width // num_strips
    overlap_px = int(strip_width * (overlap_percent / 100))

    for i in range(num_strips):
        x_start = i * strip_width
        
        # Füge Überlappung hinzu, außer beim ersten Streifen
        if i > 0:
            x_start -= overlap_px

        x_end = (i + 1) * strip_width
        # Füge Überlappung hinzu, außer beim letzten Streifen
        if i < num_strips - 1:
            x_end += overlap_px

        # Stelle sicher, dass die Box innerhalb der Bildgrenzen bleibt
        box = (max(0, x_start), 0, min(img_width, x_end), img_height)

        strip_image = image.crop(box)
        strip_path = temp_dir / f"strip_{i}.png"
        strip_image.save(str(strip_path))
        strips.append({'path': str(strip_path), 'coords': (box[0], box[1]), 'tile_width': strip_image.width, 'tile_height': strip_image.height})

    logging.info(f"Generated {len(strips)} vertical strips in '{temp_dir}'.")
    return strips

def _normalize_bbox_to_pixels(bbox: Dict[str, float], image_dims: Tuple[int, int]) -> Tuple[int, int, int, int]:
    """Konvertiert normalisierte BBox-Koordinaten in Pixelkoordinaten (x1, y1, x2, y2)."""
    img_width, img_height = image_dims
    x1 = int(bbox['x'] * img_width)
    y1 = int(bbox['y'] * img_height)
    x2 = int((bbox['x'] + bbox['width']) * img_width)
    y2 = int((bbox['y'] + bbox['height']) * img_height)
    return x1, y1, x2, y2

def _normalize_bbox_from_pixels(x1_px: int, y1_px: int, x2_px: int, y2_px: int, image_dims: Tuple[int, int]) -> Dict[str, float]:
    """Konvertiert Pixelkoordinaten (x1, y1, x2, y2) zurück in normalisierte BBox-Koordinaten."""
    img_width, img_height = image_dims
    width_px = x2_px - x1_px
    height_px = y2_px - y1_px
    return {
        'x': x1_px / img_width,
        'y': y1_px / img_height,
        'width': width_px / img_width,
        'height': height_px / img_height
    }

def _calculate_edge_density(image_segment_path: str, min_area_threshold: float = 0.0001) -> float:
    """
    Berechnet die Dichte von Kanten in einem Bildsegment mithilfe von Canny-Kantenerkennung.
    Gibt 0.0 zurück, wenn das Segment zu klein oder leer ist.
    """
    try:
        # Lade das Bildsegment in Graustufen
        img_gray = cv2.imread(image_segment_path, cv2.IMREAD_GRAYSCALE)
        if img_gray is None:
            logging.warning(f"Konnte Bildsegment '{image_segment_path}' nicht laden für Kantendichte.")
            return 0.0

        # Wende einen Median-Filter an, um Rauschen zu reduzieren
        img_blur = cv2.medianBlur(img_gray, 5) # Kernel-Größe 5

        # Canny-Kantenerkennung
        # Schwellenwerte können angepasst werden, um mehr/weniger Kanten zu finden
        edges = cv2.Canny(img_blur, 50, 150)

        # Berechne die Dichte der Kantenpixel
        edge_pixels = np.count_nonzero(edges)
        total_pixels = img_gray.shape[0] * img_gray.shape[1]

        # Wenn die Fläche zu klein ist, ist der Score nicht aussagekräftig
        if total_pixels == 0 or total_pixels < (10 * 10): # Z.B. min. 10x10 Pixel
            return 0.0

        return edge_pixels / total_pixels
    except Exception as e:
        logging.error(f"Fehler bei der Kantendichteberechnung für {image_segment_path}: {e}")
        return 0.0

def _adjust_bbox_to_content(original_image_path: str, bbox: Dict[str, float], image_dims: Tuple[int, int], padding_px: int = 5) -> Dict[str, float]:
    """
    Versucht, eine Bounding Box an den tatsächlichen Inhalt anzupassen,
    indem es leere Ränder basierend auf Kanteninformationen entfernt.
    Erhält einen kleinen Puffer (padding_px).
    """
    try:
        # Öffne das Originalbild
        img_pil = Image.open(original_image_path).convert("L") # Konvertiere zu Graustufen

        # Schneide den Bereich der BBox aus (+ kleiner Puffer, um den Rand zu erfassen)
        x1_px_orig, y1_px_orig, x2_px_orig, y2_px_orig = _normalize_bbox_to_pixels(bbox, image_dims)

        # Erweitere den Ausschnitt leicht, um Inhalt am Rand der Originalbox zu erfassen
        # Clamp-Werte, damit der Ausschnitt nicht über die Bildgrenzen hinausgeht
        padded_x1_px = max(0, x1_px_orig - padding_px)
        padded_y1_px = max(0, y1_px_orig - padding_px)
        padded_x2_px = min(image_dims[0], x2_px_orig + padding_px)
        padded_y2_px = min(image_dims[1], y2_px_orig + padding_px)

        # Wenn der gepolsterte Bereich ungültig ist, gib die Original-BBox zurück
        if padded_x2_px <= padded_x1_px or padded_y2_px <= padded_y1_px:
            logging.warning(f"Gepolsterter BBox-Bereich ist ungültig: {bbox}. Rückgabe Original.")
            return bbox

        cropped_img_pil = img_pil.crop((padded_x1_px, padded_y1_px, padded_x2_px, padded_y2_px))
        
        # Konvertiere PIL-Bild zu OpenCV-Format für bessere Bildverarbeitung
        cropped_img_np = np.array(cropped_img_pil)

        # Finde den tatsächlichen Inhalt (Nicht-Hintergrund-Pixel)
        # Hier könnten komplexere Methoden wie Otsu's Binarisierung oder adaptive Schwellenwerte verwendet werden
        _, binary_img = cv2.threshold(cropped_img_np, 200, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU) # Beispiel: invertiertes Bild
        
        # Finde Konturen oder einfach die minimalen/maximalen Koordinaten der Nicht-Null-Pixel
        non_zero_coords = np.argwhere(binary_img > 0)
        
        if non_zero_coords.size == 0:
            return bbox # Kein Inhalt gefunden, Original-BBox zurückgeben

        # Finde die minimalen/maximalen Pixelkoordinaten des Inhalts innerhalb des zugeschnittenen Bildes
        min_y, min_x = non_zero_coords.min(axis=0)
        max_y, max_x = non_zero_coords.max(axis=0)

        # Füge die Startkoordinaten des gepolsterten Ausschnitts hinzu, um absolute Pixelkoordinaten zu erhalten
        final_x1_px = padded_x1_px + min_x
        final_y1_px = padded_y1_px + min_y
        final_x2_px = padded_x1_px + max_x + 1 # +1 weil max_x ein Index ist
        final_y2_px = padded_y1_px + max_y + 1 # +1 weil max_y ein Index ist

        # Stelle sicher, dass die Box gültig ist
        if final_x2_px <= final_x1_px or final_y2_px <= final_y1_px:
             logging.warning(f"Angepasste BBox ist nach Inhaltsanpassung ungültig: {bbox}. Rückgabe Original.")
             return bbox

        # Konvertiere zurück zu normalisierten Koordinaten
        return _normalize_bbox_from_pixels(final_x1_px, final_y1_px, final_x2_px, final_y2_px, image_dims)

    except Exception as e:
        logging.error(f"Fehler bei der BBox-Anpassung an Inhalt für {bbox}: {e}", exc_info=True)
        return bbox # Im Fehlerfall die Original-BBox zurückgeben
    
def _find_dominant_orientation(binary_img: np.ndarray) -> Optional[float]:
    """
    Findet die dominante Ausrichtung von Kanten in einem binären Bild (in Grad).
    Kann nützlich sein, um gedrehte Symbole zu erkennen oder BBoxes entsprechend auszurichten.
    Gibt Winkel in Grad zurück (0-180).
    """
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    all_angles = []
    for contour in contours:
        if len(contour) > 5: # Benötigt mindestens 5 Punkte für fitEllipse
            try:
                _, _, angle = cv2.fitEllipse(contour)
                all_angles.append(float(angle)) # KORRIGIERT: Explizit zu float casten
            except cv2.Error:
                pass # Kontur ist möglicherweise nicht elliptisch genug

    if all_angles:
        return float(np.median(all_angles)) # KORRIGIERT: Explizit zu float casten
    return None

def _refine_bbox_with_contours(original_image_path: str, bbox: Dict[str, float], image_dims: Tuple[int, int], padding_px: int = 2) -> Dict[str, float]:
    """
    Versucht, die BBox präziser an die äußeren Konturen des Objekts anzupassen.
    Arbeitet mit einem kleinen Puffer und Canny-Kanten + Konturensuche.
    """
    x1_px, y1_px, x2_px, y2_px = _normalize_bbox_to_pixels(bbox, image_dims)

    padded_x1_px = max(0, x1_px - padding_px)
    padded_y1_px = max(0, y1_px - padding_px)
    padded_x2_px = min(image_dims[0], x2_px + padding_px)
    padded_y2_px = min(image_dims[1], y2_px + padding_px)

    if padded_x2_px <= padded_x1_px or padded_y2_px <= padded_y1_px:
        logging.warning(f"Gepolsterter BBox-Bereich für Kontur-Verfeinerung ungültig: {bbox}. Rückgabe Original.")
        return bbox

    try:
        img_pil = Image.open(original_image_path).convert("L") # Graustufen
        cropped_img_pil = img_pil.crop((padded_x1_px, padded_y1_px, padded_x2_px, padded_y2_px))
        img_np = np.array(cropped_img_pil)

        blurred_img = cv2.GaussianBlur(img_np, (5, 5), 0)
        edges = cv2.Canny(blurred_img, 50, 150)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return bbox

        all_x, all_y, all_w, all_h = [], [], [], []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            all_x.append(x)
            all_y.append(y)
            all_w.append(w)
            all_h.append(h)
        
        min_x_contour = min(all_x)
        min_y_contour = min(all_y)
        max_x_contour = max(x_val + w_val for x_val, y_val, w_val, h_val in zip(all_x, all_y, all_w, all_h))
        max_y_contour = max(y_val + h_val for x_val, y_val, w_val, h_val in zip(all_x, all_y, all_w, all_h))
        
        final_x1_px = padded_x1_px + min_x_contour
        final_y1_px = padded_y1_px + min_y_contour
        final_x2_px = padded_x1_px + max_x_contour
        final_y2_px = padded_y1_px + max_y_contour

        if final_x2_px <= final_x1_px or final_y2_px <= final_y1_px or \
           (final_x2_px - final_x1_px) < 5 or (final_y2_px - final_y1_px) < 5:
            logging.warning(f"Angepasste BBox nach Kontur-Verfeinerung ist zu klein/ungültig: {bbox}. Rückgabe Original.")
            return bbox

        return _normalize_bbox_from_pixels(final_x1_px, final_y1_px, final_x2_px, final_y2_px, image_dims)

    except Exception as e:
        logging.error(f"Fehler bei Kontur-basierter BBox-Verfeinerung für {bbox}: {e}", exc_info=True)
        return bbox
    
# ERSETZE DIESE FUNKTION MIT DEM FOLGENDEN AUFGEBOHRTEN CODE
def _evaluate_bbox_quality(original_image_path: str, bbox: Dict[str, float], image_dims: Tuple[int, int], 
                           config_params: Dict[str, Any], element_type: str = "Unknown") -> float:
    """
    Verbesserte heuristische Bewertung der Bounding-Box-Qualität.
    Berücksichtigt Größe, Seitenverhältnis, und Kantendichte (als Indikator für "Inhalt").

    Scores: 0.0 (sehr schlecht) bis 1.0 (sehr gut).
    """
    img_width, img_height = image_dims
    
    # Konfigurationsparameter
    min_dim_px = config_params.get('bbox_min_pixel_dim', 5)
    max_area_ratio = config_params.get('bbox_max_area_ratio', 0.8)
    aspect_ratio_range = config_params.get('bbox_aspect_ratio_range', [0.1, 10.0]) # Min, Max
    min_edge_density = config_params.get('bbox_min_edge_density', 0.005) # Z.B. 0.5% der Pixel müssen Kanten sein

    # Denormalisiere die BBox-Koordinaten für Pixelberechnungen
    x1_px, y1_px, x2_px, y2_px = _normalize_bbox_to_pixels(bbox, image_dims)
    width_px = x2_px - x1_px
    height_px = y2_px - y1_px
    
    score = 1.0 # Start mit perfektem Score und ziehe Punkte ab

    # --- 1. Grundlegende BBox-Validierung ---
    if width_px <= 0 or height_px <= 0:
        logging.debug(f"BBox ist ungültig (Breite/Höhe <= 0): {bbox}")
        return 0.0 # Ungültige Boxen sind wertlos

    # --- 2. Größe und Seitenverhältnis ---
    if width_px < min_dim_px or height_px < min_dim_px:
        score *= 0.2 # Starker Abzug für zu kleine Boxen
        logging.debug(f"BBox zu klein ({width_px}x{height_px}px): {bbox}. Score: {score:.2f}")

    if (width_px * height_px) / (img_width * img_height) > max_area_ratio:
        score *= 0.3 # Starker Abzug, wenn Box zu großen Teil des Bildes einnimmt
        logging.debug(f"BBox zu groß ({width_px}x{height_px}px): {bbox}. Score: {score:.2f}")

    aspect_ratio = width_px / height_px
    if not (aspect_ratio_range[0] <= aspect_ratio <= aspect_ratio_range[1]):
        score *= 0.5 # Moderater Abzug für unplausibles Seitenverhältnis
        logging.debug(f"BBox Seitenverhältnis unplausibel ({aspect_ratio:.2f}): {bbox}. Score: {score:.2f}")

    # --- 3. Inhaltliche Bewertung (Kantendichte) ---
    # Temporären Ausschnitt speichern, um Kantendichte zu berechnen
    temp_segment_path = Path(os.path.dirname(original_image_path)) / "temp_segments" / f"segment_{uuid.uuid4().hex[:8]}.png"
    temp_segment_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        Image.open(original_image_path).crop((x1_px, y1_px, x2_px, y2_px)).save(temp_segment_path)
        edge_density = _calculate_edge_density(str(temp_segment_path))
        
        if edge_density < min_edge_density:
            score *= 0.4 # Starker Abzug, wenn kaum Inhalt (leere Box oder Rauschen)
            logging.debug(f"BBox hat zu geringe Kantendichte ({edge_density:.4f}): {bbox}. Score: {score:.2f}")

    except Exception as e:
        logging.warning(f"Konnte Kantendichte für BBox nicht berechnen: {e}. Überspringe diese Bewertung.")
        score *= 0.8 # Leichter Abzug, da diese Bewertung nicht durchgeführt werden konnte
    finally:
        # Aufräumen des temporären Segmentpfads
        if temp_segment_path.exists():
            try:
                os.remove(temp_segment_path)
                if not any(temp_segment_path.parent.iterdir()): # Lösche Ordner, wenn leer
                    temp_segment_path.parent.rmdir()
            except OSError as e:
                logging.warning(f"Konnte temporäres Segment {temp_segment_path} nicht aufräumen: {e}")

    # Zusätzliche Regeln basierend auf Typ (optional, fortgeschritten):
    # if element_type == "Valve" and aspect_ratio > 2.0: score *= 0.7 # Ventile sind selten sehr breit
    # if element_type == "Pipe" and aspect_ratio < 0.1: score *= 0.7 # Rohre sind selten sehr kompakt

    final_score = max(0.0, min(1.0, score)) # Stelle sicher, dass der Score zwischen 0 und 1 bleibt
    logging.debug(f"Final BBox quality score for {element_type} ({bbox}): {final_score:.2f}")
    return final_score


# =============================================================================
#  Phase 2: Rastering and Synthesis Helpers
# =============================================================================
# --- Type Definitions for Clarity and Static Analysis ---
# Using TypedDicts makes the data structures explicit and helps prevent errors.
class BBox(TypedDict):
    x: float
    y: float
    width: float
    height: float

class Port(TypedDict):
    id: str
    name: str
    bbox: BBox

class Element(TypedDict):
    id: str
    label: str
    type: str
    bbox: BBox
    ports: List[Port]
    system_group: Optional[str] # Für logische Zonen wie "eChiller"

class Connection(TypedDict):
    from_id: str
    to_id: str
    from_port_id: str
    to_port_id: str
    color: Optional[str]        # Für farbliche Kodierung
    style: Optional[str]        # Für Linientypen (z.B. "dashed")
    status: Optional[str]
    original_border_coords: Optional[Dict[str, float]]
    predicted: Optional[bool]

class TileData(TypedDict):
    elements: List[Element]
    connections: List[Connection]

class TileResult(TypedDict):
    tile_width: int
    tile_height: int
    tile_coords: Tuple[int, int]
    data: TileData

class SynthesizedGraph(TypedDict):
    elements: List[Element]
    connections: List[Connection]

@dataclass 
class SynthesizerConfig:
    """Hält alle einstellbaren Parameter für den Synthesizer."""
    iou_match_threshold: float = 0.5
    # NEU: Relative Toleranzfaktoren
    border_coords_tolerance_factor_x: float = 0.015 # 1.5% der Bildbreite
    border_coords_tolerance_factor_y: float = 0.015 # 1.5% der Bildhöhe

# =============================================================================
#  FINALE, ROBUSTE GraphSynthesizer-Klasse (ersetzt deine alte komplett)
# =============================================================================
class GraphSynthesizer:
    def __init__(self, raw_results: List[TileResult], image_width: int, image_height: int, config: SynthesizerConfig = SynthesizerConfig()):
        if not image_width > 0 or not image_height > 0:
            raise ValueError("Bilddimensionen müssen positiv sein.")
        self.raw_results = raw_results
        self.image_width = image_width
        self.image_height = image_height
        self.config = config
        self.border_tolerance_x_px = self.config.border_coords_tolerance_factor_x * self.image_width
        self.border_tolerance_y_px = self.config.border_coords_tolerance_factor_y * self.image_height
        
        self._canonical_id_map: Dict[str, str] = {}
        self.final_elements: Dict[str, Element] = {}
        self.final_connections: List[Connection] = []

    def synthesize(self) -> SynthesizedGraph:
        """Führt den gesamten Syntheseprozess aus."""
        logging.info("Starte robusten Graphen-Synthese-Prozess...")
        all_elements, all_connections = self._collect_and_normalize_data()
        
        self._deduplicate_elements(all_elements)
        
        final_connections = self._stitch_connections(all_connections)
        
        logging.info(f"Synthese abgeschlossen: {len(self.final_elements)} Elemente und {len(final_connections)} Verbindungen gefunden.")
        return SynthesizedGraph(
            elements=list(self.final_elements.values()),
            connections=final_connections,
        )

    def _collect_and_normalize_data(self) -> Tuple[List[Element], List[Connection]]:
        """Sammelt und normalisiert Daten aus allen Kacheln."""
        all_elements: List[Element] = []
        all_connections: List[Connection] = []
        for result in self.raw_results:
            tile_width, tile_height, (tile_x_offset, tile_y_offset) = result['tile_width'], result['tile_height'], result['tile_coords']
            tile_data = result.get('data', {})

            elements_in_tile = tile_data.get('elements', []) if isinstance(tile_data, dict) else []
            connections_in_tile = tile_data.get('connections', []) if isinstance(tile_data, dict) else []

            for element_data in elements_in_tile:
                if self._is_valid_element(element_data):
                    # Normalisiere die BBox des Elements
                    bbox = element_data['bbox']
                    element_data['bbox'] = BBox(
                        x=(bbox['x'] * tile_width + tile_x_offset) / self.image_width,
                        y=(bbox['y'] * tile_height + tile_y_offset) / self.image_height,
                        width=(bbox['width'] * tile_width) / self.image_width,
                        height=(bbox['height'] * tile_height) / self.image_height
                    )
                    # Normalisiere die BBoxes aller Ports
                    if 'ports' in element_data and isinstance(element_data['ports'], list):
                        for port in element_data['ports']:
                            if self._is_valid_port(port):
                                port_bbox = port['bbox']
                                port['bbox'] = BBox(
                                    x=(port_bbox['x'] * tile_width + tile_x_offset) / self.image_width,
                                    y=(port_bbox['y'] * tile_height + tile_y_offset) / self.image_height,
                                    width=(port_bbox['width'] * tile_width) / self.image_width,
                                    height=(port_bbox['height'] * tile_height) / self.image_height
                                )
                    all_elements.append(element_data)

            for conn_data in connections_in_tile:
                if self._is_valid_connection(conn_data):
                    all_connections.append(conn_data)
                    
        return all_elements, all_connections

    def _deduplicate_elements(self, elements: List[Element]):
        """Gruppiert und fasst doppelte Elemente zusammen, inkl. ihrer Ports."""
        if not elements:
            return

        groups = self._group_duplicate_elements(elements)
        logging.info(f"{len(groups)} einzigartige Element-Gruppen nach Deduplizierung gefunden.")
    
        for group in groups:
            best_label = self._select_best_attribute([el.get('label') for el in group], prefer_longer=True)
            best_type = self._select_best_attribute([el.get('type') for el in group]) or "Unknown"
            best_system_group = self._select_best_attribute([el.get('system_group') for el in group], default=None)

            final_bbox = self._aggregate_bboxes([el['bbox'] for el in group])
            
            # --- NEU: Sammle und dedupliziere alle Ports aus der Gruppe ---
            all_ports_in_group: List[Port] = []
            for el in group:
                # Filter out invalid ports before extending the list
                # Sicherstellen, dass nur gültige Port-Objekte (mit 'bbox') hinzugefügt werden
                valid_ports_from_element = [
                    cast(Port, p) for p in el.get('ports', []) if self._is_valid_port(p)
                ]
                all_ports_in_group.extend(valid_ports_from_element)
            
            unique_ports = self._deduplicate_ports(all_ports_in_group)
            # -----------------------------------------------------------

            canonical_id = f"el_{uuid.uuid4().hex[:8]}"
            
            self.final_elements[canonical_id] = Element(
                id=canonical_id,
                label=best_label or f"Element_{canonical_id[:4]}",
                type=best_type,
                bbox=final_bbox,
                ports=unique_ports, # Füge die zusammengeführten Ports hinzu
                system_group=best_system_group
            )
            
            for original_element in group:
                if original_id := original_element.get('id'):
                    self._canonical_id_map[original_id] = canonical_id # Map old ID to new canonical ID
                    # Map all ports of this original element to the new canonical element ID
                    for port in original_element.get('ports', []):
                        if port_id := port.get('id'):
                            self._canonical_id_map[port_id] = canonical_id

    def _group_duplicate_elements(self, elements: List[Element]) -> List[List[Element]]:
        """
        Gruppiert Elemente, die wahrscheinlich Duplikate sind, basierend auf IoU und Label-Ähnlichkeit.
        """
        if not elements:
            return []

        # Erstelle eine Matrix für bereits zugewiesene Elemente
        assigned = [False] * len(elements)
        groups = []

        for i in range(len(elements)):
            if assigned[i]:
                continue
            
            # Starte eine neue Gruppe mit dem aktuellen Element
            current_group = [elements[i]]
            assigned[i] = True
            
            # Vergleiche mit allen nachfolgenden, noch nicht zugewiesenen Elementen
            for j in range(i + 1, len(elements)):
                if not assigned[j]:
                    iou = self._calculate_iou(elements[i]['bbox'], elements[j]['bbox'])
                    
                    label_i = self._normalize_label(elements[i].get('label', ''))
                    label_j = self._normalize_label(elements[j].get('label', ''))
                    
                    # Wenn Labels vorhanden sind, müssen sie ähnlich sein
                    if label_i and label_j:
                        label_sim = string_similarity_ratio(label_i, label_j)
                        if iou > self.config.iou_match_threshold and label_sim > 0.85:
                            current_group.append(elements[j])
                            assigned[j] = True
                    # Wenn keine Labels vorhanden sind, verlasse dich nur auf eine hohe IoU
                    elif not label_i and not label_j:
                        if iou > 0.9: # Höherer Schwellenwert für Elemente ohne Label
                            current_group.append(elements[j])
                            assigned[j] = True

            groups.append(current_group)
            
        return groups

    def _deduplicate_ports(self, ports: List[Port]) -> List[Port]:
        """Fasst Ports mit ähnlicher Position und Namen zusammen."""
        unique_ports: List[Port] = []
        unassigned_ports = list(ports)
        while unassigned_ports:
            base_port = unassigned_ports.pop(0)
            current_group = [base_port]
            remaining_ports = []
            for other_port in unassigned_ports:
                iou = self._calculate_iou(base_port['bbox'], other_port['bbox'])
                if iou > 0.8: # Hohe Überlappung deutet auf denselben Port hin
                    current_group.append(other_port)
                else:
                    remaining_ports.append(other_port)
            unassigned_ports = remaining_ports
            
            # Wähle den besten Port aus der Gruppe (z.B. den mit dem häufigsten Namen)
            best_name = self._select_best_attribute([p.get('name') for p in current_group]) or "default"
            aggregated_bbox = self._aggregate_bboxes([p['bbox'] for p in current_group])
            best_id = self._select_best_attribute([p.get('id') for p in current_group]) or f"port_{uuid.uuid4().hex[:6]}"

            unique_ports.append(Port(id=best_id, name=best_name, bbox=aggregated_bbox))
        return unique_ports

    def _stitch_connections(self, connections: List[Connection]) -> List[Connection]:
        """Fügt Verbindungen zusammen, jetzt mit korrekter Port-ID-Auflösung."""
        final_connections_list: List[Connection] = []
        for conn in connections:
            # --- ERWEITERT: Löst jetzt Element- und Port-IDs auf ---
            from_element_id = self._resolve_canonical_id(conn.get('from_id'))
            to_element_id = self._resolve_canonical_id(conn.get('to_id'))
            from_port_id = conn.get('from_port_id') # Behalte die originale Port-ID
            to_port_id = conn.get('to_port_id')     # Behalte die originale Port-ID
            
            if from_element_id and to_element_id and from_element_id != to_element_id:
                final_connections_list.append(
                    Connection(
                        from_id=from_element_id,
                        to_id=to_element_id,
                        from_port_id=from_port_id,
                        to_port_id=to_port_id,
                        color=conn.get('color'),
                        style=conn.get('style'),
                        status='stitched',
                        predicted=False,
                        original_border_coords=None
                    )
                )
        return final_connections_list

    def _resolve_canonical_id(self, original_id: Optional[str]) -> Optional[str]:
        """Findet die neue, saubere ID für eine alte Element- oder Port-ID."""
        if not original_id: return None
        return self._canonical_id_map.get(original_id)

    def _are_border_connections_match(self, conn1: Connection, conn2: Connection) -> bool:
        """Prüft, ob zwei Grenzverbindungen zusammenpassen (robust gegen Listen und Dictionaries)."""
        coords1 = conn1.get('original_border_coords')
        coords2 = conn2.get('original_border_coords')
        if not coords1 or not coords2: return False

        x1, y1, x2, y2 = 0.0, 0.0, 0.0, 0.0

        # Fall 1: Beide Koordinaten sind Dictionaries
        if isinstance(coords1, dict) and isinstance(coords2, dict):
            x1, y1 = coords1.get('x', 0.0), coords1.get('y', 0.0)
            x2, y2 = coords2.get('x', 0.0), coords2.get('y', 0.0)
        # Fall 2: Beide Koordinaten sind Listen
        elif isinstance(coords1, list) and len(coords1) >= 2 and isinstance(coords2, list) and len(coords2) >= 2:
            # Expliziter Cast, um Pylance die Sicherheit zu geben, dass dies Listen sind
            coords1_list = cast(List[float], coords1)
            coords2_list = cast(List[float], coords2)
            x1, y1 = coords1_list[0], coords1_list[1]
            x2, y2 = coords2_list[0], coords2_list[1]
        # Fall 3: Format ist gemischt oder unbekannt -> kein Match
        else:
            return False

        # Gemeinsame Logik für den Distanzvergleich
        if not (abs(x1 - x2) * self.image_width < self.border_tolerance_x_px and 
                abs(y1 - y2) * self.image_height < self.border_tolerance_y_px):
            return False
        
        border_id1 = conn1.get('to_id') if conn1.get('to_id', '').startswith("BORDER_") else conn1.get('from_id')
        border_id2 = conn2.get('to_id') if conn2.get('to_id', '').startswith("BORDER_") else conn2.get('from_id')
        return self._are_compatible_borders(border_id1, border_id2)

    def _create_stitched_connection(self, conn1: Connection, conn2: Connection) -> Optional[Connection]:
        """Erstellt eine zusammengefügte Verbindung aus zwei passenden Grenzverbindungen."""
        source_conn, dest_conn = (conn1, conn2) if conn1.get('to_id', '').startswith("BORDER_") else (conn2, conn1)
        from_id = self._resolve_canonical_id(source_conn.get('from_id'))
        to_id = self._resolve_canonical_id(dest_conn.get('to_id'))
        if from_id and to_id:
            return Connection(from_id=from_id, to_id=to_id, from_port_id=source_conn.get('from_port_id', ''), to_port_id=dest_conn.get('to_port_id', ''), color=None, style=None, status='stitched', predicted=False, original_border_coords=None)
        return None

    # Statische Methoden und Hilfsfunktionen
    @staticmethod
    def _is_valid_port(port: Any) -> bool:
        if not isinstance(port, dict): return False
        bbox = port.get('bbox')
        return ('id' in port and 'name' in port and isinstance(bbox, dict) and all(k in bbox for k in ['x', 'y', 'width', 'height']))

    @staticmethod
    def _is_valid_element(element: Any) -> bool:
        if not isinstance(element, dict): return False
        bbox = element.get('bbox')
        return ('id' in element and isinstance(bbox, dict) and all(k in bbox for k in ['x', 'y', 'width', 'height']))

    @staticmethod
    def _is_valid_connection(connection: Any) -> bool:
        return (isinstance(connection, dict) and 'from_id' in connection and 'to_id' in connection)

    @staticmethod
    def _normalize_label(label: Optional[str]) -> str:
        if label is None: return ""
        return str(label).lower().replace(" ", "").replace("-", "").replace("_", "").strip()

    @staticmethod
    def _calculate_iou(box_a: BBox, box_b: BBox) -> float:
        x_a = max(box_a['x'], box_b['x'])
        y_a = max(box_a['y'], box_b['y'])
        x_b = min(box_a['x'] + box_a['width'], box_b['x'] + box_b['width'])
        y_b = min(box_a['y'] + box_a['height'], box_b['y'] + box_b['height'])
        inter_area = max(0, x_b - x_a) * max(0, y_b - y_a)
        if inter_area == 0: return 0.0
        box_a_area = box_a['width'] * box_a['height']
        box_b_area = box_b['width'] * box_b['height']
        union_area = box_a_area + box_b_area - inter_area
        return inter_area / union_area if union_area > 0 else 0.0

    @staticmethod
    def _aggregate_bboxes(bboxes: List[BBox]) -> BBox:
        if not bboxes: return BBox(x=0, y=0, width=0, height=0)
        min_x = min(b['x'] for b in bboxes)
        min_y = min(b['y'] for b in bboxes)
        max_x = max(b['x'] + b['width'] for b in bboxes)
        max_y = max(b['y'] + b['height'] for b in bboxes)
        return BBox(x=min_x, y=min_y, width=max_x - min_x, height=max_y - min_y)

    @staticmethod
    def _select_best_attribute(attributes: List[Optional[str]], default: Optional[str] = None, prefer_longer: bool = False) -> Optional[str]:
        """
        Wählt den besten Attributwert aus einer Liste aus, z.B. den häufigsten oder längsten.
        Kann jetzt auch None als Standardwert und Rückgabewert verarbeiten.
        """
        # Filtere None-Werte heraus, bevor gezählt wird
        valid_attributes = [attr for attr in attributes if attr is not None]
        if not valid_attributes:
            return default

        counts = defaultdict(int)
        for attr in valid_attributes:
            counts[attr] += 1
        
        if prefer_longer:
            return max(counts, key=lambda k: (counts[k], len(k)))
        else:
            return max(counts, key=lambda k: counts[k])

    @staticmethod
    def _are_compatible_borders(border1_id: Optional[str], border2_id: Optional[str]) -> bool:
        if not border1_id or not border2_id: return False
        opposite_pairs = {("LEFT", "RIGHT"), ("TOP", "BOTTOM")}
        
        def extract_info(border_id: str) -> Optional[Dict[str, str]]:
            if not border_id.startswith("BORDER_"): return None
            parts = border_id.split("_")
            if len(parts) < 2: return None
            return {"direction": parts[1], "type": "_".join(parts[2:])}

        info1 = extract_info(border1_id)
        info2 = extract_info(border2_id)
        if not info1 or not info2: return False

        if (info1["direction"], info2["direction"]) not in opposite_pairs and \
           (info2["direction"], info1["direction"]) not in opposite_pairs:
            return False
        
        type1_norm = GraphSynthesizer._normalize_label(info1.get("type"))
        type2_norm = GraphSynthesizer._normalize_label(info2.get("type"))
        return type1_norm == type2_norm

    @staticmethod
    def _classify_connections(connections: List[Connection]) -> Tuple[List[Connection], List[Connection]]:
        normal, border = [], []
        for conn in connections:
            if conn.get('from_id', '').startswith("BORDER_") or conn.get('to_id', '').startswith("BORDER_"):
                border.append(conn)
            else:
                normal.append(conn)
        return normal, border

    def _calculate_duplicate_confidence(self, el1: Element, el2: Element) -> float:
        iou = self._calculate_iou(el1['bbox'], el2['bbox'])
        if iou < 0.1: return 0.0
        iou_score = min(iou / self.config.iou_match_threshold, 1.0)
        norm_label1 = self._normalize_label(el1.get('label'))
        norm_label2 = self._normalize_label(el2.get('label'))
        label_score = string_similarity_ratio(norm_label1, norm_label2) if norm_label1 and norm_label2 else 0.0
        return (iou_score * 0.7) + (label_score * 0.3)
    
    def export_debug_image(self, original_image_path: str, output_path: str):
        """Draws the final synthesized graph onto the original image for debugging."""
        if not self.final_elements:
            logging.warning("Cannot generate debug image, synthesis has not been run or found no elements.")
            return

        try:
            img = Image.open(original_image_path).convert("RGB")
            draw = ImageDraw.Draw(img)
            img_width, img_height = img.size

            # Elemente zeichnen
            for el_id, element in self.final_elements.items():
                bbox = element['bbox']
                # Koordinaten de-normalisieren
                shape = [
                    bbox['x'] * img_width, 
                    bbox['y'] * img_height,
                    (bbox['x'] + bbox['width']) * img_width,
                    (bbox['y'] + bbox['height']) * img_height
                ]
                draw.rectangle(shape, outline="red", width=3)
                label = element.get('label')
                if label:
                    draw.text((shape[0], shape[1] - 18), label, fill="red") 

            if hasattr(self, 'final_connections') and self.final_connections:
                for conn in self.final_connections:
                    from_el = self.final_elements.get(conn['from_id'])
                    to_el = self.final_elements.get(conn['to_id'])

                    if from_el and to_el:
                        from_bbox = from_el['bbox']
                        to_bbox = to_el['bbox']

                        from_center_x = (from_bbox['x'] + from_bbox['width']/2) * img_width
                        from_center_y = (from_bbox['y'] + from_bbox['height']/2) * img_height
                        to_center_x = (to_bbox['x'] + to_bbox['width']/2) * img_width
                        to_center_y = (to_bbox['y'] + to_bbox['height']/2) * img_height

                        draw.line([(from_center_x, from_center_y), (to_center_x, to_center_y)], fill="blue", width=2)

            img.save(output_path)
            logging.info(f"Visual debug image saved to {output_path}")
        except Exception as e:
            logging.error(f"Failed to generate visual debug image: {e}", exc_info=True)


def generate_raster_grid(image_path: str, tile_size: int = 1024, overlap: int = 128, excluded_zones: Optional[List[Dict]] = None, output_folder: Optional[Path] = None) -> List[Dict]:
    """
    Generates a list of overlapping tiles (raster grid) from a source image,
    avoiding specified excluded zones.
    """
    try:
        image = Image.open(image_path)
    except FileNotFoundError:
        logging.error(f"Cannot generate raster: Image file not found at {image_path}")
        return []

    img_width, img_height = image.size
    logging.info(f"Generating raster grid for image of size {img_width}x{img_height}...")
    
    tiles = []
    # KORRIGIERT: Verwende output_folder, falls vorhanden, sonst Standard
    temp_dir = output_folder if output_folder else Path(os.path.dirname(image_path)) / "temp_tiles"
    os.makedirs(temp_dir, exist_ok=True)
    
    stride = tile_size - overlap

    for y in range(0, img_height, stride):
        for x in range(0, img_width, stride):
            right = min(x + tile_size, img_width)
            bottom = min(y + tile_size, img_height)
            box = (x, y, right, bottom)
            
            tile_is_excluded = False
            if excluded_zones:
                for zone in excluded_zones:
                    ex_x1 = zone['x'] * img_width
                    ex_y1 = zone['y'] * img_height
                    ex_x2 = ex_x1 + zone['width'] * img_width
                    ex_y2 = ex_y1 + zone['height'] * img_height
                    if not (right < ex_x1 or x > ex_x2 or bottom < ex_y1 or y > ex_y2):
                        tile_is_excluded = True
                        break
            
            if tile_is_excluded:
                logging.debug(f"Skipping tile at ({x},{y}) due to excluded zone.")
                continue

            tile_image = image.crop(box)
            tile_path = os.path.join(temp_dir, f"tile_{x}_{y}.png")
            tile_image.save(tile_path)
            tiles.append({'path': tile_path, 'coords': (x, y), 'tile_width': tile_image.width, 'tile_height': tile_image.height})

    logging.info(f"Generated {len(tiles)} tiles in '{temp_dir}'.")
    return tiles

def calculate_tile_complexity(tile_path: str, canny_threshold1=50, canny_threshold2=150) -> float:
    """
    Calculates a complexity score for a tile based on edge density (Canny edge detection).
    """
    try:
        tile = cv2.imread(tile_path, cv2.IMREAD_GRAYSCALE)
        if tile is None: return 0.0
        
        edges = cv2.Canny(tile, canny_threshold1, canny_threshold2)
        edge_pixels = np.count_nonzero(edges)
        total_pixels = tile.shape[0] * tile.shape[1]
        
        return edge_pixels / total_pixels if total_pixels > 0 else 0.0
    except Exception as e:
        logging.error(f"Error during complexity calculation for {tile_path}: {e}")
        return 0.0

def is_tile_complex(tile_path: str, edge_pixel_threshold=0.01) -> bool:
    """
    Checks if an image tile is "complex" enough for analysis based on its complexity score.
    """
    complexity_score = calculate_tile_complexity(tile_path)
    return complexity_score > edge_pixel_threshold

def merge_overlapping_boxes(boxes: List[Dict]) -> List[Dict]:
    """
    Merges bounding boxes that overlap into larger, comprehensive zones.
    """
    if not boxes:
        return []

    rects = [[b['x'], b['y'], b['x'] + b['width'], b['y'] + b['height']] for b in boxes]
    
    merged_rects = []
    while len(rects) > 0:
        r1 = rects.pop(0)
        was_merged = False
        for i in range(len(merged_rects)):
            r2 = merged_rects[i]
            # Check for overlap
            if not (r1[2] < r2[0] or r1[0] > r2[2] or r1[3] < r2[1] or r1[1] > r2[3]):
                # Merge
                min_x = min(r1[0], r2[0])
                min_y = min(r1[1], r2[1])
                max_x = max(r1[2], r2[2])
                max_y = max(r1[3], r2[3])
                merged_rects[i] = [min_x, min_y, max_x, max_y]
                was_merged = True
                # Since r1 is merged, we need to re-check against the newly expanded box
                rects.append(merged_rects.pop(i))
                break
        if not was_merged:
            merged_rects.append(r1)
            
# Filter ungültige Boxen (Breite oder Höhe <= 0)
    return [{'x': r[0], 'y': r[1], 'width': r[2] - r[0], 'height': r[3] - r[1]} 
            for r in merged_rects if (r[2] - r[0]) > 0 and (r[3] - r[1]) > 0] # HINZUGEFÜGT: Prüfung > 0


def generate_raster_grid_in_zones(image_path: str, zones: List[Dict], tile_size: int = 1024, overlap: int = 128, output_folder: Optional[Path] = None) -> List[Dict]:
    """
    Erzeugt Kacheln, die sich spezifisch innerhalb der übergebenen "Zonen" (Hotspots) befinden.
    """
    try:
        image = Image.open(image_path)
    except FileNotFoundError:
        logging.error(f"Kann Raster nicht erstellen: Bilddatei nicht gefunden unter {image_path}")
        return []

    img_width, img_height = image.size
    logging.info(f"Erzeuge feingranulares Raster innerhalb von {len(zones)} Zonen...")
    
    tiles = []
    temp_dir = output_folder or Path(os.path.dirname(image_path)) / "temp_tiles"
    os.makedirs(temp_dir, exist_ok=True)
    
    stride = tile_size - overlap
    processed_boxes = set()

    for zone in zones:
        if not (isinstance(zone, dict) and all(k in zone for k in ['x', 'y', 'width', 'height'])):
            continue

        # Berechne die Pixel-Koordinaten für jede Zone
        x_start_raw = int(zone['x'] * img_width)
        y_start_raw = int(zone['y'] * img_height)
        x_end_raw = int((zone['x'] + zone['width']) * img_width)
        y_end_raw = int((zone['y'] + zone['height']) * img_height)

        # SICHERHEITSPRÜFUNG: Stelle sicher, dass Start < Ende ist
        zone_x_start = min(x_start_raw, x_end_raw)
        zone_x_end = max(x_start_raw, x_end_raw)
        zone_y_start = min(y_start_raw, y_end_raw)
        zone_y_end = max(y_start_raw, y_end_raw)

        # Iteriere über die Zone und erstelle Kacheln
        for y in range(zone_y_start, zone_y_end, stride):
            for x in range(zone_x_start, zone_x_end, stride):
                box = (x, y, min(x + tile_size, img_width), min(y + tile_size, img_height))
                
                if box in processed_boxes:
                    continue

                if box[2] > box[0] and box[3] > box[1]:
                    tile_image = image.crop(box)
                    tile_path = temp_dir / f"tile_fine_{x}_{y}.png"
                    tile_image.save(str(tile_path))
                    tiles.append({'path': str(tile_path), 'coords': (x, y), 'tile_width': tile_image.width, 'tile_height': tile_image.height})
                    processed_boxes.add(box)

    logging.info(f"-> {len(tiles)} feingranulare Kacheln innerhalb der Zonen erstellt.")
    return tiles

def predict_and_complete_graph(
    elements: List[Element],          # Now correctly typed as List[Element]
    connections: List[Connection],    # Now correctly typed as List[Connection]
    logger: logging.Logger
) -> List[Connection]:                # Return type also correctly typed as List[Connection]
    """
    Uses geometric heuristics to add probable, missing connections between
    nearby, unconnected elements.
    """
    logger.info("Starting predictive graph completion to close gaps...")

    # Initialize a directed graph (DiGraph) to correctly represent 'from' and 'to' in connections.
    # We will use its undirected view for finding isolated nodes.
    G = nx.DiGraph()

    node_positions = {}
    for el in elements:
        # Use 'id' for nodes in the graph as it's typically the unique identifier.
        # Check that 'id' and 'bbox' keys exist to prevent errors with malformed data.
        if el.get('id') and el.get('bbox'):
            G.add_node(el['id']) # Add node by its unique ID
            pos = el['bbox']
            # Store the center of the bounding box for distance calculations
            node_positions[el['id']] = (pos['x'] + pos['width']/2, pos['y'] + pos['height']/2)

    for conn in connections:
        from_id = conn.get('from_id')
        to_id = conn.get('to_id')
        # Only add an edge if both from_id and to_id exist as nodes in the graph
        if from_id in G and to_id in G:
            G.add_edge(from_id, to_id)

    # Convert the graph to an undirected one to find isolated nodes (nodes with no connections)
    G_undirected = G.to_undirected()
    isolated_nodes = list(nx.isolates(G_undirected))

    if not isolated_nodes:
        logger.info("No isolated nodes found. Predictive completion not necessary.")
        return connections # Return the original connections if no prediction is needed

    logger.info(f"Executing heuristic for {len(isolated_nodes)} isolated nodes...")
    newly_predicted_connections: List[Connection] = [] # Type hint for the new list of connections

    # Iterate through all unique pairs of isolated nodes
    for i, node1_id in enumerate(isolated_nodes):
        for node2_id in isolated_nodes[i+1:]: # Starts from i+1 to avoid duplicate pairs and self-comparison
            # Defensive check: ensure both nodes have stored positions
            if node1_id not in node_positions:
                logger.warning(f"Node {node1_id} found in isolated_nodes but has no position data. Skipping.")
                continue
            if node2_id not in node_positions:
                logger.warning(f"Node {node2_id} found in isolated_nodes but has no position data. Skipping.")
                continue

            pos1 = node_positions[node1_id]
            pos2 = node_positions[node2_id]

            # Calculate the Euclidean distance between the centers of the two nodes
            distance = ((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)**0.5

            # HEURISTIC: If two isolated nodes are very close, predict a connection.
            # The threshold (0.05) is an example; tune this value based on your diagram's scale.
            if distance < 0.05:
                logger.info(f"PREDICTION: Adding probable connection between close, isolated nodes: {node1_id} <-> {node2_id} (Distance: {distance:.3f})")

                # Create a new Connection TypedDict instance for the predicted connection.
                # Adheres to the new structure with port IDs.
                new_conn: Connection = Connection(
                    from_id=node1_id,
                    to_id=node2_id,
                    from_port_id="",
                    to_port_id="",
                    color=None,
                    style=None,
                    predicted=True,
                    status="predicted_by_heuristic",
                    original_border_coords=None
                )
                newly_predicted_connections.append(new_conn)

                # Add this newly predicted edge to our main graph 'G'.
                # This doesn't affect the current `isolated_nodes` list (which was computed earlier),
                # but it ensures 'G' reflects the predicted state for any subsequent operations.
                G.add_edge(node1_id, node2_id)

    # Return the original connections combined with the newly predicted connections.
    return connections + newly_predicted_connections

def crop_image_for_correction(image_path: str, bbox: Dict, context_margin: float = 0.1) -> Optional[str]:
    """
    Crops the original image to a specific bounding box plus a margin for context.
    Returns the path to the temporary cropped image file.
    """
    try:
        image = Image.open(image_path)
        img_width, img_height = image.size

        x1 = int(bbox['x'] * img_width)
        y1 = int(bbox['y'] * img_height)
        x2 = int((bbox['x'] + bbox['width']) * img_width)
        y2 = int((bbox['y'] + bbox['height']) * img_height)

        margin_x = int((x2 - x1) * context_margin)
        margin_y = int((y2 - y1) * context_margin)

        crop_x1 = max(0, x1 - margin_x)
        crop_y1 = max(0, y1 - margin_y)
        crop_x2 = min(img_width, x2 + margin_x)
        crop_y2 = min(img_height, y2 + margin_y)

        cropped_image = image.crop((crop_x1, crop_y1, crop_x2, crop_y2))
        
        temp_dir = os.path.join(os.path.dirname(image_path), "temp_tiles")
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = os.path.join(temp_dir, f"correction_snippet_{x1}_{y1}.png")
        
        cropped_image.save(temp_path)
        logging.info(f"Created correction snippet: {temp_path}")
        return temp_path

    except Exception as e:
        logging.error(f"Error creating cropped image for correction: {e}", exc_info=True)
        return None

def get_local_graph_context(element_label: str, all_elements: List[Dict], all_connections: List[Dict], search_radius: float = 0.2) -> str:
    """
    Generates a text summary of an element's local neighborhood to provide
    rich context for a correction prompt.
    """
    elements_by_label = {el.get("label"): el for el in all_elements if el.get("label")}
    problem_element = elements_by_label.get(element_label)
    if not problem_element:
        return "Element in focus not found."

    problem_bbox = problem_element.get("bbox", {})
    if not problem_bbox:
        return f"Element '{element_label}' has no location data."
        
    center_x = problem_bbox.get('x', 0) + problem_bbox.get('width', 0) / 2
    center_y = problem_bbox.get('y', 0) + problem_bbox.get('height', 0) / 2
    
    nearby_elements = []
    for el in all_elements:
        if el.get("label") == element_label: continue
        el_bbox = el.get("bbox", {})
        if not el_bbox: continue
        
        el_center_x = el_bbox.get('x', 0) + el_bbox.get('width', 0) / 2
        el_center_y = el_bbox.get('y', 0) + el_bbox.get('height', 0) / 2
        
        distance = ((center_x - el_center_x)**2 + (center_y - el_center_y)**2)**0.5
        if distance < search_radius:
            nearby_elements.append(el)

    context = f"- Element in focus: {{'label': '{element_label}', 'type': '{problem_element.get('type')}'}}\n"
    if not nearby_elements:
        context += "- No other elements were found nearby."
    else:
        context += "- Nearby elements are:\n"
        for el in nearby_elements:
            context += f"  - {{'label': '{el.get('label')}', 'type': '{el.get('type')}'}}\n"
            
    return context

# =============================================================================
#  Phase 3: KPI Calculation Helpers
# =============================================================================

def calculate_raw_kpis_with_truth(ai_data: Dict, truth_data: Dict, type_aliases: Dict) -> Dict:
    """
    Compares the AI response with the ground truth data to calculate detailed KPIs
    for elements, types, and connections.
    """
    logging.info("Starting detailed KPI calculation by comparing with ground truth...")
    
    ai_elements = {el['id']: el for el in ai_data.get('elements', []) if el.get('id')}
    truth_elements = {el['id']: el for el in truth_data.get('elements', []) if el.get('id')}
    ai_ids = set(ai_elements.keys())
    truth_ids = set(truth_elements.keys())

    correctly_found_ids = ai_ids.intersection(truth_ids)
    missed_ids = truth_ids - ai_ids
    hallucinated_ids = ai_ids - truth_ids
    
    precision = len(correctly_found_ids) / len(ai_ids) if ai_ids else 0
    recall = len(correctly_found_ids) / len(truth_ids) if truth_ids else 0
    
    type_correct_count = 0
    type_errors = []
    alias_map = {alias.lower(): official for official, alias_list in type_aliases.items() for alias in alias_list}
    
    for el_id in correctly_found_ids:
        ai_type = ai_elements.get(el_id, {}).get('type', 'N/A')
        truth_type = truth_elements.get(el_id, {}).get('type', 'N/A')
        
        # Normalize types for comparison
        ai_type_lower = ai_type.lower()
        truth_type_lower = truth_type.lower()
        
        # Check for direct match or alias match
        if ai_type_lower == truth_type_lower or alias_map.get(ai_type_lower) == truth_type:
            type_correct_count += 1
        else:
            # KORREKTUR: Erzeuge ein Dictionary statt eines Strings
            type_errors.append({
                "element_id": el_id,
                "ai_type": ai_type,
                "truth_type": truth_type
            })
            
    type_accuracy = type_correct_count / len(correctly_found_ids) if correctly_found_ids else 0

    def get_edges(connections):
        edges = set()
        if not connections: return edges
        for conn in connections:
            if 'from_id' in conn and 'to_id' in conn:
                edges.add(tuple(sorted((conn.get('from_id'), conn.get('to_id')))))
        return edges

    ai_edges = get_edges(ai_data.get('connections', []))
    truth_edges = get_edges(truth_data.get('connections', []))
    
    correct_connections = ai_edges.intersection(truth_edges)
    connection_precision = len(correct_connections) / len(ai_edges) if ai_edges else 0
    connection_recall = len(correct_connections) / len(truth_edges) if truth_edges else 0
    
    kpi_summary = {
        "element_precision": round(precision, 3), "element_recall": round(recall, 3),
        "type_accuracy": round(type_accuracy, 3),
        "connection_precision": round(connection_precision, 3),
        "connection_recall": round(connection_recall, 3),
        "error_details": {
            "missed_elements": list(missed_ids), "hallucinated_elements": list(hallucinated_ids),
            "typing_errors": type_errors,
            "missed_connections": [list(e) for e in (truth_edges - ai_edges)],
            "hallucinated_connections": [list(e) for e in (ai_edges - truth_edges)]
        }
    }
    logging.info("Detailed KPI calculation complete.")
    return kpi_summary

def calculate_connectivity_kpis(elements: List[Dict], connections: List[Dict]) -> Dict[str, Any]:
    """
    Calculates graph-theory-based KPIs using the networkx library for internal
    consistency checks when no ground truth is available.
    """
    logging.info("Calculating connectivity KPIs...")
    G = nx.DiGraph()

    element_labels = {element.get('label') for element in elements if element.get('label')}
    if not element_labels:
        return {"total_nodes": 0, "total_edges": 0, "subgraph_count": 0, "isolated_elements_count": 0, "dangling_edges_count": 0}

    G.add_nodes_from(element_labels)

    for conn in connections:
        from_node = conn.get('from_id')
        to_node = conn.get('to_id')
        if from_node and to_node and from_node in G and to_node in G:
            G.add_edge(from_node, to_node)

    G_undirected = G.to_undirected()
    num_subgraphs = nx.number_connected_components(G_undirected)
    isolated_nodes = len(list(nx.isolates(G_undirected)))

    dangling_count = 0
    if num_subgraphs > 1:
        try:
            largest_cc = max(nx.connected_components(G_undirected), key=len)
            dangling_count = G.number_of_nodes() - len(largest_cc)
        except ValueError:
            dangling_count = 0

    kpis = {
        "total_nodes": G.number_of_nodes(),
        "total_edges": G.number_of_edges(),
        "subgraph_count": num_subgraphs,
        "isolated_elements_count": isolated_nodes,
        "dangling_components_node_count": dangling_count
    }
    logging.info(f"Connectivity KPIs: {kpis}")
    return kpis

def calculate_semantic_kpis(elements: List[Dict], knowledge_repo: Dict) -> Dict[str, Any]:
    """
    Calculates KPIs related to the semantic correctness of the recognition.
    """
    logging.info("Calculating semantic KPIs...")
    total_elements = len(elements)
    if total_elements == 0:
        return {"type_coverage_percent": 100.0, "unidentified_element_count": 0}

    identified_elements_count = sum(1 for el in elements if el.get('type', '').lower() != 'unknown')
    
    known_types_from_legend = set(knowledge_repo.get('symbol_map', {}).values())
    unidentified_elements_count = sum(1 for el in elements if el.get('type') not in known_types_from_legend and el.get('type') is not None)

    type_coverage = (identified_elements_count / total_elements) * 100 if total_elements > 0 else 100.0
    
    kpis = {
        "identified_elements_count": identified_elements_count,
        "unidentified_element_count": unidentified_elements_count,
        "type_coverage_percent": round(type_coverage, 2)
    }
    logging.info(f"Semantic KPIs: {kpis}")
    return kpis


def calculate_final_quality_score(final_kpis: Dict[str, Any], use_truth_kpis: bool = False) -> float:
    """
    Calculates a single, aggregated quality score from all KPI sets based on
    internal consistency checks.
    """
    logging.info("Calculating final internal quality score...")
    score = 100.0
   
    subgraph_penalty = (final_kpis.get("subgraph_count", 1) - 1) * 25.0
    score -= subgraph_penalty
    if subgraph_penalty > 0:
        logging.warning(f"Applying penalty of {subgraph_penalty} for {final_kpis.get('subgraph_count')} subgraphs.")

    isolated_penalty = final_kpis.get("isolated_elements_count", 0) * 15.0
    score -= isolated_penalty
    if isolated_penalty > 0:
        logging.warning(f"Applying penalty of {isolated_penalty} for {final_kpis.get('isolated_elements_count')} isolated elements.")

    dangling_penalty = final_kpis.get("dangling_components_node_count", 0) * 5.0
    score -= dangling_penalty
    if dangling_penalty > 0:
        logging.warning(f"Applying penalty of {dangling_penalty} for nodes in dangling components.")

    unidentified_penalty = final_kpis.get("unidentified_element_count", 0) * 2.0
    score -= unidentified_penalty
    if unidentified_penalty > 0:
        logging.warning(f"Applying penalty of {unidentified_penalty} for unidentified element types.")
        
    # --- NEUE REALITÄTS-PRÜFUNG HINZUFÜGEN ---
    total_nodes = final_kpis.get("total_nodes", 0)
    if total_nodes < 10: # Wenn weniger als 10 Elemente gefunden wurden...
        # ...reduziere den Score drastisch, weil wahrscheinlich etwas grundlegend falsch gelaufen ist.
        low_element_penalty = (10 - total_nodes) * 5.0 
        score -= low_element_penalty
        logging.warning(f"Applying heavy penalty of {low_element_penalty} due to very low element count ({total_nodes}).")
    # --- ENDE DER NEUEN LOGIK ---

    final_score = max(0.0, round(score, 2))
    
    logging.info(f"Final calculated internal quality score: {final_score}")
    return final_score

# =============================================================================
#  Phase 4: Visual Debugger Graph Generation
# =============================================================================

def abstract_network_for_cgm(
    data: Dict[str, Any],
    logger: logging.Logger,
    type_aliases: Dict[str, List[str]],
    relevant_types: Set[str]
) -> Dict[str, Any]:
    """
    Transforms detailed P&ID data into a high-level CGM, now abstracting
    entire system groups into single nodes for ultimate clarity.
    """
    logger.info("Starting ADVANCED network abstraction with SYSTEM GROUPING...")
    all_elements = data.get('elements', [])
    connections = data.get('connections', [])

    if not all_elements or not connections:
        return {"connectors": []}

    # KORREKTUR: Filtere Elemente basierend auf der relevant_types-Liste, bevor der Graph gebaut wird.
    elements = [
        el for el in all_elements 
        if el.get('type') in relevant_types
    ]
    logger.info(f"Von {len(all_elements)} Elementen sind {len(elements)} für das CGM relevant und werden verarbeitet.")

    element_map = {el['id']: el for el in elements}
    
    # Erstelle einen Graphen, der sowohl Elemente als auch System-Gruppen kennt
    G = nx.DiGraph()
    system_groups = {} # Speichert, welche Elemente zu welcher Gruppe gehören

    for el in elements:
        G.add_node(el['id'])
        if group_name := el.get('system_group'):
            if group_name not in G:
                G.add_node(group_name) # Füge die Gruppe als eigenen Knoten hinzu
            system_groups[el['id']] = group_name

    for conn in connections:
        from_id, to_id = conn['from_id'], conn['to_id']
        
        # Finde den wahren Start- und Endpunkt (entweder das Element selbst oder seine Gruppe)
        source_node = system_groups.get(from_id, from_id)
        target_node = system_groups.get(to_id, to_id)

        if source_node != target_node and not G.has_edge(source_node, target_node):
            G.add_edge(source_node, target_node)

    # Entferne alle einzelnen Elemente, die jetzt Teil einer Gruppe sind
    nodes_to_remove = [el_id for el_id, group in system_groups.items()]
    G.remove_nodes_from(nodes_to_remove)

    # Ab hier ist die Logik zur Erstellung der Connectors dieselbe, aber auf dem abstrahierten Graphen
    abstract_connectors = []
    for from_node, to_node in G.edges():
        abstract_connectors.append({
            "name": f"Conn_{from_node}_to_{to_node}",
            "from_converter_ports": [{"unit_name": from_node, "port": "Out"}],
            "to_converter_ports": [{"unit_name": to_node, "port": "In"}]
        })

    logger.info(f"Network abstracted to {len(abstract_connectors)} high-level connectors.")
    return {"connectors": abstract_connectors}

def export_full_analysis_debug_image(
    original_image_path: str,
    output_path: str,
    analysis_data: Dict[str, Any],
    coarse_graph_data: Dict[str, Any],
    hotspot_zones: List[Dict],
    excluded_zones: List[Dict],
    metadata: Dict[str, Any],
    logger: logging.Logger,
    config_params: Dict[str, Any]
):
    """
    Creates a comprehensive, beautiful debug image that visualizes all stages 
    of the analysis pipeline with advanced, type-specific symbols and a clear legend.
    """
    logger.info(f"Generating full analysis debug image to: {output_path}")
    try:
        img = Image.open(original_image_path).convert("RGB")
        img_width, img_height = img.size
        
        dpi = config_params.get("debug_image_dpi", 150)
        fig_width, fig_height = img_width / dpi, img_height / dpi
        fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
        
        ax.imshow(img, extent=(0, img_width, img_height, 0))
        ax.set_xlim(0, img_width)
        ax.set_ylim(img_height, 0)
        ax.axis('off')

        # --- Zeichne Analyse-Ebenen (Phasen 1 & 2) ---
        for zone in excluded_zones:
            if zone:
                rect = mpatches.Rectangle((zone['x']*img_width, zone['y']*img_height), zone['width']*img_width, zone['height']*img_height,
                                          fill=True, facecolor='gray', alpha=0.4, hatch='//', zorder=10)
                ax.add_patch(rect)
        for zone in hotspot_zones:
            if zone:
                rect = mpatches.Rectangle((zone['x']*img_width, zone['y']*img_height), zone['width']*img_width, zone['height']*img_height,
                                          fill=False, edgecolor='cyan', linewidth=3, linestyle='--', zorder=11)
                ax.add_patch(rect)
        for el in coarse_graph_data.get('elements', []):
            if bbox := el.get('bbox'):
                rect = mpatches.Rectangle((bbox['x']*img_width, bbox['y']*img_height), bbox['width']*img_width, bbox['height']*img_height,
                                          fill=False, edgecolor='orange', linewidth=3, linestyle='-', zorder=12)
                ax.add_patch(rect)

        # --- Zeichne finale, fusionierte Komponenten (Phase 3) ---
        final_elements = {el['id']: el for el in analysis_data.get('elements', [])}
        label_font_size = max(6, config_params.get("debug_image_label_font_base_size", 9) - (len(final_elements) // config_params.get("debug_image_label_reduce_factor", 20)))

        for el_id, el in final_elements.items():
            if not (bbox := el.get('bbox')): continue
            
            el_type = el.get('type', 'Unknown')
            x, y, w, h = bbox['x']*img_width, bbox['y']*img_height, bbox['width']*img_width, bbox['height']*img_height
            center_x, center_y = x + w/2, y + h/2

            facecolor = 'lightgreen' # Default for Fused Component
            if 'Pump' in el_type: facecolor = 'lightcoral'
            elif 'Valve' in el_type: facecolor = 'lightsalmon' # Different color for distinction
            elif 'Sensor' in el_type or 'Indicator' in el_type: facecolor = 'gold'
            elif 'Tank' in el_type or 'Storage' in el_type: facecolor = 'lightsteelblue'
            elif 'Mixer' in el_type or 'Reactor' in el_type: facecolor = 'lightblue'
            
            rect = mpatches.Rectangle((x, y), w, h, fill=True, facecolor=facecolor, alpha=0.8, edgecolor='black', zorder=13)
            ax.add_patch(rect)
            
            display_label = textwrap.fill(el.get('label', el_id), width=15)
            ax.text(center_x, center_y, f"{display_label}\n({el_type})", color='black',
                    ha='center', va='center', fontsize=label_font_size, weight='bold', zorder=15,
                    bbox=dict(boxstyle="round,pad=0.3", fc='white', ec='none', alpha=0.5))

        # --- Zeichne finale Verbindungen ---
        for conn in analysis_data.get('connections', []):
            from_el, to_el = final_elements.get(conn['from_id']), final_elements.get(conn['to_id'])
            if not (from_el and to_el and from_el.get('bbox') and to_el.get('bbox')): continue

            start_bbox, end_bbox = from_el['bbox'], to_el['bbox']
            start_pos = (start_bbox['x'] + start_bbox['width']/2) * img_width, (start_bbox['y'] + start_bbox['height']/2) * img_height
            end_pos = (end_bbox['x'] + end_bbox['width']/2) * img_width, (end_bbox['y'] + end_bbox['height']/2) * img_height
            
            line_color = 'blue'
            line_style = '-'
            if conn.get('predicted'):
                line_color = 'purple'
                line_style = '-.'
            
            arrow = mpatches.FancyArrowPatch(start_pos, end_pos, 
                                             mutation_scale=config_params.get("debug_image_arrow_mutation_scale", 20), 
                                             arrowstyle='->', color=line_color, linewidth=2.5, linestyle=line_style,
                                             shrinkA=config_params.get("debug_image_arrow_shrink_px", 15), 
                                             shrinkB=config_params.get("debug_image_arrow_shrink_px", 15), zorder=14)
            ax.add_patch(arrow)

        # --- Titel und Legende ---
        plt.title(f"Full Analysis Debug View: {metadata.get('title', 'N/A')}\n(Project: {metadata.get('project', 'Unknown Project')})", 
                  size=config_params.get("debug_image_title_size", 22), pad=20)

        legend_elements = [
            mpatches.Patch(facecolor='gray', alpha=0.4, hatch='//', label='Phase 1: Excluded Zone'),
            mpatches.Patch(edgecolor='cyan', facecolor='none', lw=3, ls='--', label='Phase 2b: Hotspot Zone'),
            mpatches.Patch(edgecolor='orange', facecolor='none', lw=3, label='Phase 2a: Coarse Component'),
            mpatches.Patch(facecolor='lightgreen', alpha=0.8, ec='black', label='Phase 3: Final Component'),
            Line2D([0], [0], color='blue', lw=2.5, label='Final Connection (Detected)'),
            Line2D([0], [0], color='purple', lw=2.5, ls='-.', label='Final Connection (Predicted/Fused)')
        ]
        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0.01, 0.99),
                  fontsize='large', title='Analysis Layers', title_fontsize='x-large', 
                  frameon=True, fancybox=True, facecolor='white', framealpha=0.8)

        plt.tight_layout()
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Full analysis debug image successfully saved to {output_path}")

    except Exception as e:
        logger.error(f"Failed to generate full analysis debug image: {e}", exc_info=True)

def draw_cgm_graph(cgm_data: Dict, output_path: str, logger: logging.Logger, config_params: Dict[str, Any]): # <- NEU: config_params hinzugefügt
    """
    Draws a simple, abstract topological graph from CGM network data.
    """
    logger.info(f"Drawing abstract CGM graph to: {output_path}")
    if not cgm_data or not cgm_data.get('connectors'):
        logger.warning("No CGM connectors available to draw graph.")
        return

    import matplotlib.pyplot as plt
    import networkx as nx

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(20, 16))
    G = nx.DiGraph()

    all_units = set()
    for connector in cgm_data.get('connectors', []):
        for port in connector.get('from_converter_ports', []):
            if port.get('unit_name'):
                all_units.add(port.get('unit_name'))
        for port in connector.get('to_converter_ports', []):
            if port.get('unit_name'):
                all_units.add(port.get('unit_name'))
    
    for unit in all_units:
        G.add_node(unit)

    for connector in cgm_data.get('connectors', []):
        from_units = [p.get('unit_name') for p in connector.get('from_converter_ports', []) if p.get('unit_name')]
        to_units = [p.get('unit_name') for p in connector.get('to_converter_ports', []) if p.get('unit_name')]
        for from_u in from_units:
            for to_u in to_units:
                if G.has_node(from_u) and G.has_node(to_u):
                    G.add_edge(from_u, to_u)

    # Lese Plot-Parameter aus Konfiguration
    node_size = config_params.get("cgm_plot_node_size", 2000) # <- Nutzt config_params
    font_size = config_params.get("cgm_plot_font_size", 8) # <- Nutzt config_params
    edge_width = config_params.get("cgm_plot_edge_width", 1.5) # <- Nutzt config_params
    arrow_size = config_params.get("cgm_plot_arrow_size", 20) # <- Nutzt config_params
    layout_k = config_params.get("cgm_plot_layout_k", 0.6) # <- Nutzt config_params
    layout_iterations = config_params.get("cgm_plot_layout_iterations", 50) # <- Nutzt config_params

    pos = nx.spring_layout(G, k=layout_k, iterations=layout_iterations, seed=42)
    nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color='skyblue', alpha=0.9)
    nx.draw_networkx_labels(G, pos, font_size=font_size, font_weight='bold')
    nx.draw_networkx_edges(G, pos, edgelist=G.edges(),
                           edge_color='gray',
                           width=edge_width,
                           arrows=True,
                           arrowstyle='->',
                           arrowsize=arrow_size)
    
    ax.margins(0.1)
    title_font_size = config_params.get("kpi_plot_title_size", 22) # <- Nutzt config_params
    plt.title("Abstract CGM Network Topology", size=title_font_size, pad=20)
    plt.axis('off')
    plt.tight_layout()
    cgm_plot_dpi = config_params.get("cgm_plot_dpi", 300) # <- Nutzt config_params
    plt.savefig(output_path, dpi=cgm_plot_dpi, bbox_inches='tight')
    plt.close(fig)
    logger.info("Abstract CGM graph successfully saved.")
