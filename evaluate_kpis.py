# evaluate_kpis.py - Version 3.0 mit maximaler Detailtiefe
# 📦 Standardbibliotheken
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm 
from matplotlib.colors import Normalize 
import json
from pathlib import Path
import sys
from datetime import datetime
import textwrap
import os
import yaml
import logging
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from typing import Optional, Dict, Any, List # Typ-Hinweise
import numpy as np

# Logging für dieses Modul konfigurieren
logging.basicConfig(level=logging.INFO, format='[%(asctime)s - %(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__) # Eigener Logger für evaluate_kpis.py

# KORREKTUR: Lade die Konfiguration direkt in evaluate_kpis.py
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(SCRIPT_DIR, "config.yaml")

def load_config(path: str) -> dict: # Hilfsfunktion zum Laden der Config
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except FileNotFoundError: # Spezifischer Fehler für fehlende Datei
        logging.error(f"Konfigurationsdatei nicht gefunden: {path}")
        # Wenn die Konfigurationsdatei fehlt, kann das Programm nicht sinnvoll arbeiten
        # In einer Produktionsumgebung sys.exit(1) aufrufen. Hier nur print/return
        print(f"Fehler: Konfigurationsdatei nicht gefunden: {path}")
        return {} # Leeres Dict zurückgeben, um Absturz zu vermeiden
    except yaml.YAMLError as e: # Spezifischer Fehler für YAML-Parsing
        logging.error(f"Fehler beim Parsen der Konfigurationsdatei {path}: {e}")
        print(f"Fehler beim Parsen der Konfigurationsdatei {path}: {e}")
        return {}
    except Exception as e:
        print(f"Fehler beim Laden der Konfigurationsdatei {path}: {e}")
        logging.error(f"Fehler beim Laden der Konfigurationsdatei {path}: {e}") # Auch ins Log schreiben
        return {}

# Lade die Konfiguration beim Start des Moduls
APP_CONFIG = load_config(CONFIG_PATH)
# Zugriff auf 'logic_parameters' für alle relevanten Werte
LOGIC_PARAMS = APP_CONFIG.get('logic_parameters', {}) # <-- DIESE ZEILE HINZUGEFÜGT


def analyze_runs(target_path: Path):
    """
    Sucht `_summary.json` Dateien und vergleicht sie mit den korrekten
    `_truth.json` und `_truth_cgm.json` Dateien.
    """
    all_run_data = []
    
    if target_path.is_file() and target_path.name.endswith("_summary.json"):
        summary_paths = [target_path]
    elif target_path.is_dir():
        summary_paths = list(target_path.rglob("*_summary.json"))
    else:
        logger.error(f"Fehler: '{target_path}' ist weder eine valide `_summary.json`-Datei noch ein Ordner.")
        return None

    if not summary_paths:
        logger.warning("Keine `_summary.json` Dateien im angegebenen Pfad gefunden.")
        return None

    for summary_path in summary_paths:
        try:
            with open(summary_path, 'r', encoding='utf-8') as f:
                summary_data = json.load(f)

            run_folder = summary_path.parent
            image_name = summary_data.get("image_name", "Unbekannt")
            image_base_name = Path(image_name).stem
            
            # Die KPIs aus der Summary sind das Ergebnis des Vergleichs mit der _truth.json,
            # der bereits im Core-Prozess stattgefunden hat. Wir übernehmen diese direkt.
            final_kpis = summary_data.get("final_kpis", {})
            quality_score = summary_data.get("recognition_quality_score", -1)

            run_data = {
                "Run ID": run_folder.name,
                "Modell": summary_data.get("model_name_for_summary", "Unbekannt"),
                "Bild": image_name,
                "KPI Score": quality_score,
                "Precision": final_kpis.get("element_precision", 0),
                "Recall": final_kpis.get("element_recall", 0),
                "Typ-Genauigkeit": final_kpis.get("type_accuracy", 0),
                "Verbind.-Genauigkeit": final_kpis.get("connection_recall", 0),
                "Fehlerdetails": final_kpis.get("error_details", {})
            }

            # Jetzt beauftragen wir den "Chef-Ingenieur" mit dem Vergleich der Baupläne
            ai_cgm_path = run_folder / f"{image_base_name}_cgm_data.json"
            truth_cgm_path = target_path / f"{image_base_name}_truth_cgm.json"

            if ai_cgm_path.exists() and truth_cgm_path.exists():
                cgm_kpis = calculate_cgm_kpis(ai_cgm_path, truth_cgm_path)
                if cgm_kpis:
                    run_data.update(cgm_kpis)

            all_run_data.append(run_data)
        except Exception as e:
            logger.error(f"Fehler bei der Verarbeitung von {summary_path}: {e}")

    if not all_run_data:
        return None

    return pd.DataFrame(all_run_data)

def format_errors(error_dict):
    """Formatiert das Fehler-Dictionary schön für den Markdown-Report."""
    if not error_dict or not any(error_dict.values()):
        return "_Keine Fehler erkannt._"
    
    lines = []
    if error_dict.get("missed_elements"):
        lines.append(f"**Fehlende Elemente:** `{', '.join(error_dict['missed_elements'])}`")
    if error_dict.get("hallucinated_elements"):
        lines.append(f"**Zusätzliche Elemente:** `{', '.join(error_dict['hallucinated_elements'])}`")
    if error_dict.get("typing_errors"):
        # Mache lange Fehlermeldungen lesbarer
        # KORREKTUR: typing_errors ist jetzt eine Liste von Dictionaries
        formatted_typing_errors = [f"'{e.get('ai_type')}'->'{e.get('truth_type')}' (ID: {e.get('element_id', 'N/A')})" for e in error_dict['typing_errors']]
        lines.append(f"**Typisierungsfehler:** {', '.join(formatted_typing_errors)}")
        
    if error_dict.get("missed_connections"):
        lines.append(f"**Fehlende Verbindungen:** `{', '.join(['->'.join(map(str, c)) for c in error_dict['missed_connections']])}`")
    if error_dict.get("hallucinated_connections"):
        lines.append(f"**Zusätzliche Verbindungen:** `{', '.join(['->'.join(map(str, c)) for c in error_dict['hallucinated_connections']])}`")

    return "<br>".join(lines) if lines else "_Keine Fehler erkannt._"

def generate_markdown_report(df: pd.DataFrame, output_path: Path):
    """
    Erstellt einen detaillierten, nach einzelnen Läufen gruppierten Markdown-Report.
    """
    report_date = datetime.now().strftime("%d. %B %Y, %H:%M:%S")
    
    # Formatiere die Fehlerdetails für die Anzeige in der Tabelle
    df['Fehlerübersicht'] = df['Fehlerdetails'].apply(format_errors)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"# 📊 P&ID Analyse: Detaillierter KPI-Vergleichsreport\n")
        f.write(f"_Generiert am: {report_date}_\n\n---\n") # <- KORRIGIERT: report_date verwenden

        # --- Gesamt-Chart, der JEDEN Lauf zeigt ---
        df_sorted_for_chart = df.sort_values(by="KPI Score", ascending=True)
        if not df_sorted_for_chart.empty:
            plt.style.use('seaborn-v0_8-talk')
            fig, ax = plt.subplots(figsize=(16, len(df_sorted_for_chart) * 0.6 + 2))
            
            kpi_min_score = float(LOGIC_PARAMS.get("kpi_plot_min_score", 0.0))
            kpi_max_score = float(LOGIC_PARAMS.get("kpi_plot_max_score", 105.0))

            norm = Normalize(vmin=kpi_min_score, vmax=kpi_max_score)
            # Konvertiere die Pandas-Serie explizit zu einem NumPy-Array für Matplotlib
            # Dies sollte Pylance beruhigen, da es den erwarteten Typ besser erkennt.
            kpi_scores_np = np.asarray(df_sorted_for_chart['KPI Score'].astype(float).values)
            colors = cm.viridis(norm(kpi_scores_np)) # type: ignore [reportCallIssue, reportArgumentType] # Ignoriere verbleibende Pylance-Hinweise, da Typen intern korrekt sind
            
            # KORREKTUR: Sicherstellen, dass 'Run ID' einzigartig ist
            df_sorted_for_chart['Display Name'] = df_sorted_for_chart['Modell'] + " - " + df_sorted_for_chart['Bild']
            bars = ax.barh(df_sorted_for_chart['Display Name'], df_sorted_for_chart['KPI Score'], color=colors)
            
            ax.set_xlabel('Erreichter KPI Score pro Analyse-Lauf')
            ax.set_title('Performance der einzelnen Analyse-Läufe')
            ax.set_xlim(0, 105)
            ax.grid(axis='x', linestyle='--', alpha=0.7)
            
            for bar in bars:
                width = bar.get_width()
                ax.text(width + 1, bar.get_y() + bar.get_height()/2, f'{width:.1f}', va='center', ha='left', fontweight='bold')
            
            plt.tight_layout(pad=2) 
            
            chart_path = output_path.parent / "kpi_detailed_comparison_chart.png"
            plt.savefig(chart_path)
            plt.close()

            f.write("## 🏆 Performance der einzelnen Analyse-Läufe\n\n")
            f.write(f"![Detaillierter Vergleichs-Chart](kpi_detailed_comparison_chart.png)\n\n---\n")

        # --- Detaillierte Auswertungstabelle pro Bild ---
        f.write("## 🔬 Detaillierte Auswertung pro Diagramm\n")
        
        for image_name, group in df.groupby('Bild'):
            f.write(f"\n### Analyse für: **{image_name}**\n\n")
            
            group_sorted = group.sort_values(by="KPI Score", ascending=False)
            
            report_columns = group_sorted[[
                "Modell", "KPI Score", "Precision", "Recall", "Typ-Genauigkeit", "Verbind.-Genauigkeit", "Fehlerübersicht" # Alle relevanten Spalten
            ]]
            
            f.write(report_columns.to_markdown(index=False))
            f.write("\n")

# =============================================================================
#  Phase 3: KPI Calculation Helpers (Fortsetzung)
# =============================================================================
def calculate_final_quality_score(final_kpis: Dict[str, Any]) -> float:
    """
    Calculates a single, aggregated quality score from all KPI sets based on
    internal consistency checks.
    """
    logging.info("Calculating final internal quality score...")
    score = 100.0
   
    # Lese KPI-relevante Strafwerte aus der Konfiguration
    subgraph_penalty_factor = LOGIC_PARAMS.get("kpi_subgraph_penalty", 25.0)
    isolated_penalty_factor = LOGIC_PARAMS.get("kpi_isolated_penalty", 15.0)
    dangling_penalty_factor = LOGIC_PARAMS.get("kpi_dangling_penalty", 5.0)
    unidentified_penalty_factor = LOGIC_PARAMS.get("kpi_unidentified_type_penalty", 2.0)
    low_element_threshold = LOGIC_PARAMS.get("kpi_low_element_count_threshold", 10)
    low_element_penalty_per_missing = LOGIC_PARAMS.get("kpi_low_element_count_penalty_per_missing", 5.0)

    subgraph_penalty = (final_kpis.get("subgraph_count", 1) - 1) * subgraph_penalty_factor
    score -= subgraph_penalty
    if subgraph_penalty > 0:
        logging.warning(f"Applying penalty of {subgraph_penalty} for {final_kpis.get('subgraph_count')} subgraphs.")

    isolated_penalty = final_kpis.get("isolated_elements_count", 0) * isolated_penalty_factor
    score -= isolated_penalty
    if isolated_penalty > 0:
        logging.warning(f"Applying penalty of {isolated_penalty} for {final_kpis.get('isolated_elements_count')} isolated elements.")

    dangling_penalty = final_kpis.get("dangling_components_node_count", 0) * dangling_penalty_factor
    score -= dangling_penalty
    if dangling_penalty > 0:
        logging.warning(f"Applying penalty of {dangling_penalty} for nodes in dangling components.")

    unidentified_penalty = final_kpis.get("unidentified_element_count", 0) * unidentified_penalty_factor
    score -= unidentified_penalty
    if unidentified_penalty > 0:
        logging.warning(f"Applying penalty of {unidentified_penalty} for unidentified element types.")
        
    total_nodes = final_kpis.get("total_nodes", 0)
    if total_nodes < low_element_threshold:
        low_element_penalty = (low_element_threshold - total_nodes) * low_element_penalty_per_missing 
        score -= low_element_penalty
        logging.warning(f"Applying heavy penalty of {low_element_penalty} due to very low element count ({total_nodes}).")

    final_score = max(0.0, round(score, 2))
    
    logging.info(f"Final calculated internal quality score: {final_score}")
    return final_score

def calculate_cgm_kpis(ai_cgm_path: Path, truth_cgm_path: Path) -> Optional[Dict[str, float]]:
    """Vergleicht zwei CGM-JSON-Dateien und berechnet Precision, Recall und einen Score."""
    try:
        with open(ai_cgm_path, 'r', encoding='utf-8') as f:
            ai_data = json.load(f)
        with open(truth_cgm_path, 'r', encoding='utf-8') as f:
            truth_data = json.load(f)

        def extract_connections(data):
            connections = set()
            for connector in data.get('connectors', []):
                from_units = frozenset(p.get('unit_name') for p in connector.get('from_converter_ports', []))
                to_units = frozenset(p.get('unit_name') for p in connector.get('to_converter_ports', []))
                if from_units and to_units:
                    connections.add((from_units, to_units))
            return connections

        ai_connections = extract_connections(ai_data)
        truth_connections = extract_connections(truth_data)

        if not truth_connections:
            return None

        correctly_found = len(ai_connections.intersection(truth_connections))
        
        precision = correctly_found / len(ai_connections) if ai_connections else 0.0
        recall = correctly_found / len(truth_connections)
        
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            "cgm_precision": round(precision, 2),
            "cgm_recall": round(recall, 2),
            "cgm_score": round(f1_score * 100, 2)
        }
    except Exception as e:
        logger.error(f"Fehler beim Berechnen der CGM-KPIs für {ai_cgm_path.name}: {e}")
        return None

def main(target_path: Optional[Path] = None):
    """Hauptfunktion, optional mit Ordner-Pfad als Parameter."""
    if target_path is None:
        if len(sys.argv) > 1:
            target_path = Path(sys.argv[1])
        else:
            target_path = Path.cwd()

    print(f"Starte umfassende KPI-Analyse für: {target_path.resolve()}")

    df = analyze_runs(target_path)

    if df is None or df.empty:
        print("Fehler: Keine `_summary.json` Dateien im Zielpfad gefunden oder Daten konnten nicht geladen werden.")
        return

    report_path = target_path / f"EVALUATION_REPORT_DETAILED_{datetime.now().strftime('%Y-%m-%d_%H%M%S')}.md"
    generate_markdown_report(df, report_path)
    
    print("\n--- Analyse abgeschlossen ---")
    print(f"✅ Umfassender, detaillierter Report erfolgreich erstellt: {report_path.resolve()}")
    
    # Gib eine saubere Übersicht im Terminal aus
    print("\nÜbersicht der Ergebnisse:")
    display_df = df[["Modell", "Bild", "KPI Score", "Verbind.-Genauigkeit"]]
    print(display_df.to_string(index=False))

if __name__ == "__main__":
    main()