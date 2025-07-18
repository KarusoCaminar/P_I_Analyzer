# Agents und Tools in diesem Repository

Dieses Repository enthält mehrere Schlüsselkomponenten, die als "Agents" oder "Tools" betrachtet werden können, da sie spezifische Aufgaben in der P&ID-Analyse-Pipeline ausführen. Jules kann diese Beschreibungen nutzen, um die Funktionsweise des Codes besser zu verstehen und relevantere Pläne oder Code-Vervollständigungen zu generieren.

---

## 1. `cli_runner.py` (CLI Runner Agent)

### Beschreibung
Der `cli_runner.py` ist der Kommandozeilen-Einstiegspunkt der Anwendung. Er ist für das Parsen von Befehlszeilenargumenten, das Laden der Konfiguration und das Initialisieren des `CoreProcessor` verantwortlich, um eine einzelne P&ID-Analyse durchzuführen. Er agiert als Orchestrator für einen einzelnen Analyse-Lauf.

### Funktionen
- **Konfigurationsmanagement:** Lädt und validiert die Anwendungskonfiguration.
- **Argumenten-Parsing:** Verarbeitet Kommandozeilenargumente wie `model_id`, `strategy_name`, `image_path` und `output_dir`.
- **Analyse-Start:** Initialisiert den `CoreProcessor` mit der entsprechenden Konfiguration und startet den Analyseprozess.
- **Fehlerbehandlung:** Fängt kritische Fehler ab und beendet das Programm ordnungsgemäß.

### Interaktion
Wird direkt über die Kommandozeile aufgerufen, z.B. `python cli_runner.py --image_path <path_to_image> --model_id <model>`.

---

## 2. `core_processor.py` (Core Analysis Processor)

### Beschreibung
Der `core_processor.py` ist das Herzstück der P&ID-Analyse. Er implementiert die iterative Analyse- und Selbstkorrekturlogik, indem er visuelle Informationen mit LLM-Interaktionen kombiniert. Er ist verantwortlich für die Extraktion von Elementen, die Beziehungserkennung und die Validierung der Ergebnisse.

### Funktionen
- **Initialisierung:** Lädt LLM-Handler, Knowledge Manager und Konfigurationen.
- **Iterative Analyse:** Führt eine mehrstufige Analyse durch, die LLM-Aufrufe, Bildverarbeitung und logische Validierung umfasst.
- **Selbstkorrektur (Critic-Agent):** Nutzt einen "Critic-Agenten" (LLM), um Probleme in den Analyseergebnissen zu identifizieren und Korrekturen vorzuschlagen.
- **Fehlerbehandlung:** Verarbeitet spezifische Fehler während der Analyse und versucht, diese zu beheben.
- **Bildverarbeitung:** Schneidet Bildausschnitte zu, führt BBox-Verfeinerungen durch und berechnet Metriken.
- **Post-Processing:** Fügt Symbole und Texte zusammen, dedupliziert Elemente und Ports.

### Interaktion
Wird intern vom `cli_runner.py` oder dem `orchestrator.py` instanziiert und gesteuert.

---

## 3. `llm_handler.py` (LLM Interaction Handler)

### Beschreibung
Der `llm_handler.py` ist die zentrale Schnittstelle zur Interaktion mit Large Language Models (LLMs), insbesondere für Text- und multimodale (Bild + Text) Prompts. Er kapselt die API-Aufrufe, implementiert Caching und Retry-Logik.

### Funktionen
- **LLM-Aufrufe:** Sendet Prompts an das konfigurierte LLM (z.B. Gemini).
- **Multimodale Prompts:** Unterstützt die Übertragung von Bildern zusammen mit Textprompts.
- **Disk-Caching:** Nutzt einen Festplatten-Cache, um redundante LLM-Aufrufe zu vermeiden und Kosten zu sparen.
- **Retry-Logik:** Implementiert exponentielles Backoff für fehlgeschlagene API-Aufrufe.
- **JSON-Parsing und Data Rescue:** Versucht, JSON-Antworten auch bei Fehlern oder Inkonsistenzen zu parsen und zu "retten".
- **Embedding-Generierung:** Generiert visuelle Embeddings für Bilder.
- **Sicherheits-Einstellungen:** Deaktiviert Safety Settings für technische Diagramme, um Blockaden zu vermeiden.

### Interaktion
Wird intern von `core_processor.py` und `knowledge_bases.py` genutzt, um mit LLMs zu kommunizieren.

---

## 4. `knowledge_bases.py` (Knowledge Management Agent)

### Beschreibung
Der `knowledge_bases.py` verwaltet das Wissen der Anwendung, einschließlich der Symbolbibliothek, der gelernten Korrekturmuster und der Aliasnamen. Er ist entscheidend für die Verbesserung der Analysegenauigkeit über die Zeit durch maschinelles Lernen und Problemlösung.

### Funktionen
- **Symbolbibliothek:** Speichert und verwaltet Informationen über bekannte P&ID-Symbole, einschließlich ihrer visuellen Embeddings.
- **Lern-Datenbank:** Speichert erfolgreiche Korrekturmuster und gelernte Lösungen für wiederkehrende Probleme.
- **Vektorsuche:** Ermöglicht die Suche nach ähnlichen Symbolen oder Problemlösungen basierend auf ihren Embeddings.
- **Konsistenzprüfung:** Normalisiert und bereinigt die Wissensbasis.
- **Alias-Management:** Verarbeitet und löst Aliasnamen für Elementtypen auf.

### Interaktion
Wird von `core_processor.py` verwendet, um auf Symbolinformationen zuzugreifen und gelernte Korrekturen anzuwenden. Wird auch vom `orchestrator.py` im Vortraining-Schritt verwendet.

---

## 5. `orchestrator.py` (Training Orchestrator)

### Beschreibung
Der `orchestrator.py` ist ein übergeordneter Agent, der für das Training und die iterative Optimierung der P&ID-Analyse-Pipeline verantwortlich ist. Er führt mehrere Analyseläufe mit unterschiedlichen Konfigurationen durch, bewertet die Ergebnisse und identifiziert die besten Einstellungen.

### Funktionen
- **Phasenbasiertes Training:** Führt das Training in verschiedenen Phasen durch (z.B. Symbol-Vortraining, inkrementelles Lernen).
- **Parameter-Tuning:** Generiert und testet verschiedene Kombinationen von Konfigurationsparametern.
- **Ergebnisbewertung:** Analysiert die KPIs der einzelnen Läufe und identifiziert die beste Konfiguration.
- **Konfigurationsaktualisierung:** Speichert die besten gefundenen Konfigurationen in der `config.yaml`.
- **Subprozess-Management:** Startet `cli_runner.py`-Instanzen als separate Prozesse.

### Interaktion
Wird typischerweise für längere Trainings- oder Optimierungsläufe verwendet und steuert die Gesamtpipeline.

---

## 6. `evaluate_kpis.py` (KPI Evaluation Tool)

### Beschreibung
Der `evaluate_kpis.py` ist ein Skript zum Auswerten und Visualisieren der Key Performance Indicators (KPIs) der Analyseergebnisse. Es aggregiert Daten aus mehreren Analyse-Läufen und generiert Berichte zur Leistungsbewertung.

### Funktionen
- **KPI-Berechnung:** Berechnet den Gesamtqualitäts-Score basierend auf verschiedenen Fehlertypen.
- **Datenaggregation:** Liest und konsolidiert Analyse-Zusammenfassungsdateien.
- **Berichtsgenerierung:** Erzeugt textbasierte und grafische Berichte (z.B. mit Matplotlib) zur Visualisierung der Performance.

### Interaktion
Kann nach mehreren Analyse-Läufen ausgeführt werden, um deren Ergebnisse zu vergleichen und zu bewerten.

---

## 7. `gui.py` (Graphical User Interface)

### Beschreibung
Die `gui.py` stellt eine grafische Benutzeroberfläche bereit, um die Interaktion mit der P&ID-Analyse-Pipeline zu vereinfachen. Sie ermöglicht es Benutzern, Konfigurationen anzupassen, Analysen zu starten und den Fortschritt zu verfolgen.

### Funktionen
- **Benutzerinteraktion:** Bietet Steuerelemente zum Auswählen von Bildern, Modellen und Konfigurationen.
- **Statusanzeige:** Zeigt den aktuellen Status der Analyse und Log-Meldungen an.
- **Konfigurationsanpassung:** Ermöglicht die dynamische Anpassung bestimmter Analyseparameter.
- **Thread-Management:** Nutzt Threads und Queues, um die GUI während langlaufender Operationen reaktionsfähig zu halten.
- **Ergebnisvisualisierung:** Kann Analyseergebnisse und Debug-Visualisierungen anzeigen.

### Interaktion
Wird als separate Anwendung gestartet und bietet eine visuelle Schnittstelle zur Steuerung des Systems.

---

## Wichtige gemeinsame Konventionen und Muster:

- **Konfigurationsdateien:** Die meisten Komponenten greifen auf eine zentrale `config.yaml` zu. Änderungen an dieser Datei können das Verhalten der Agents stark beeinflussen.
- **Logging:** Die Anwendung nutzt ein konsistentes Logging-Setup für die Nachvollziehbarkeit.
- **Type Hinting:** Um die Code-Klarheit und statische Analyse zu verbessern, werden umfangreiche Typ-Hinweise verwendet.
- **Fehlerbehandlung:** Robuste `try-except`-Blöcke und Retry-Mechanismen sind weit verbreitet, um Ausfälle zu minimieren.
- **Temporäre Dateien:** Bestimmte Operationen erzeugen temporäre Dateien für Zwischenergebnisse (z.B. Bildsegmente, temporäre Konfigurationen).
