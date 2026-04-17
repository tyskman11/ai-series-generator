# KI Serien Training

## Zweck

Dieses Projekt baut aus vorhandenen Serienfolgen eine lokale KI-Trainings- und Generierungs-Pipeline.

Von einer echten Folge als `mp4` geht es bis zu:

- importierten und geschnittenen Szenen
- transkribierten Dialogen und Sprecher-Clustern
- erkannten Gesichtern und verknuepften Figuren
- einem lokalen Serien-Datensatz und Modell
- einer neu generierten Folge als Text und Shotlist
- einem renderbaren Storyboard-/TTS-Draft-Video
- optionalen Figur-Referenzen fuer spaetere Clone- und Preview-Schritte

Die Pipeline ist modular. Jeder Schritt kann einzeln gestartet werden, oder alles laeuft ueber `99_process_next_episode.py`.

## Auf Einen Blick

| Bereich | Stand | Kurz gesagt |
| --- | --- | --- |
| Import / Split | stabil | Folgen werden importiert, registriert und in Szenen zerlegt |
| Transkription / Sprecher | nutzbar | Whisper + Sprecher-Cluster laufen lokal, mit GPU-Unterstuetzung wenn vorhanden |
| Gesichter / Figuren | aktiv in Feintuning | wiederkehrende Figuren werden von Einmal-Gesichtern getrennt |
| Datensatz / Modell | nutzbar | aus verknuepften Szenen entsteht ein lokales Serienmodell |
| Generierung / Render | Preview-faehig | neue Folgen, Shotlists und Final-/Draft-Videos mit echten Episodentiteln koennen erzeugt werden |
| Voice / Clone-Pfad | optional | Standard bleibt lizenzfrei bei lokalem `pyttsx3`, XTTS nur bewusst als Opt-in |
| GitHub-Sync | aktiv | nur Root-Skripte + `README.md`, nur Upload/Spiegelung, keine Downloads |

## Was Gerade Schon Gut Funktioniert

- vorhandene Serienfolgen koennen als komplette lokale Pipeline verarbeitet werden
- die Pipeline kann neue Folgen als Markdown und Shotlist aus dem bisherigen Material ableiten
- neue Folgen tragen jetzt neben `folge_0x` auch einen echten Episodentitel wie `Folge 05: Das Baumhaus-Spiel`
- die Titelfindung zieht dafuer jetzt bevorzugt markante Begriffe aus echten Originaldialogen wie `Rollerteller` oder `Videospiele`, statt rohe Problemwoerter direkt in den Titel zu heben
- `16_render_episode.py` erzeugt ein visuelles Draft-Video statt nur Textartefakte
- die Standardausgabe bleibt bewusst lokal und lizenzarm, damit der Default-Workflow robust bleibt
- der GitHub-Sync spiegelt jetzt nur erlaubte lokale Dateien nach GitHub und holt nie Remote-Inhalte herunter
- lange Laeufe zeigen jetzt ein gemeinsames Live-Dashboard statt bloesser Prozentzaehler: Skript, Schritt, aktuelle Datei/Figur/Folge, Fortschrittsbalken, `Ende aktuell` und `Ende gesamt` werden direkt im Terminal laufend an derselben Stelle aktualisiert

## Wichtige Dokumentationsregel

Diese Datei ist ein Pflichtdokument.

Bei jeder Aenderung an:

- `00_prepare_runtime.py` bis `19_sync_to_github.py`
- `99_process_next_episode.py`
- `pipeline_common.py`
- `ai_series_project/configs/project.json`
- CLI-Optionen
- Ordnerstruktur
- Ausgabeformaten
- Environment-Variablen
- bekannten Einschraenkungen oder Workarounds

muss diese `README.md` im gleichen Arbeitsgang mit aktualisiert werden.

Wenn unklar ist, ob eine Aenderung dokumentiert werden muss, gilt: lieber README mit anpassen.

Zusatzregel ab jetzt:

- `README.md` muss immer auch die beiden Bereiche `Gerade in Bearbeitung` und `Geplant` aktuell halten
- wenn sich Prioritaeten aendern, muessen diese beiden Bereiche im gleichen Arbeitsgang mitgezogen werden

## Fokus Im Moment

- Hauptfiguren schneller von Hintergrundgesichtern trennen
- bereits bekannte Gesichter in `06_review_unknowns.py` vor der Review automatisch wiederfinden und zusammenfuehren
- den Standardpfad stabil, lokal und ohne Lizenzpflichten halten
- den lokalen Code-Stand sauber nach GitHub spiegeln, ohne jemals etwas herunterzuladen

## Gerade in Bearbeitung

- `06_review_unknowns.py`: Hauptfiguren muessen weiterhin manuell benannt werden; bekannte Gesichter werden davor jetzt aber mit einem robusteren Mehrfachreferenz-Matcher pro Figur wiedererkannt und zusammengefuehrt, damit fuer die Review wirklich nur noch Unbekannte uebrig bleiben
- `06_review_unknowns.py` gewichtet bekannte Figuren jetzt zusaetzlich nach Identitaetsstaerke: grosse, gut belegte Figuren duerfen minimal entspannter gematcht werden, schwache Identitaeten bleiben bewusst strenger
- `06_review_unknowns.py` lernt jetzt nach jeder manuellen Benennung sofort weiter: weitere offene Face-Cluster werden iterativ gegen die neu gewonnene Figur referenziert und offene Sprecher-Reviews mit genau einer sichtbaren benannten Figur werden konservativ direkt mitgezogen
- `06_review_unknowns.py` blendet fuer offene Cluster jetzt zusaetzlich automatische Rollenhinweise wie `hauptfigur-kandidat`, `nebenfigur-kandidat` oder `statist-kandidat` ein, damit Hauptfiguren schneller benannt und Nebenfiguren schneller als `statist` einsortiert werden koennen
- `04_diarize_and_transcribe.py` versucht verbleibende `speaker_unknown`-Segmente jetzt nach dem ersten Cluster-Pass noch einmal konservativ ueber Nachbarschaft und Embedding-Aehnlichkeit auf bestehende Sprecher zurueckzufuehren
- `04_diarize_and_transcribe.py` hat jetzt zusaetzlich einen zweiten folgeweiten Konsens-Pass: `speaker_unknown` wird nur dann auf eine bekannte Stimme gezogen, wenn Embedding und sprachliches Profil derselben Figur zusammenpassen
- das Zusammenspiel aus `04`-Sprecher-Clustern und `05`-Figurenfilter wird weiter auf kompletten Folgen beobachtet, besonders bei Nebenfiguren und den danach noch verbleibenden `speaker_unknown`
- der lizenzfreie Standardpfad bleibt Standard: lokales `pyttsx3` statt lizenzpflichtiger Voice-Wege
- die automatischen Platzhalternamen werden jetzt selbstheilend gehalten: unbenannte Figuren bleiben `face_###`, unbenannte Stimmen bleiben `speaker_###`, und manuell gesetztes `statist` bleibt bewusst eine Nebenfigur
- der aktuelle End-to-End-Stand wird ueber Tests und Smoke-Runs mitgefuehrt, damit Aenderungen nicht nur im Code, sondern auch im Ablauf verifiziert bleiben
- neue Folgen koennen jetzt gesammelt in einem Lauf als direkt sichtbare Draft-Episoden erzeugt werden
- `99_process_next_episode.py` wird gerade auf echten Inbox-Batch-Betrieb fuer viele neue Quellfolgen gleichzeitig gefestigt
- Inbox-Dateien werden jetzt bereits in `02_import_episode.py` direkt nach erfolgreicher Vollkopie nach `data/raw/episodes` entfernt; `99_process_next_episode.py` arbeitet dadurch immer nur noch mit den importierten Arbeitskopien weiter
- `99_process_next_episode.py` schreibt jetzt nach jedem erfolgreich abgeschlossenen Schritt einen Autosave-Checkpoint, behaelt davon nur die letzten zwei und setzt bei einem Abbruch spaeter automatisch am letzten sauberen Schritt fort
- `99_process_next_episode.py` schreibt jetzt zusaetzlich laufend `current_status.json` und `current_status.md` mit Episoden- und Globalstatus, damit grosse Inbox-Stapel pro Folge transparent beobachtet werden koennen
- `03`, `04`, `05`, `07`, `09`, `10`, `11`, `12`, `13`, `14`, `15`, `16`, `17`, `18` und `99` nutzen jetzt einen gemeinsamen Live-Reporter als mehrzeiliges Terminal-Dashboard statt unklarer Prozent-Ausgaben; die Ausgabe nennt immer Schritt, aktuelle Datei/Figur/Folge, Fortschrittsbalken, `Ende aktuell` und `Ende gesamt`
- `03_split_scenes.py` erkennt jetzt auch beim Neustart sauber, wenn eine Folge bereits erfolgreich in Szenen zerlegt wurde, und ueberspringt den Schritt dann idempotent statt blind erneut zu exportieren
- `03_split_scenes.py` arbeitet ohne `--episode-file` jetzt den kompletten aktuellen Stapel in `data/raw/episodes` nacheinander ab, statt nur eine einzelne Arbeitskopie
- `04_diarize_and_transcribe.py`, `05_link_faces_and_speakers.py` und `07_build_dataset.py` arbeiten ohne explizite Folge jetzt den kompletten offenen Stapel nacheinander ab, statt nur eine einzelne Folge zu nehmen oder immer wieder am selben Ordner haengenzubleiben
- `02_import_episode.py` bis `07_build_dataset.py` schreiben jetzt ebenfalls pro Folge Start-/Erfolgs-/Fehler-Autosaves unter `ai_series_project/runtime/autosaves/steps`
- `02_import_episode.py` kann jetzt optional den kompletten Inbox-Stapel in einem Lauf importieren, damit `03` danach alle bereits importierten Folgen nacheinander abarbeiten kann
- `04_diarize_and_transcribe.py`, `05_link_faces_and_speakers.py` und `07_build_dataset.py` bewerten einen Schritt jetzt erst dann als wirklich fertig, wenn aktuelle Prozessversion, Erfolgsmarker und Schritt-Autosave zusammenpassen
- `07_build_dataset.py` und `10_train_foundation_models.py` haben jetzt zusaetzlich `--force`, damit man nach manueller Figurenpflege oder Datenkorrekturen bewusst sauber neu aufbauen kann
- `18_refresh_after_manual_review.py` fuehrt jetzt den echten Rebuild nach manueller Figurenpflege als festen Kettenlauf aus: `07 --force -> 08 -> 09 --force -> 10 --force -> 11 --force -> 12 --force -> 13 --force -> 14 -> 15 -> 16`
- `17_generate_preview_episodes.py` und `18_refresh_after_manual_review.py` respektieren jetzt dieselbe Reihenfolge wie `99`: Solange in `06_review_unknowns.py` noch offene Review-Faelle liegen, starten weder Training noch Generierung noch Render; `18` erlaubt nur mit bewusstem `--allow-open-review` einen Override
- `11_train_adapter_models.py` trainiert jetzt lokale Adapter-Profile fuer Bild, Stimme und Clip-Dynamik und haengt sich standardmaessig zwischen `10` und `14` in die Trainingskette
- `12_train_fine_tune_models.py` trainiert jetzt darauf aufbauend lokale Fine-Tune-Profile mit Zielschritten pro Modalitaet und haengt sich standardmaessig zwischen `11` und `14` in die Trainingskette
- `13_run_backend_finetunes.py` erzeugt jetzt daraus konkrete Backend-Fine-Tune-Laeufe pro Modalitaet, materialisiert lokale Backend-Artefakte pro Figur und haengt sich standardmaessig zwischen `12` und `14` in die Trainingskette
- der neue Fine-Tune-Block wird jetzt auf echtem Bestand beobachtet, besonders dort, wo bisher nur Bild/Video-Modalitaeten, aber noch keine starken Voice-Samples vorhanden sind
- `17_generate_preview_episodes.py` schreibt jetzt ebenfalls einen globalen Wrapper-Autosave fuer komplette Multi-Episoden-Laeufe
- benannte Figuren koennen jetzt direkt als Hauptfigur priorisiert werden, damit `08` sie bevorzugt als Fokusrollen verwendet
- `16_render_episode.py` schreibt jetzt pro benannter Figur auch ein lokales `voice_model`-JSON und legt das Render-Manifest sowohl im Draft- als auch im Final-Ordner ab
- Voice-Samples werden jetzt ueber `10 -> 11 -> 12 -> 13 -> 16` mit Dauer-, Qualitaets- und Clone-Bereitschaftsmetriken durchgetragen, damit der Stimmenpfad pro Figur sichtbar belastbar bleibt
- die neue Voice-Clone-Bereitschaft blockiert jetzt bewusst erst den Render-Pfad; Generierung darf weiter mit trainierten Packs laufen, auch wenn eine Figur fuer echten Clone noch nicht stark genug ist
- `16_render_episode.py` wurde fuer lange Windows-Renderlaeufe auf eine FFmpeg-Concat-Liste umgestellt, damit der Final-Render nicht mehr an `WinError 206` durch zu lange Kommandozeilen scheitert
- der lokale TTS-Fallback in `16_render_episode.py` bevorzugt jetzt auf Windows zwingend vorhandene deutsche Systemstimmen wie `Hedda` statt alte englische Altprofile weiterzuverwenden
- `14_generate_episode_from_trained_model.py` erzeugt neue Preview-Folgen jetzt wieder bewusst als synthetische Modellvorschau statt als Zusammenstellung von Originalsegmenten
- die Ziellaenge neuer Folgen wird jetzt aus den verarbeiteten Quellfolgen abgeleitet, statt starr auf einen Serienbereich festgelegt zu sein
- `09_prepare_foundation_training.py` zieht fehlende Basismodelle jetzt standardmaessig automatisch nach und prueft vorhandene Downloads auf neuere Remote-Revisionen, damit vorhandene Modelle bei Bedarf aktualisiert werden
- `09_prepare_foundation_training.py` erkennt bei alten bereits vorhandenen Hugging-Face-Downloads jetzt auch den lokalen Cache-Stand und behandelt identische Modelle als `skip` statt denselben Stand erneut komplett herunterzuladen
- `09_prepare_foundation_training.py` behandelt unvollstaendige Hugging-Face-Cache-Dateien bei identischer Revision jetzt zuerst als lokalen Cleanup-Fall statt als erneuten Voll-Download; nur bei echter anderer Revision wird ein Reparatur-/Update-Lauf gestartet
- der alte Karten-/Crop-Look wird im Fallback gerade durch Vollbild-Referenzframes ersetzt; der peinliche Portrait-Lip-Sync ist im Default jetzt bewusst deaktiviert
- der fruehere Cartoon-/Mii-Mund im Portrait-Fallback ist entfernt; der lokale Fallback arbeitet jetzt mit mehreren echten Referenzframes und sanfterem Gesichtswarp
- `14_generate_episode_from_trained_model.py` mischt jetzt deutlich staerker echte Sprecher-Samples in die neuen Dialoge, damit `16` mehr Originalstimmen wiederverwenden kann
- `14_generate_episode_from_trained_model.py` schreibt jetzt echte Episodentitel in Markdown, Shotlist und Render-Metadaten statt nur `folge_0x` und filtert schwache Keyword-Titel wie `meinem`, `soll` oder `wieder` deutlich haerter aus
- `09_prepare_foundation_training.py` bereitet jetzt den naechsten Block fuer echtes Bild-/Video-/Voice-Fine-Tuning vor: 720p-Frames, kurze Trainingsclips, Voice-Referenzen, Download-Ziele und Trainingsplaene
- `17_generate_preview_episodes.py` und `99_process_next_episode.py` fuehren jetzt standardmaessig erst `07`, dann `08`, dann `09`, dann `10`, dann `11`, dann `12`, dann `13` aus, bevor `14`, `15` oder `16` ueberhaupt starten
- `14_generate_episode_from_trained_model.py` blockt jetzt hart, wenn das Training fehlt oder aelter als das aktuelle Serienmodell ist
- `16_render_episode.py` blockt jetzt zusaetzlich, wenn fuer die Fokusfiguren keine trainierten Trainingspacks vorhanden sind, statt still weiter nur auf altes System-TTS zu kippen
- der aktuelle Render-Pfad wird gerade gegen lange Segmentstapel auf Windows weiter gehaertet, damit auch grosse Finalfolgen stabil fertig zusammengebaut werden
- der Stimmenpfad wird gerade weiter von einfachem System-TTS weggezogen; kurzfristig ist der englische Windows-Fallback entfernt, mittelfristig braucht der Projektstand einen echten lokalen Clone-Runner statt nur Profil-/Fallback-Logik

## Geplant

- Hauptfiguren in `06_review_unknowns.py` anhand der neuen Priorisierung sauber benennen
- Nebenfiguren in `06_review_unknowns.py` gezielt als `statist` markieren, damit sie spaeter auftauchen duerfen, aber nicht zu Hauptfiguren werden
- danach `18_refresh_after_manual_review.py` ueber den final benannten Bestand laufen lassen, damit Datensaetze, Modell, Foundation-Packs, Folge, Serienbibel und Render mit echten Namen und Prioritaeten neu aufgebaut werden
- verbleibende `speaker_unknown`-Faelle in `04` nach dem neuen Zwei-Stufen-Rescue-Pass auf kompletten Staffeln weiter feinjustieren
- die neue digiKam-inspirierte Mehrfachreferenz-Erkennung in `06_review_unknowns.py` nach der neuen Identitaetsgewichtung auf kompletten Staffeln weiter feinjustieren
- den neuen Fine-Tune-Block hinter `12_train_fine_tune_models.py` spaeter von lokalen Profilen zu echten Fine-Tune-Laeufen auf Modellgewichts-Ebene ausbauen
- die neuen lokalen Fine-Tune-Profile nach den jetzt durchgezogenen Voice-Qualitaetsmetriken weiter mit besseren Voice-Samples und spaeter echten Stimmenklon-Backends verknuepfen
- den Render-Pfad nach den neuen Voice-Clone-Metriken weiter so ausbauen, dass clone-bereite Figuren spaeter sauber auf echte lokale Stimmen-Backends geschaltet werden koennen
- die neuen Backend-Fine-Tune-Laeufe aus `13_run_backend_finetunes.py` nach den jetzt materialisierten lokalen Backend-Artefakten spaeter an echte Trainingsbackends statt nur an lokale Laufartefakte anbinden
- die neue Modell-Update-Pruefung in `09_prepare_foundation_training.py` auf laengeren Serienlaeufen beobachten, besonders bei grossen Remote-Modellen
- spaeter erst wieder einen Lip-Sync-Pfad aktivieren, wenn er qualitativ wirklich auf Serienniveau wirkt
- die neuen lokalen Fine-Tune-Profile spaeter zu echten Modellgewichts-Fine-Tunes fuer Bild/Video/Stimme ausbauen, sobald die vorbereiteten Datensaetze stabil genug sind
- spaeter gezielte Qualitaetsverbesserungen fuer Render, Figurenkonsistenz und Episodengenerierung angehen

## Aktueller Gesamtstand

Das Projekt ist im Moment am staerksten in diesem Bereich:

- aus realen Serienfolgen automatisch verwertbare Trainingsdaten gewinnen
- Figuren, Stimmen und Szenen zu einem lokalen Serienwissen verdichten
- daraus neue Episoden als Text, Shotlist und Preview-Render ableiten

Der Schwerpunkt liegt aktuell noch nicht auf perfekt fotorealistischer Endausgabe, sondern auf:

- sauberer Datenaufbereitung
- wiederkehrenden Figuren statt Einmal-Treffern
- nachvollziehbarer lokaler Pipeline
- brauchbaren Preview- und Review-Schritten
- reproduzierbarer Generierung statt vollautomatischer TV-Endproduktion

## Aktueller Stand

Das Projekt kann aktuell:

- `mp4`-Folgen einlesen
- Szenen exportieren
- Audio je Szene mit Whisper transkribieren
- einfache Sprecher-Cluster ueber Audio-Embeddings bilden
- Gesichter pro Szene erkennen und in Face-Cluster gruppieren
- Sprecher mit wahrscheinlichen Figuren verknuepfen
- daraus einen Trainingsdatensatz bauen
- ein lokales Serienmodell aus den Datensaetzen ableiten
- neue Folgen als Text/Shotlist erzeugen
- neue Folgen mit einem echten Episodentitel statt nur `folge_0x` versehen
- daraus ein Draft-Video mit Karten und TTS rendern
- daraus auch ein `final`-MP4 im Final-Ordner schreiben
- pro benannter Figur Referenzbilder und Referenzaudio fuer spaetere Clone-Schritte sammeln
- pro benannter Figur ein lokales Stimmprofil aus Referenzaudio ableiten
- pro benannter Figur ein lokales `voice_model`-JSON mit Referenzpfad, Rate und Stimmmerkmalen schreiben
- pro benannter Figur ein lokales Retrieval-Modell aus Originalsegmenten aufbauen, damit passende neue Zeilen wenn moeglich mit echter Originalstimme wiedergegeben werden
- neue Dialoge als synthetische Preview aus dem trainierten Modell erzeugen
- die Ziel-Laufzeit neuer Folgen aus den verarbeiteten Quellfolgen ableiten
- lokale Adapter-Profile fuer Bild, Stimme und Clip-Dynamik aus den Foundation-Manifeste trainieren
- lokale Fine-Tune-Profile mit Zielschritten pro Modalitaet aus den Adapter-Profilen ableiten
- konkrete Backend-Fine-Tune-Laeufe pro Modalitaet aus den Fine-Tune-Profilen erzeugen und als lokale Backend-Artefakte materialisieren
- im Standardpfad ohne externe Lizenzannahmen mit lokalem `pyttsx3` rendern
- optional auf ausdruecklichen Wunsch einen XTTS-/Coqui-Pfad vorbereiten
- fuer Fallback-Segmente jetzt statt UI-Karten bevorzugt Vollbild-Referenzframes mit minimaler Untertitel-Einblendung rendern
- 720p-Trainingsmaterial fuer einen spaeteren Bild-/Video-/Voice-Fine-Tune-Block vorbereiten

Das Projekt kann aktuell noch nicht:

- vollstaendig neue animierte TV-Folgen generieren
- produktionsreifes Deepfake-Lipsync auf Spielfilm-Niveau
- perfekte stiltreue Voice-Clones fuer jede Figur ohne gutes Referenzmaterial
- stiltreue Video-Generierung auf Produktionsniveau

`16_render_episode.py` rendert neue Preview-Folgen im Standard jetzt wieder aus synthetisch erzeugten Episoden-Assets und greift nicht mehr automatisch auf zusammengewuerfelte Originalsegmente zurueck. Der bisher peinliche Portrait-Lip-Sync ist im Standardpfad deaktiviert. Ein optionaler XTTS-/Coqui-Weg bleibt bewusst ausgeschaltet, solange er nicht ausdruecklich angefordert und freigeschaltet wurde.

## Projektstruktur

Wichtige Root-Dateien:

- `00_prepare_runtime.py`: Runtime und Pakete vorbereiten
- `01_setup_project.py`: Projektstruktur anlegen
- `02_import_episode.py`: naechste Folge aus der Inbox importieren
- `03_split_scenes.py`: importierte Folge in Szenen zerlegen
- `04_diarize_and_transcribe.py`: Audio extrahieren, transkribieren, Sprecher clustern
- `05_link_faces_and_speakers.py`: Gesichter erkennen und mit Stimmen verknuepfen
- `06_review_unknowns.py`: offene Zuordnungen anzeigen und manuell benennen
- `07_build_dataset.py`: Trainingsdatensatz bauen
- `08_train_series_model.py`: Serienmodell trainieren
- `09_prepare_foundation_training.py`: 720p-Frames, Clips, Voice-Referenzen und Basis-Downloads fuer spaeteres Fine-Tuning vorbereiten
- `10_train_foundation_models.py`: lokale Foundation-Packs fuer Bild, Video und Stimme trainieren
- `11_train_adapter_models.py`: lokale Adapter-Profile fuer Bild, Stimme und Clip-Dynamik trainieren
- `12_train_fine_tune_models.py`: lokale Fine-Tune-Profile aus den Adapter-Profilen trainieren
- `13_run_backend_finetunes.py`: konkrete Backend-Fine-Tune-Laeufe aus den lokalen Fine-Tune-Profilen ableiten
- `14_generate_episode_from_trained_model.py`: neue Folge aus trainiertem Serienmodell erzeugen
- `15_build_series_bible.py`: Serienbibel aktualisieren
- `16_render_episode.py`: Storyboard-/TTS-Draft rendern, Final-Export und lokale Stimmprofile schreiben
- `17_generate_preview_episodes.py`: mehrere neue sichtbare Preview-Episoden am Stueck erzeugen
- `18_refresh_after_manual_review.py`: Datensaetze, Modell, Trainingspacks und Folge nach manueller Figurenpflege bewusst neu aufbauen; blockt standardmaessig bei offenen Review-Faellen
- `19_sync_to_github.py`: Root-Skripte und README optional nach GitHub spiegeln
- `99_process_next_episode.py`: komplette Pipeline ausfuehren
- `pipeline_common.py`: gemeinsame Helfer fuer Pfade, Config, Runtime und Registry

Wichtige Projektordner:

- `ai_series_project/data/inbox/episodes`: neue Eingabe-`mp4`
- `ai_series_project/data/raw/episodes`: importierte Arbeitskopien
- `ai_series_project/data/raw/audio`: exportierte WAV-Dateien
- `ai_series_project/data/processed/scene_clips`: Szenenclips je Folge
- `ai_series_project/data/processed/scene_index`: Szenenlisten als CSV
- `ai_series_project/data/processed/speaker_segments`: segmentierte Audio-/Transkript-Caches
- `ai_series_project/data/processed/speaker_transcripts`: kombinierte Sprechersegmente pro Folge
- `ai_series_project/data/processed/faces`: Face-Caches pro Folge
- `ai_series_project/data/processed/linked_segments`: verknuepfte Sprecher-/Figuren-Segmente
- `ai_series_project/data/datasets/video_training`: Trainingsdatensaetze
- `ai_series_project/characters/maps`: `character_map.json` und `voice_map.json`
- `ai_series_project/characters/previews`: Gesichts- und Sprecher-Previews
- `ai_series_project/characters/review`: Review-Queue fuer unklare Zuordnungen
- `ai_series_project/generation/model`: lokales Serienmodell
- `ai_series_project/generation/story_prompts`: generierte Folgen als Markdown
- `ai_series_project/generation/shotlists`: Shotlists fuer neue Folgen
- `ai_series_project/generation/renders/drafts`: gerenderte Draft-Videos
- `ai_series_project/series_bible/episode_summaries`: Serienbibel
- `ai_series_project/training/foundation/adapters`: lokale Adapter-Profile und Adapter-Zusammenfassungen
- `ai_series_project/training/foundation/finetunes`: lokale Fine-Tune-Profile und Fine-Tune-Zusammenfassungen
- `ai_series_project/training/foundation/backend_runs`: konkrete Backend-Fine-Tune-Laeufe und Backend-Zusammenfassungen
- `runtime/venv`: lokale Python-Umgebung
- `tools`: Tool-Fallback auf Root-Ebene

## Voraussetzungen

- Windows / PowerShell
- funktionierendes lokales Python
- genug Speicherplatz fuer Szenen, Audio und Render-Artefakte
- genug Zeit: `04` und `05` koennen auf CPU lange laufen
- optional NVIDIA-GPU fuer schnellere Transkription, Face-Embeddings und FFmpeg-NVENC-Render

Die Runtime wird durch `00_prepare_runtime.py` eingerichtet. Dabei werden je nach Verfuegbarkeit unter anderem diese Pakete vorbereitet:

- `torch`
- `torchvision`
- `torchaudio`
- `numpy`
- `Pillow`
- `opencv-python`
- `librosa`
- `openai-whisper`
- `scenedetect[opencv]`
- `facenet-pytorch`
- `speechbrain` (optional, aber aktuell bevorzugt fuer robustere Sprecher-Embeddings)
- `pyttsx3`
- `TTS` nur optional und nicht Teil des lizenzfreien Standardpfads

## Schnellstart

1. Neue Folgen als `mp4` nach `ai_series_project/data/inbox/episodes` legen.
2. `python 00_prepare_runtime.py`
3. `python 01_setup_project.py`
4. `python 99_process_next_episode.py`
5. Ergebnisse in `generation/story_prompts`, `generation/shotlists`, `generation/renders/drafts` und `series_bible/episode_summaries` pruefen.
6. Optional den lokalen Sync-Helfer starten, um nur Root-Skripte und `README.md` nach GitHub zu pushen.
7. Optional `python 09_prepare_foundation_training.py`, wenn du aus dem verarbeiteten Material zusaetzlich 720p-Trainingsdaten und Modell-Download-Ziele fuer spaeteres Fine-Tuning vorbereiten willst.

## Standard-Workflow im Detail

### 00 - Runtime vorbereiten

`python 00_prepare_runtime.py`

Dieser Schritt:

- erstellt bei Bedarf `runtime/venv`
- ersetzt eine alte `venv` mit `include-system-site-packages = true` automatisch durch eine saubere isolierte Umgebung
- aktualisiert `pip`, `setuptools` und `wheel`
- installiert benoetigte Pakete
- versucht bei NVIDIA-Verfuegbarkeit automatisch eine CUDA-faehige `torch`-Installation
- legt die Projektstruktur und die Config an

Wenn GPU-Unterstuetzung verfuegbar ist und in der Config nicht deaktiviert wurde, wird sie ab diesem Schritt fuer die spaeteren Torch-/FFmpeg-Schritte vorbereitet.
Dabei arbeitet die Pipeline im Alltag als Hybrid-Modus: CPU bleibt fuer I/O, OpenCV, Audio-Features und Fallbacks aktiv, GPU uebernimmt Torch-/Whisper-/FaceNet-Inferenz. Ohne GPU laeuft alles nur auf CPU.
Zusatz fuer die neue Clone-Stufe: Wenn `TTS` erfolgreich installiert ist, nutzt `10` fuer XTTS automatisch bevorzugt die GPU. Ohne GPU oder ohne `TTS` faellt der Render sauber auf den bisherigen lokalen TTS-Weg zurueck.
Im lizenzfreien Standardpfad installiert `00` das optionale XTTS-/Coqui-Paket nicht automatisch. Wenn du diesen optionalen Weg spaeter trotzdem bewusst vorbereiten willst, setze vor `00` zusaetzlich `SERIES_ENABLE_OPTIONAL_TTS=1`.
Wichtig fuer XTTS: Das Coqui-Modell darf hier nicht stillschweigend mit einer Lizenzbestaetigung gestartet werden. XTTS wird deshalb erst benutzt, wenn du die Lizenz selbst bestaetigt und danach `cloning.voice_clone_engine='xtts'` sowie `cloning.xtts_license_accepted=true` in der Config gesetzt oder beim Rendern `SERIES_ACCEPT_COQUI_LICENSE=1` gesetzt hast.

Ohne `00` sollten die anderen Schritte nicht gestartet werden.

### 01 - Projektstruktur erzeugen

`python 01_setup_project.py`

Dieser Schritt stellt sicher, dass die gesamte Projektstruktur und die Config-Datei vorhanden sind.

### 02 - Folge importieren

`python 02_import_episode.py`

Dieser Schritt:

- nimmt die erste noch nicht verarbeitete Datei aus `data/inbox/episodes`
- kopiert sie nach `data/raw/episodes`
- loescht die Inbox-Datei direkt danach, sobald die Kopie in `data/raw/episodes` vollstaendig vorliegt
- entfernt ausserdem Inbox-Dubletten direkt, wenn dieselbe Folge bereits als Arbeitskopie, Metadaten- oder Szenenbestand vorhanden ist
- schreibt Metadaten nach `data/processed/metadata`
- markiert die Datei in `logs/processing_registry.json` als importiert
- schreibt zusaetzlich einen Schritt-Autosave fuer Start, Erfolg oder Fehler nach `runtime/autosaves/steps/02_import_episode`

Wichtig:

- `02` verarbeitet immer nur die naechste unregistrierte Datei
- mit `python 02_import_episode.py --all` importiert `02` stattdessen alle aktuell im Inbox-Ordner liegenden Folgen am Stueck
- danach wird ausschliesslich mit der Arbeitskopie in `data/raw/episodes` weitergearbeitet
- wenn die Kopie unvollstaendig waere, bleibt die Inbox-Datei bewusst erhalten und `02` bricht ab
- wenn dieselbe Folge schon bearbeitet wurde und als Dublette nochmal im Inbox-Ordner liegt, raeumt `02` diese Dublette ebenfalls direkt weg
- eine `mp4` direkt im Root-Ordner wird nicht automatisch verarbeitet

### 03 - Szenen schneiden

`python 03_split_scenes.py`

Dieser Schritt:

- nimmt ohne `--episode-file` alle aktuell importierten Arbeitskopien aus `data/raw/episodes` nacheinander
- erkennt Szenen mit `scenedetect`
- exportiert je Szene eine `scene_XXXX.mp4`
- schreibt die Szenenliste als CSV nach `data/processed/scene_index`
- schreibt zusaetzlich einen Erfolgsmarker fuer den abgeschlossenen Szenenschnitt nach `data/processed/scene_index`
- schreibt parallel einen Schritt-Autosave fuer Start, Erfolg oder Fehler nach `runtime/autosaves/steps/03_split_scenes`

Wenn keine Szene erkannt wird, faellt der Schritt auf feste Segmente anhand von `default_scene_seconds_fallback` zurueck.

Wichtig:

- wenn bereits ein Szenenordner existiert, beendet sich `03` ohne Neuberechnung
- wenn `delete_input_after_split` auf `true` steht, wird nur die importierte Arbeitskopie unter `data/raw/episodes` nach erfolgreichem Split geloescht
- die eigentliche Inbox-Datei ist zu diesem Zeitpunkt bereits in `02_import_episode.py` nach erfolgreicher Vollkopie nach `data/raw/episodes` entfernt worden
- ohne `--episode-file` arbeitet `03` jetzt den kompletten aktuellen Bestand in `data/raw/episodes` von oben nach unten ab
- bereits vorhandene Dateien im Raw-Ordner werden dabei ganz normal mitgenommen; schon erfolgreich gesplittete Folgen werden erkannt, sauber uebersprungen und ihre haengengebliebene Arbeitskopie bei Bedarf entfernt
- optional kann `03` jetzt gezielt eine importierte Arbeitsdatei verarbeiten: `python 03_split_scenes.py --episode-file "<dateiname>.mp4"`
- wenn fuer dieselbe Folge bereits ein vollstaendiger erfolgreicher Szenenschnitt vorliegt, erkennt `03` das ueber Erfolgsmarker oder vorhandene Szenenliste plus Clips, raeumt bei Bedarf nur noch die haengengebliebene Arbeitskopie weg und beendet sich sauber ohne erneuten Export
- das gilt auch dann, wenn die Raw-Arbeitskopie schon geloescht ist und `03 --episode-file ...` spaeter nur noch gegen den vorhandenen Szenenbestand prueft

### 04 - Audio, Transkription, Sprecher-Cluster

`python 04_diarize_and_transcribe.py`

Dieser Schritt:

- verarbeitet ohne `--episode` automatisch alle noch offenen Szenenordner unter `data/processed/scene_clips` nacheinander
- mit `--episode "<folgenordner>"` kann weiter gezielt genau eine Folge verarbeitet werden
- exportiert Audio pro Szene nach `data/raw/audio`
- transkribiert jede Szene mit Whisper
- schneidet daraus Segmente
- berechnet Voice-Embeddings, bevorzugt ueber `speechbrain` mit MFCC-Fallback
- bildet Sprecher-Cluster (`speaker_001`, `speaker_002`, ...)
- schreibt waehrend des Laufs einen Schritt-Autosave mit bereits fertig bearbeiteten Szenen

Wichtige Ausgaben:

- `data/processed/speaker_segments/<folge>/...`
- `data/processed/speaker_transcripts/<folge>_segments.json`
- `data/processed/speaker_segments/<folge>/_speaker_clusters.json`

Wichtig:

- Whisper verwendet standardmaessig `large-v3`
- bei nutzbarer CUDA-GPU wird Whisper direkt auf GPU geladen
- fuer Sprecher-Embeddings wird bevorzugt ebenfalls GPU genutzt, wenn `speechbrain` verfuegbar ist
- kleine Einzelsegmente werden aggressiver als `speaker_unknown` behandelt, damit `05` spaeter nicht auf Einmal-Clustern aufbaut
- verbleibende `speaker_unknown`-Zeilen bekommen danach jetzt noch einen konservativen Rescue-Pass ueber Nachbarschaft und Embedding-Aehnlichkeit zu stabilen Sprechern
- dieser Schritt kann sehr lange dauern
- vorhandene Cache-Dateien werden wiederverwendet, solange die interne `process_version` passt
- optional kann `04` gezielt einen Szenenordner verarbeiten: `python 04_diarize_and_transcribe.py --episode "<folgenordner>"`
- wenn fuer eine Folge bereits die aktuellen Segment- und Cluster-Dateien aus `04` vorhanden sind, wird sie sauber uebersprungen
- `04` speichert seinen Fortschritt ueber Szenen-Caches und zusaetzliche Schritt-Autosaves; dadurch kann ein Abbruch spaeter pro Szene weiterlaufen und wird erst bei aktueller Prozessversion als abgeschlossen gewertet

### 05 - Gesichter und Stimmen verknuepfen

`python 05_link_faces_and_speakers.py`

Dieser Schritt:

- liest die Sprechersegmente aus `04`
- verarbeitet ohne `--episode` automatisch alle noch offenen Folgen nacheinander
- mit `--episode "<folgenordner>"` kann weiter gezielt genau eine Folge verarbeitet werden
- erkennt Gesichter in den Szenen ueber `facenet-pytorch` mit OpenCV-Fallback
- bildet Face-Cluster (`face_001`, `face_002`, ...)
- konsolidiert Mehrfacherkennungen zuerst lokal innerhalb einer Szene
- filtert nach dem Voll-Lauf automatisch Einmal-Gesichter mit zu wenig Szenen/Treffern aus der `character_map`
- ordnet sichtbare Face-Cluster zeitbasiert pro Dialogsegment zu
- verknuepft diese segmentnahen Face-Cluster mit Sprecher-Clustern
- uebernimmt ausschliesslich manuell vergebene Figurennamen aus `character_map.json`
- ignoriert Cluster mit dem Namen `noface` kuenftig automatisch
- behandelt `statist` als bewusst gesetzte Nebenfigur: sichtbar, aber nicht als Hauptrolle
- baut `character_map.json`, `voice_map.json`, `linked_segments.json` und `review_queue.json`

Wichtige technische Details:

- wenn eine CUDA-GPU nutzbar ist, laufen MTCNN und Face-Embeddings auf GPU
- wenn MTCNN in einzelnen Frames keine Box liefert, faellt `05` auf OpenCV-Haar-Cascades zurueck
- alte Platzhalterdaten werden beim Laden automatisch normalisiert, damit fruehere `figur_###`-/`stimme_###`-Artefakte nicht weitergeschleppt werden
- unbenannte Figuren bleiben technisch `face_###`; unbenannte Sprecher bleiben technisch `speaker_###`
- `statist` wird intern vereinheitlicht und ohne Alias gespeichert, damit mehrere Extras nicht als eine eindeutige Hauptfigur verwechselt werden
- `sample_every_n_frames`, `max_faces_per_frame`, `max_scene_clusters`, `max_visible_faces_per_segment`, `segment_visibility_padding_seconds`, `min_face_size`, `face_cluster_min_scenes`, `face_cluster_min_detections`, `embedding_threshold`, `scene_embedding_threshold` und `detection_confidence_threshold` kommen aus der Projekt-Config

Wichtige Ausgaben:

- `data/processed/faces/<folge>/...`
- `characters/maps/character_map.json`
- `characters/maps/voice_map.json`
- `data/processed/linked_segments/<folge>_linked_segments.json`
- `characters/review/review_queue.json`

Neue CLI-Optionen:

- `python 05_link_faces_and_speakers.py --help`
- `python 05_link_faces_and_speakers.py --fresh`
- `python 05_link_faces_and_speakers.py --episode "Game.Shakers.S01E01.GERMAN.720p.WEB.H264-BiMBAMBiNO"`
- `python 05_link_faces_and_speakers.py --fresh --episode "Game.Shakers.S01E01.GERMAN.720p.WEB.H264-BiMBAMBiNO"`

Bedeutung:

- normaler Lauf: nutzt vorhandene Maps und Face-Caches weiter
- ohne `--episode` verarbeitet `05` jetzt den kompletten offenen Stapel statt nur den naechsten Szenenordner
- `--fresh`: loescht die bisherigen `05`-Artefakte vor dem Lauf
- `--episode`: waehlt gezielt einen Szenenordner
- wenn fuer eine Folge bereits `linked_segments` und `face_summary` vorhanden sind, beendet sich `05` ohne `--fresh` sauber mit `bereits vorhanden`
- `05` setzt bei Teilabbruch weiter auf seine bestehenden Face-Szenencaches und Schritt-Autosaves; zusaetzlich markiert `_face_linking_success.json` nur vollstaendig erfolgreiche Laeufe
- nur fuer einen bewusst kompletten Reset bleibt `--fresh` der richtige Weg
- `--fresh` ist bewusst nur zusammen mit `--episode` erlaubt, damit globale Maps bei einem Batchlauf nicht versehentlich fuer alle Folgen geleert werden

Sehr wichtig:

- `character_map.json` und `voice_map.json` sind aktuell globale Dateien
- `--fresh` setzt daher die aktuellen Maps komplett zurueck und nicht nur einen einzelnen Character
- wenn du ein Cluster spaeter auf `noface` setzt, wird es bei kuenftigen `05`-Laeufen nicht mehr als sichtbare Figur verwendet

### 07 - Trainingsdatensatz bauen

`python 07_build_dataset.py`

Dieser Schritt:

- liest `linked_segments`
- verarbeitet ohne `--episode` automatisch alle noch offenen Folgen nacheinander
- mit `--episode "<folgenordner>"` kann weiter gezielt genau eine Folge verarbeitet werden
- mit `--force` werden vorhandene Datensaetze bewusst neu aufgebaut
- baut pro Szene einen kompakten Datensatz mit Figuren, Transkript, Keywords und Segmentdetails
- schreibt den Datensatz nach `data/datasets/video_training`

Wichtig:

- optional kann `07` jetzt gezielt einen Szenenordner verarbeiten: `python 07_build_dataset.py --episode "<folgenordner>"`
- ohne `--episode` verarbeitet `07` jetzt den kompletten offenen Stapel statt nur den naechsten Szenenordner
- `python 07_build_dataset.py --force` baut den kompletten Datensatzstapel bewusst neu auf
- wenn ein Datensatz fuer die Folge bereits vorhanden ist, wird sie sauber uebersprungen
- `07` schreibt waehrend des Aufbaus Schritt-Autosaves und sichert einen erfolgreichen Lauf zusaetzlich ueber `dataset_manifest.json` mit aktueller `process_version`

### Schritt-Autosaves

Lange oder mehrstufige Schritte schreiben jetzt zusaetzlich eigene Fortschrittsdateien nach:

- `ai_series_project/runtime/autosaves/steps`

Aktuell relevant:

- `00_prepare_runtime.py`: speichert Start/Erfolg/Fehler des Runtime-Setups
- `01_setup_project.py`: speichert Start/Erfolg/Fehler der Projektinitialisierung
- `02_import_episode.py`: speichert Start/Erfolg/Fehler pro importierter Folge
- `03_split_scenes.py`: speichert Start/Erfolg/Fehler pro Folge zusaetzlich zum Split-Erfolgsmarker
- `04_diarize_and_transcribe.py`: speichert laufend bereits fertig transkribierte Szenen pro Folge
- `05_link_faces_and_speakers.py`: speichert laufend bereits analysierte Szenen und den finalen Linking-Status pro Folge
- `07_build_dataset.py`: speichert laufend bereits uebernommene Szenen und den finalen Datensatzstatus pro Folge
- `08_train_series_model.py`: schreibt Start/Erfolg des Modelltrainings
- `09_prepare_foundation_training.py`: schreibt Start/Erfolg der Vorbereitungsphase
- `10_train_foundation_models.py`: speichert Fortschritt pro Figur und ueberspringt bereits fertige Foundation-Packs
- `14_generate_episode_from_trained_model.py`: speichert Start/Erfolg pro erzeugter Folge
- `15_build_series_bible.py`: speichert Start/Erfolg der Serienbibel
- `16_render_episode.py`: speichert Fortschritt pro Render-Folge und setzt bei vorhandenen Segmenten am bestehenden Stand wieder an
- `11_train_adapter_models.py`: speichert Fortschritt pro Figur und schreibt zusaetzlich einen globalen Adapter-Zusammenfassungsstand
- `12_train_fine_tune_models.py`: speichert Fortschritt pro Figur und schreibt zusaetzlich einen globalen Fine-Tune-Zusammenfassungsstand
- `13_run_backend_finetunes.py`: speichert Fortschritt pro Figur und schreibt zusaetzlich einen globalen Backend-Zusammenfassungsstand
- `17_generate_preview_episodes.py`: speichert Start/Erfolg/Fehler fuer komplette Multi-Episoden-Laeufe

### 08 - Serienmodell trainieren

`python 08_train_series_model.py`

Dieser Schritt:

- liest alle `*_dataset.json` aus `data/datasets/video_training`
- baut daraus ein lokales Serienmodell
- schreibt nur das trainierte Serienmodell
- filtert haeufige Fuellwoerter und typische Untertitel-/Artefaktbegriffe aus der Themenauswahl heraus

Wichtige Ausgaben:

- `generation/model/series_model.json`

Wichtig:

- das "Training" ist aktuell ein lokales heuristisches Modell aus Statistiken, Sprecherbeispielen, Keywords und einer Markov-Chain
- dies ist kein grosses neuronales Generationsmodell
- die eigentliche Episodengenerierung passiert jetzt erst in `14_generate_episode_from_trained_model.py`

### 09 - Foundation-Training vorbereiten

`python 09_prepare_foundation_training.py`

Dieser Schritt:

- sammelt benannte Hauptfiguren mit genug Szenen- und Zeilenmaterial
- exportiert pro Figur 720p-Frames aus echten Szenenclips
- exportiert pro Figur kurze 720p-Trainingsclips aus echten Szenenclips
- kopiert vorhandene Voice-Referenzen in ein einheitliches Trainingslayout
- schreibt pro Figur ein Manifest und einen gemeinsamen Trainingsplan
- laedt fehlende konfigurierte Basis-Modelle standardmaessig automatisch nach
- prueft vorhandene Modell-Downloads jetzt standardmaessig auf neuere Remote-Revisionen und aktualisiert sie bei Bedarf
- kann automatische Downloads mit `--skip-downloads` fuer einen Lauf bewusst ueberspringen

Wichtige Ausgaben:

- `training/foundation/datasets/frames/<figur>/`
- `training/foundation/datasets/video/<figur>/`
- `training/foundation/datasets/voice/<figur>/`
- `training/foundation/manifests/<figur>_manifest.json`
- `training/foundation/plans/foundation_training_plan.json`
- `training/foundation/plans/foundation_training_plan.md`

Beispiele:

- `python 09_prepare_foundation_training.py`
- `python 09_prepare_foundation_training.py --limit-characters 4`
- `python 09_prepare_foundation_training.py --episode "Game.Shakers.S01E01.GERMAN.720p.WEB.H264-BiMBAMBiNO"`
- `python 09_prepare_foundation_training.py --download-models`
- `python 09_prepare_foundation_training.py --skip-downloads`

Wichtig:

- `--download-models` triggert jetzt nicht nur Erst-Downloads, sondern auch die aktive Update-Pruefung gegen das Remote-Modell
- ohne `--skip-downloads` prueft `09` vorhandene Modellordner standardmaessig ebenfalls auf Updates
- die automatische Update-Pruefung wird ueber `foundation_training.check_model_updates=true|false` gesteuert
- bereits vorhandene alte Downloads ohne `.foundation_download.json` werden jetzt beim ersten erfolgreichen Skip selbstheilend mit lokaler Revisions-Metadatei versehen
- vorhandene `.incomplete`-Dateien im Hugging-Face-Cache fuehren bei gleicher Revision jetzt nur zu einem lokalen Cleanup statt zu einem doppelten Redownload

### 10 - Foundation-Modelle trainieren

`python 10_train_foundation_models.py`

Dieser Schritt:

- liest die Manifeste aus `09`
- trainiert lokale Foundation-Packs fuer Bild, Video und Stimme pro Figur
- trainiert mit `--force` vorhandene Foundation-Packs bewusst neu
- schreibt pro Figur einen `foundation_pack.json`-Checkpoint
- schreibt eine Trainingszusammenfassung fuer den aktuellen Lauf

Wichtige Ausgaben:

- `training/foundation/checkpoints/<figur>/foundation_pack.json`
- `training/foundation/checkpoints/foundation_training_summary.json`

Wichtig:

- ohne `--force` werden bereits fertige Foundation-Packs sauber uebersprungen
- `python 10_train_foundation_models.py --force` trainiert vorhandene Packs bewusst neu, zum Beispiel nach manuellen Figurenkorrekturen
- der naechste optionale Trainingsblock ist jetzt `python 11_train_adapter_models.py`

### 11 - Lokale Adapter-Profile trainieren

`python 11_train_adapter_models.py`

Dieser Schritt:

- liest die Foundation-Manifeste aus `09`
- liest die Foundation-Packs aus `10`
- trainiert daraus lokale Adapter-Profile fuer Bild, Stimme und Clip-Dynamik
- schreibt pro Figur ein `adapter_profile.json`
- schreibt eine globale Adapter-Zusammenfassung fuer den aktuellen Lauf

Wichtige Ausgaben:

- `training/foundation/adapters/<figur>/adapter_profile.json`
- `training/foundation/adapters/adapter_training_summary.json`

Wichtig:

- ohne `--force` werden bereits fertige Adapter-Profile sauber uebersprungen
- `python 11_train_adapter_models.py --force` trainiert vorhandene Adapter-Profile bewusst neu
- im aktuellen Stand sind das lokale Profil-/Adapter-Checkpoints, noch keine echten Fine-Tunes auf Modellgewichts-Ebene

### 12 - Lokale Fine-Tune-Profile trainieren

`python 12_train_fine_tune_models.py`

Dieser Schritt:

- liest die Adapter-Zusammenfassung aus `11`
- trainiert daraus lokale Fine-Tune-Profile mit Zielschritten pro Modalitaet
- schreibt pro Figur ein `fine_tune_profile.json`
- schreibt eine globale Fine-Tune-Zusammenfassung fuer den aktuellen Lauf

Wichtige Ausgaben:

- `training/foundation/finetunes/<figur>/fine_tune_profile.json`
- `training/foundation/finetunes/fine_tune_training_summary.json`

Wichtig:

- ohne `--force` werden bereits fertige Fine-Tune-Profile sauber uebersprungen
- `python 12_train_fine_tune_models.py --force` trainiert vorhandene Fine-Tune-Profile bewusst neu
- im aktuellen Stand sind das lokale Fine-Tune-Checkpoints und Zielschritt-Profile, noch keine echten Modellgewichts-Fine-Tunes

### 13 - Konkrete Backend-Fine-Tune-Laeufe ableiten

`python 13_run_backend_finetunes.py`

Dieser Schritt:

- liest die Fine-Tune-Zusammenfassung aus `12`
- erzeugt daraus konkrete Backend-Laufdeskriptoren pro Figur und Modalitaet
- mappt Bild, Video und Stimme auf konkrete Backend-Typen wie `lora-image`, `motion-adapter` oder `speaker-adapter`
- schreibt pro Figur einen `backend_fine_tune_run.json`
- schreibt eine globale Backend-Zusammenfassung fuer den aktuellen Lauf

Wichtige Ausgaben:

- `training/foundation/backend_runs/<figur>/backend_fine_tune_run.json`
- `training/foundation/backend_runs/backend_fine_tune_summary.json`

Wichtig:

- ohne `--force` werden bereits fertige Backend-Laeufe sauber uebersprungen
- `python 13_run_backend_finetunes.py --force` erzeugt vorhandene Backend-Laeufe bewusst neu
- im aktuellen Stand sind das konkrete lokale Run-Deskriptoren fuer spaetere echte Backend-Anbindungen, noch keine echten externen Trainingsjobs

### 14 - Neue Folge aus trainiertem Modell erzeugen

`python 14_generate_episode_from_trained_model.py`

Dieser Schritt:

- liest das bereits trainierte `series_model.json`
- verlangt davor frisches Foundation-Training
- kann optional zusaetzlich aktuelles Adapter-Training verlangen, wenn `adapter_training.required_before_generate=true` gesetzt ist
- erzeugt daraus eine neue Folge als Markdown
- erzeugt eine passende Shotlist als JSON
- vergibt zusaetzlich einen sichtbaren Episodentitel wie `Folge 05: Das Baumhaus-Spiel`
- leitet die Ziel-Laufzeit neuer Folgen aus den verarbeiteten Quellfolgen ab
- markiert neue Folgen in der Shotlist als `synthetic_preview`, damit sie nicht automatisch als Original-Remix gerendert werden

Wichtige Ausgaben:

- `generation/story_prompts/folge_XX.md`
- `generation/shotlists/folge_XX.json`

Die technische Datei-ID bleibt absichtlich `folge_XX`, damit die Pipeline stabil bleibt. Der sichtbare Titel steht zusaetzlich in `episode_title`, `episode_label` und `display_title`.

Wichtig:

- solange noch keine Figuren manuell benannt wurden, verwendet `14` bewusst generische Platzhalter wie `Hauptfigur A` und `Hauptfigur B` statt technische Namen wie `face_001` oder `speaker_001`
- Figuren mit dem Namen `statist` koennen spaeter in Szenen als Nebenfigur auftauchen, werden aber nicht als Hauptfigur ausgewaehlt
- neue Folgen variieren jetzt pro Episoden-Index sichtbar, statt bei jedem Lauf dieselbe Struktur zu wiederholen
- priorisierte benannte Figuren werden bevorzugt als Hauptfiguren ausgewaehlt
- im Standardpfad erzeugt `14` Preview-Folgen synthetisch aus dem trainierten Modell und nicht aus geplanten Originalsegmenten
- nur wenn `generation.prefer_original_dialogue_remix=true` gesetzt ist, plant `14` Dialogzeilen wieder bevorzugt aus echten vorhandenen Originalsegmenten

### 06 - Face-Cluster benennen und Review pruefen

`python 06_review_unknowns.py`

Dieser Schritt kann jetzt drei Dinge:

- ohne Zusatzparameter direkt die Face-Review fuer unbenannte Cluster starten
- offene Review-Faelle aus `review_queue.json` anzeigen
- Face-Cluster manuell benennen
- bereits benannte Face-Cluster gezielt umbenennen
- Face-Cluster mit `noface` markieren, damit sie kuenftig ignoriert werden

Wichtige Beispiele:

- `python 06_review_unknowns.py --list-faces`
- `python 06_review_unknowns.py --created`
- `python 06_review_unknowns.py`
- `python 06_review_unknowns.py --all`
- `python 06_review_unknowns.py --assign-face face_001 --name "Babe Carano"`
- `python 06_review_unknowns.py --assign-face face_022 --name "Babe Carano" --priority`
- `python 06_review_unknowns.py --set-priority "Babe Carano"`
- `python 06_review_unknowns.py --clear-priority "Babe Carano"`
- `python 06_review_unknowns.py --assign-face face_014 --name "Mr. Sammich"`
- `python 06_review_unknowns.py --assign-face face_023 --name "Teague/Busboy"`
- `python 06_review_unknowns.py --assign-face face_055 --name "statist"`
- `python 06_review_unknowns.py --rename-face face_023 --rename-to "Teague"`
- `python 06_review_unknowns.py --rename-face "Mr. Sammich" --rename-to "Mr. Sammich Sr."`
- `python 06_review_unknowns.py --assign-face face_041 --ignore`
- `python 06_review_unknowns.py --review-faces --open-previews`
- `python 06_review_unknowns.py --show-queue`

Bedeutung:

- beim einfachen Start gleicht `06` zuerst offene Face-Cluster gegen bereits bekannte benannte Figuren ab und nutzt dafuer jetzt mehrere hochwertige Referenz-Cluster pro Figur statt nur einen einzelnen Figuren-Centroid
- danach werden wirklich nur noch die uebrigen unbekannten Face-Cluster nacheinander gezeigt
- die Review startet jetzt mit den haeufigsten wiederkehrenden Face-Clustern zuerst
- `06` arbeitet nicht mehr nur einen alten 25er-Snapshot ab: nach jeder Namensvergabe wird sofort neu gescannt, bekannte Gesichter werden erneut automatisch zusammengefuehrt und erst dann der naechste wirklich offene Fall gezeigt
- nach jeder Namensvergabe laeuft dieser Auto-Lernschritt jetzt mehrfach bis zum Stillstand, damit nicht nur ein einzelner weiterer Treffer, sondern moeglichst viele unmittelbar passende offene Face- und Sprecher-Faelle direkt in derselben Session verschwinden
- der automatische Vorab-Abgleich in `06` arbeitet jetzt qualitativ aehnlicher wie moderne Foto-Manager: Referenz-Cluster werden nach Qualitaet gewichtet, mehrere Referenzen derselben Figur werden gemeinsam ausgewertet und ein Treffer wird nur uebernommen, wenn Score, Abstand zur zweitbesten Figur und Konsens ueber mehrere Referenzen passen
- das Standardlimit fuer `06` ist jetzt `20` offene Face-Cluster pro normalem Start
- dieses `20`er-Limit ist jetzt ein echtes Session-Budget: nach jedem bearbeiteten Fall sinkt die Session-Anzeige wirklich herunter und `06` fuellt die Session nach Re-Scans nicht wieder heimlich auf `20` auf
- sobald diese Session-Restmenge `0` erreicht, beendet `06` den normalen Lauf sauber; nur `python 06_review_unknowns.py --all` bearbeitet danach wirklich weiter alle noch offenen Faelle
- `--list-faces` zeigt standardmaessig nur bereits benannte Figuren
- `--created` zeigt nur die bereits erstellten echten Figurennamen als einfache Liste wie `Babe`, `Kenzie` oder `Hudson`
- `06` normalisiert beim Start alte Platzhalterdaten automatisch zurueck auf `face_###` und `speaker_###`, statt fruehere Auto-Namen weiterzutragen
- wenn derselbe Figurenname mehrfach vergeben wird, bleiben alle diese Face-Cluster bewusst unter derselben Figur gruppiert; ein weiterer `Babe`-Treffer erhoeht also die Face-Anzahl fuer `Babe`, statt eine neue Figur zu erzeugen
- `character_map.json` speichert diese Gruppierung jetzt auch explizit unter `identities` sowie pro Cluster ueber `identity_name`, `identity_primary_cluster`, `identity_cluster_ids` und `identity_cluster_count`
- pro Cluster wird eine Montage mit Szene links und Ausschnitt rechts erzeugt und in einer Vorschau mit direkter Eingabe gezeigt
- pro Cluster werden auch `Szenen` und `Treffer` angezeigt, damit Hauptfiguren schneller erkennbar sind
- pro offenem Cluster zeigt `06` jetzt zusaetzlich automatische Rollenhinweise wie `hauptfigur-kandidat`, `nebenfigur-kandidat` oder `statist-kandidat` samt Review-Tipp an
- pro benannter Figur zeigt `06` jetzt auch `Figuren-Faces`, also wie viele unterschiedliche Face-Cluster bereits derselben Figur zugeordnet sind
- im Vorschaufenster werden jetzt gleichzeitig die Session-Restmenge und die tatsaechlich insgesamt noch offenen Face-Cluster laufend angezeigt
- der naechste Face-Cluster wird erst nach einer echten Benennung oder `noface` geoeffnet
- waehrend der Review werden Beispielnamen wie `Babe Carano`, `Mr. Sammich`, `Teague/Busboy`, `noface = ignorieren` und `statist = statist` direkt eingeblendet
- Namen koennen jetzt direkt im Vorschaufenster eingegeben und mit Enter uebernommen werden
- im Vorschaufenster gibt es zusaetzlich eine direkte Hauptfiguren-Checkbox
- im Vorschaufenster gibt es jetzt ausserdem Schnellwahl-Buttons fuer bereits bekannte Figuren wie `Babe`, `Kenzie` oder `Double G`, damit du wiederkehrende Namen direkt anklicken kannst
- wenn eine Figur einmal als Hauptfigur markiert wurde, bleibt diese Prioritaet fuer dieselbe Figur automatisch erhalten; spaetere gleichnamige Cluster uebernehmen sie auch dann, wenn das Haekchen nicht erneut gesetzt wird
- unter Windows kann derselbe Name alternativ auch direkt im Terminal eingegeben werden, waehrend die Vorschau offen ist; nach Enter schliesst die Vorschau automatisch und die naechste Figur folgt
- im Terminal kann `!Name` direkt als priorisierte Hauptfigur gespeichert werden
- wenn das Vorschaufenster nur geschlossen wird, faellt `08` automatisch auf die normale Terminal-Eingabe fuer denselben Cluster zurueck
- `--assign-face ... --name ...` speichert einen dauerhaften Figurennamen in `character_map.json`
- `--priority` markiert die Figur als bevorzugte Hauptfigur fuer `07`
- `--set-priority ...` und `--clear-priority ...` aendern die Hauptfiguren-Priorisierung spaeter auch ohne Umbenennung
- `--rename-face ... --rename-to ...` benennt einen vorhandenen Charakter per Face-ID oder aktuellem Namen um
- `--assign-face ... --ignore` setzt den Namen intern auf `noface`
- `--assign-face ... --name "statist"` speichert bewusst eine Nebenfigur, die spaeter sichtbar sein darf, aber nicht als Hauptrolle zaehlt
- `--review-faces` geht interaktiv durch automatisch benannte Cluster
- `--all` bedeutet jetzt ausdruecklich: wirklich alle aktuell offenen Face-Cluster bearbeiten
- `--created` ist die schnelle reine Namensliste aller bereits benannten Figuren
- `--show-queue` zeigt nur die offene Sprecher-/Segment-Review an
- nach einer Namensvergabe werden `voice_map.json`, alle vorhandenen `linked_segments.json` und die `review_queue.json` direkt aktualisiert
- dabei werden jetzt auch offene Sprecher-Reviews konservativ direkt weiter reduziert, wenn fuer einen Sprecher ueber mehrere offene Segmente hinweg immer dieselbe einzelne benannte Figur sichtbar ist
- die Grenzwerte fuer den automatischen Vorab-Abgleich liegen jetzt in `character_detection.review_known_face_threshold`, `review_known_face_margin`, `review_known_face_reference_count`, `review_known_face_top_k`, `review_known_face_consensus_threshold`, `review_known_face_min_consensus`, `review_known_face_strong_match_threshold` und `review_known_face_min_reference_quality`

### 15 - Serienbibel aktualisieren

`python 15_build_series_bible.py`

Dieser Schritt:

- liest das trainierte `series_model.json`
- erstellt daraus eine kompakte Serienbibel als JSON und Markdown
- fasst Figuren, Sprecher, haeufige Keywords und Datensatzquellen zusammen

Wichtige Ausgaben:

- `series_bible/episode_summaries/auto_series_bible.json`
- `series_bible/episode_summaries/auto_series_bible.md`

### 16 - Draft-Video rendern

`python 16_render_episode.py`

Dieser Schritt:

- nimmt standardmaessig die neueste Shotlist
- kann ueber `SERIES_RENDER_EPISODE` gezielt auf eine Folge zeigen
- erzeugt Kartenbilder, TTS-Audio und MP4-Segmente
- uebernimmt sichtbare Episodentitel aus `07` in Shotlist und Render-Manifest
- sammelt fuer benannte Figuren automatisch Referenzaudio aus vorhandenen Sprechersegmenten
- rendert Shotlists mit `generation_mode=synthetic_preview` bewusst ohne automatischen Originalsegment-Reuse
- versucht Originalsegment-Reuse nur noch, wenn dieser explizit in der Config aktiviert wurde und die Shotlist nicht als synthetische Preview markiert ist
- startet nur noch, wenn das Foundation-Training frischer als das aktuelle Serienmodell ist
- prueft vor dem Rendern, ob fuer die Fokusfiguren trainierte Foundation-/Voice-Packs vorhanden sind
- rendert standardmaessig mit lokalem `pyttsx3`
- versucht XTTS-Voice-Cloning nur noch bei ausdruecklich passender Config und expliziter Lizenzfreigabe
- bevorzugt beim lokalen System-TTS auf Windows aktiv vorhandene deutsche Stimmen und verwirft alte englische Profile, wenn eine deutsche Stimme verfuegbar ist
- erzeugt fuer benannte Figuren sonst ein referenzbasiertes Talking-Head-/Lip-Sync-Preview aus mehreren echten Face-Frames
- faellt fuer unbekannte Figuren oder fehlende Assets auf statische Karten + Standard-TTS zurueck
- setzt alles zu einem Draft-Video zusammen
- schreibt zusaetzlich ein `final`-MP4 nach `generation/renders/final/<folge>`
- schreibt ausserdem ein Render-Manifest in Draft und Final
- schreibt pro benannter Figur ein lokales `voice_model`-JSON unter `characters/voice_models`
- nutzt fuer den finalen Zusammenschnitt jetzt eine FFmpeg-Concat-Datei statt einer extrem langen Eingabeliste, damit Windows nicht mehr an zu langen Render-Kommandos scheitert

Wenn der installierte FFmpeg-Encoder es unterstuetzt und GPU-Nutzung aktiviert ist, wird fuer das Rendern automatisch NVENC verwendet. Sonst faellt `16` auf CPU-Encoding zurueck.
Wenn XTTS lokal verfuegbar ist, wird fuer das Voice Cloning ebenfalls das bevorzugte Torch-Geraet verwendet. Die eingebaute Lip-Sync-Variante ist ein audio-reaktiver Preview-Modus auf Basis der erkannten Face-Crops und kein externes Wav2Lip-/SadTalker-Setup.
Ohne lizenzpflichtiges Clone-Modell bleibt die Stimme trotzdem ein lokaler TTS-Fallback; zusaetzlich werden jetzt aber pro benannter Figur Referenz-WAVs und ein lokales Stimmprofil unter `characters/voice_samples` aufgebaut.

Beispiel fuer gezielten Render in PowerShell:

```powershell
$env:SERIES_RENDER_EPISODE='folge_02'
python 16_render_episode.py
Remove-Item Env:SERIES_RENDER_EPISODE
```

### 17 - Mehrere sichtbare Folgen erzeugen

`python 17_generate_preview_episodes.py`

Dieser Schritt:

- baut zuerst den Datensatz aus dem aktuellen reviewten Bestand neu auf
- trainiert danach das Serienmodell
- bereitet danach automatisch den Foundation-Trainingssatz vor
- trainiert danach automatisch die Foundation-Packs
- trainiert danach automatisch auch die lokalen Adapter-Profile
- trainiert danach automatisch auch die lokalen Fine-Tune-Profile
- erzeugt danach automatisch auch die konkreten Backend-Fine-Tune-Laeufe
- erzeugt erst danach mehrere neue Folgen hintereinander
- rendert jede neue Folge direkt als Draft-MP4
- rendert jede neue Folge direkt auch als Final-MP4
- aktualisiert danach die Serienbibel
- schreibt zusaetzlich einen globalen Schritt-Autosave fuer den gesamten Multi-Episoden-Lauf

Beispiele:

- `python 17_generate_preview_episodes.py`
- `python 17_generate_preview_episodes.py --count 3`

Wichtig:

- Standard sind `2` neue sichtbare Folgen pro Lauf
- die Reihenfolge ist jetzt hart: `07 -> 08 -> 09 -> 10 -> 11 -> 12 -> 13 -> 14 -> 16 -> 15`
- gerendert wird jeweils direkt die gerade neu erzeugte Folge
- die Draft-Videos landen wie gewohnt unter `generation/renders/drafts/folge_XX`
- die Final-Videos liegen unter `generation/renders/final/folge_XX`

Beispiel fuer XTTS nach eigener Lizenzbestaetigung:

```powershell
$env:SERIES_ACCEPT_COQUI_LICENSE='1'
python 16_render_episode.py
Remove-Item Env:SERIES_ACCEPT_COQUI_LICENSE
```

Wichtige Ausgaben:

- `generation/renders/drafts/<folge>/<folge>_draft.mp4`
- `generation/renders/final/<folge>/<folge>_final.mp4`
- `generation/renders/drafts/<folge>/<folge>_render_manifest.json`
- `generation/renders/final/<folge>/<folge>_render_manifest.json`
- `characters/voice_models/<figur>_voice_model.json`

Wichtig:

- das Ergebnis ist ein Storyboard-/Preview-Render mit optionalem Voice Cloning und referenzbasiertem Talking Head
- unbekannte oder nicht benannte Figuren bleiben bei statischen Karten
- im Standardpfad bleibt der Voice-Teil bewusst bei lokalem `pyttsx3`
- ohne ausdruecklich aktivierten XTTS-Pfad bleibt der Voice-Cloning-Teil ebenfalls bewusst im Fallback
- keine echte vollautomatische Video-Generierung auf Produktionsniveau

### 99 - Alles in einem Lauf

`python 99_process_next_episode.py`

Fuehrt diese Schritte nacheinander aus:

- `01_setup_project.py`
- danach fuer jede neue noch unregistrierte Inbox-Folge einzeln:
- `02_import_episode.py`
- `03_split_scenes.py --episode-file <datei>`
- `04_diarize_and_transcribe.py --episode <folgenordner>`
- `05_link_faces_and_speakers.py --episode <folgenordner>`
- danach manuelle Review:
- `06_review_unknowns.py`
- erst danach global:
- `07_build_dataset.py`
- `08_train_series_model.py`
- `09_prepare_foundation_training.py`
- `10_train_foundation_models.py`
- `11_train_adapter_models.py`
- `12_train_fine_tune_models.py`
- `13_run_backend_finetunes.py`
- `14_generate_episode_from_trained_model.py`
- `15_build_series_bible.py`
- `16_render_episode.py`

Wichtig:

- `99` arbeitet jetzt alle neuen Quellfolgen im Inbox-Ordner nacheinander ab, nicht nur die erste
- wenn z. B. `60` neue `mp4`-Folgen in `data/inbox/episodes` liegen, werden diese `60` Folgen nacheinander importiert, gesplittet, transkribiert und verknuepft
- jede einzelne Inbox-Folge wird bereits in `02_import_episode.py` entfernt, sobald ihre Arbeitskopie erfolgreich in `data/raw/episodes` liegt
- erst nachdem alle neuen Quellfolgen bis `05` vorverarbeitet wurden und keine offenen Review-Faelle fuer `06_review_unknowns.py` mehr uebrig sind, werden `07`, `08`, `09`, `10`, `11`, `12`, `13`, `14`, `15` und `16` einmal fuer den aktualisierten Gesamtstand ausgefuehrt
- nach jedem erfolgreich abgeschlossenen Schritt schreibt `99` einen Autosave unter `ai_series_project/runtime/autosaves/99_process_next_episode`
- parallel schreibt `99` dort jetzt auch `current_status.json` und `current_status.md` mit aktuellem Episodenstatus, Globalstatus, Rest-Inbox und laufendem Schritt
- es bleiben dabei bewusst maximal die letzten `2` Autosaves erhalten
- wenn `99` unterwegs abbricht, nimmt der naechste Start automatisch den letzten gueltigen Autosave und setzt genau dort wieder an
- `99` trainiert bewusst nicht ueber offene Review-Faelle hinweg; wenn nach `05` noch offene Eintraege in der Review-Queue liegen, stoppt der Lauf sauber und fordert zuerst `06_review_unknowns.py`
- `14` und `16` blocken jetzt bewusst, wenn Training fehlt oder aelter als das aktuelle Serienmodell ist; `11`, `12` und `13` sind im Standardpfad ebenfalls Pflicht vor Generierung/Render
- wenn keine neue Inbox-Folge vorhanden ist, beendet sich `99` sauber ohne neuen Generierungsdurchlauf

### 19 - GitHub synchronisieren (optional)

`python 19_sync_to_github.py`

Dieser Schritt:

- initialisiert bei Bedarf ein lokales Git-Repository
- synchronisiert ausschliesslich Root-`*.py`-Skripte plus `README.md`
- ignoriert alle Projektordner und deren Inhalte vollstaendig
- ignoriert den lokalen Sync-Helfer selbst absichtlich, damit er nicht mit hochgeladen wird
- laedt niemals Inhalte von GitHub herunter und fuehrt bewusst kein `clone`, `fetch` oder `pull` aus
- aktualisiert bei jedem Lauf auch die GitHub-`About`-Beschreibung
- prueft vor dem Push, dass bei Script-Aenderungen auch `README.md` mitgeaendert wurde
- verwendet standardmaessig das Repository `https://github.com/tyskman11/ai-series-generator`
- verwendet fuer neue lokale Commits standardmaessig die E-Mail `baumscarry@gmail.com`
- legt bei vorhandenem `GITHUB_TOKEN` ein fehlendes GitHub-Repo automatisch an
- erstellt bei neuen Aenderungen einen Commit und spiegelt den lokalen Stand ueber die GitHub-Git-API nach GitHub

Wichtig:

- lokal ist immer die Quelle der Wahrheit fuer diesen Helfer
- die GitHub-`About`-Beschreibung sagt kurz, dass das Projekt aus Serienfolgen neue KI-Preview-Episoden ableiten soll und dass alle Skripte mit `GPT-5 Codex` KI-generiert wurden
- wenn der Remote-Branch andere Historie oder andere Dateien hat, wird nichts heruntergeladen; stattdessen schreibt die GitHub-API den Ziel-Branch auf den lokalen erlaubten Dateistand um

Wichtige Umgebungsvariablen:

- `GITHUB_TOKEN`: persoenliches GitHub-Token fuer Repo-Erstellung und Push ohne gespeicherte Credentials
- `GITHUB_OWNER`: optionaler Override fuer den Standard-Owner `tyskman11`
- `GITHUB_REPO`: optionaler Override fuer das Standard-Repo `ai-series-generator`
- `GITHUB_PRIVATE`: `true` oder `false` fuer neue Repositories
- `GIT_USER_NAME`: optionaler Git-Name fuer lokale Commits
- `GIT_USER_EMAIL`: optionale Git-E-Mail fuer lokale Commits, Standard `baumscarry@gmail.com`

Nuetzliche Beispiele:

- nur pruefen, was passieren wuerde: `python 19_sync_to_github.py --dry-run`
- mit gespeicherten Git-Credentials spiegeln: `python 19_sync_to_github.py`
- mit GitHub-API-Token spiegeln: `set GITHUB_TOKEN=<token>` und danach `python 19_sync_to_github.py`

### 18 - Rebuild nach manueller Figuren-Review

`python 18_refresh_after_manual_review.py`

Dieser Schritt:

- baut nach manueller Figurenpflege den kompletten abhaengigen Stand bewusst neu auf
- startet dafuer in fester Reihenfolge:
- `07_build_dataset.py --force`
- `08_train_series_model.py`
- `09_prepare_foundation_training.py --force`
- `10_train_foundation_models.py --force`
- `11_train_adapter_models.py --force`
- `12_train_fine_tune_models.py --force`
- `13_run_backend_finetunes.py --force`
- danach optional auch:
- `14_generate_episode_from_trained_model.py`
- `15_build_series_bible.py`
- `16_render_episode.py`

Nuetzliche Varianten:

- `python 18_refresh_after_manual_review.py --skip-downloads`
- `python 18_refresh_after_manual_review.py --stop-after-training`

Wichtig:

- dieser Helfer ist genau fuer den Fall gedacht, dass nach `06` viele Figuren sauber benannt oder priorisiert wurden
- damit muessen `07`, `08`, `09`, `10`, `11`, `12` und `13` nicht mehr einzeln mit `--force` von Hand hintereinander gestartet werden

## Nuetzliche Debug- und Test-Hinweise

### Szene-Limit fuer Smoke-Tests

Viele Schritte respektieren `SERIES_MAX_SCENES`.

Beispiel:

```powershell
$env:SERIES_MAX_SCENES='20'
python 04_diarize_and_transcribe.py
Remove-Item Env:SERIES_MAX_SCENES
```

Das ist hilfreich fuer kurze Testlaeufe.

### Tests und Smoke-Runs

Fuer den aktuellen Stand gibt es zusaetzlich einen kleinen automatisierten Testblock:

```powershell
python -m unittest discover -s tests -v
```

Aktuell geprueft sind unter anderem:

- Platzhaltererkennung fuer manuelle gegen technische Namen
- Stimmenzuordnung mit und ohne manuell benannten Face-Cluster
- Normalisierung alter `figur_###`-/`stimme_###`-Artefakte
- generische Episoden-Fallbacks, solange noch keine Figuren benannt wurden
- `statist` als Nebenfiguren-Status ohne Hauptrollen-Promotion
- unterschiedliche Episoden-Ausgabe fuer unterschiedliche Folgen-Indizes
- Hauptfiguren-Priorisierung bei der Benennung
- nachtraegliches Priorisieren und Entpriorisieren benannter Figuren

Empfohlener Smoke-Run nach groesseren Aenderungen:

```powershell
python 05_link_faces_and_speakers.py
python 07_build_dataset.py
python 08_train_series_model.py
python 09_prepare_foundation_training.py --skip-downloads
python 10_train_foundation_models.py
python 11_train_adapter_models.py
python 12_train_fine_tune_models.py
python 13_run_backend_finetunes.py
python 14_generate_episode_from_trained_model.py
python 15_build_series_bible.py
python 16_render_episode.py
```

Wenn nach `06` viele Figuren neu benannt oder priorisiert wurden, ist statt der einzelnen Rebuild-Schritte jetzt dieser eine Lauf gedacht:

```powershell
python 18_refresh_after_manual_review.py --skip-downloads
```

### Wiederholung von Schritt 05

Wenn `05` einfach noch einmal laufen soll:

```powershell
python 05_link_faces_and_speakers.py
```

Wenn `05` wirklich neu rechnen soll:

```powershell
python 05_link_faces_and_speakers.py --fresh
```

Wenn gezielt eine Folge neu gerechnet werden soll:

```powershell
python 05_link_faces_and_speakers.py --fresh --episode "Game.Shakers.S01E01.GERMAN.720p.WEB.H264-BiMBAMBiNO"
```

### Tools

`ffmpeg` wird zuerst im Projektpfad erwartet. Falls dort nichts liegt, gibt es einen Fallback auf den Root-`tools`-Ordner und danach auf `ffmpeg` im System-`PATH`.

## Bekannte Grenzen

- `03`, `04`, `05` und `06` arbeiten inzwischen stapelweise ueber offene Folgen, aber die wirklich saubere Qualitaet der Zuordnung haengt weiter stark an manueller Figurenpflege in `08`
- `05 --fresh` bleibt bewusst ein gezielter Einzel-Folgen-Modus, weil `character_map.json` und `voice_map.json` global sind
- `character_map.json` und `voice_map.json` sind global und noch nicht pro Folge getrennt.
- Die Sprecherzuordnung ist heuristisch, nicht diarization-grade production quality.
- Auch nach dem neuen Filter koennen in `05` noch Nebenfiguren oder Hintergrundgesichter als eigene Cluster auftauchen.
- Solange Hauptfiguren noch nicht manuell benannt wurden, erzeugt `07` generische Hauptrollen (`Hauptfigur A/B`) statt echte Rollennamen.
- Das Trainingsmodell in `07` ist regel-/datengetrieben und nicht mit einem grossen multimodalen KI-Modell vergleichbar.
- `16` rendert weiter lokal und modular; auch das `final`-MP4 ist noch keine vollwertige TV-Endproduktion.
- das `final`-MP4 nutzt aktuell denselben lokalen Render-Inhalt wie der Draft; es ist ein sauber abgelegter Final-Export, aber noch keine qualitativ andere Filmfassung.
- Die beste lokale Stimmennaehe kommt jetzt aus der Wiederverwendung passender Originalsegmente; fuer komplett neue Saetze ohne guten Treffer bleibt der Fallback eine generierte Stimme und klingt deshalb nicht identisch zur Originalfigur.
- Das eingebaute Lip-Sync in `16` ist jetzt ohne Cartoon-Mund, aber weiterhin ein lokaler Fallback und nicht dieselbe Qualitaet wie spezialisierte Deepfake-/Lip-Sync-Modelle.
- XTTS-Voice-Cloning funktioniert nur fuer Figuren mit brauchbarem Referenzaudio und installiertem `TTS`-Paket.
- XTTS braucht zusaetzlich eine von dir explizit bestaetigte Coqui-Lizenzfreigabe; ohne diese rendert `16` absichtlich weiter mit Fallback-TTS.
- `large-v3` auf CPU ist moeglich, aber deutlich langsamer als auf GPU.

## Empfohlene Pflege bei Code-Aenderungen

Wenn ein Skript geaendert wird, sollte mindestens geprueft werden, ob die README in einem dieser Bereiche angepasst werden muss:

- Befehl zum Starten
- neue oder entfernte CLI-Optionen
- neue Environment-Variablen
- geaenderte Eingabepfade
- geaenderte Ausgabepfade
- neue Cache- oder Reset-Logik
- geaenderte Reihenfolge im Gesamtworkflow
- neue Einschraenkungen oder bekannte Probleme

## Minimaler Tagesbetrieb

Wenn einfach nur eine neue Folge verarbeitet werden soll:

```powershell
python 00_prepare_runtime.py
python 01_setup_project.py
python 99_process_next_episode.py
```

Wenn nur neue Episoden auf Basis des vorhandenen Datensatzes erzeugt werden sollen:

```powershell
python 08_train_series_model.py
python 09_prepare_foundation_training.py --skip-downloads
python 10_train_foundation_models.py
python 11_train_adapter_models.py
python 12_train_fine_tune_models.py
python 13_run_backend_finetunes.py
python 14_generate_episode_from_trained_model.py
python 16_render_episode.py
```

Wenn nach manueller Figurenbenennung alles sauber mit den neuen Namen neu aufgebaut werden soll:

```powershell
python 18_refresh_after_manual_review.py --skip-downloads
```

Wenn direkt mehrere neue sichtbare Folgen erzeugt werden sollen:

```powershell
python 17_generate_preview_episodes.py --count 2
```

