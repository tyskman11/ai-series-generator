# KI Serien Training

## Zweck

Dieses Projekt verarbeitet vorhandene Serienfolgen als `mp4` und baut daraus eine lokale Pipeline fuer:

- Import von Folgen
- Szenenschnitt
- Transkription und einfache Sprecher-Cluster
- Gesichtserkennung und Verknuepfung von Figuren und Stimmen
- Aufbau eines Trainingsdatensatzes
- Ableitung eines lokalen Serienmodells
- Generierung neuer Folgen als Markdown und Shotlist
- Rendern eines Storyboard-/TTS-Draft-Videos
- optionales Voice Cloning aus vorhandenen Sprechersegmenten
- referenzbasiertes Face-Cloning aus erkannten Face-Crops
- eingebautes Lip-Sync-/Talking-Head-Fallback fuer benannte Figuren

Die Pipeline ist modular. Jeder Schritt kann einzeln gestartet werden, oder alles laeuft ueber `99_process_next_episode.py`.

## Wichtige Dokumentationsregel

Diese Datei ist ein Pflichtdokument.

Bei jeder Aenderung an:

- `00_prepare_runtime.py` bis `10_render_episode.py`
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
- daraus ein Draft-Video mit Karten und TTS rendern
- pro benannter Figur Referenzbilder und Referenzaudio fuer spaetere Clone-Schritte sammeln
- bei verfuegbarem `TTS`-Paket automatisch XTTS-basiertes Voice Cloning versuchen
- fuer benannte Figuren aus Face-Crops ein audio-reaktives Talking-Head-/Lip-Sync-Preview rendern

Das Projekt kann aktuell noch nicht:

- vollstaendig neue animierte TV-Folgen generieren
- produktionsreifes Deepfake-Lipsync auf Spielfilm-Niveau
- perfekte stiltreue Voice-Clones fuer jede Figur ohne gutes Referenzmaterial
- stiltreue Video-Generierung auf Produktionsniveau

`10_render_episode.py` erzeugt jetzt ein erweitertes Preview-Video mit statischem Karten-Fallback, optionalem XTTS-Voice-Cloning und eingebautem audio-reaktivem Face-/Lip-Sync-Fallback, aber weiterhin keine finale KI-TV-Episode.

## Projektstruktur

Wichtige Root-Dateien:

- `00_prepare_runtime.py`: Runtime und Pakete vorbereiten
- `01_setup_project.py`: Projektstruktur anlegen
- `02_import_episode.py`: naechste Folge aus der Inbox importieren
- `03_split_scenes.py`: importierte Folge in Szenen zerlegen
- `04_diarize_and_transcribe.py`: Audio extrahieren, transkribieren, Sprecher clustern
- `05_link_faces_and_speakers.py`: Gesichter erkennen und mit Stimmen verknuepfen
- `06_build_dataset.py`: Trainingsdatensatz bauen
- `07_generate_episode.py`: Serienmodell trainieren und neue Folge erzeugen
- `08_review_unknowns.py`: offene Zuordnungen anzeigen
- `09_build_series_bible.py`: Serienbibel aktualisieren
- `10_render_episode.py`: Storyboard-/TTS-Draft rendern
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
- `pyttsx3`
- `TTS` (optional fuer XTTS-Voice-Cloning)

## Schnellstart

1. Neue Folgen als `mp4` nach `ai_series_project/data/inbox/episodes` legen.
2. `python 00_prepare_runtime.py`
3. `python 01_setup_project.py`
4. `python 99_process_next_episode.py`
5. Ergebnisse in `generation/story_prompts`, `generation/shotlists`, `generation/renders/drafts` und `series_bible/episode_summaries` pruefen.
6. Optional den lokalen Sync-Helfer starten, um nur Root-Skripte und `README.md` nach GitHub zu pushen.

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
Wichtig fuer XTTS: Das Coqui-Modell darf hier nicht stillschweigend mit einer Lizenzbestaetigung gestartet werden. XTTS wird deshalb erst benutzt, wenn du die Lizenz selbst bestaetigt und danach `cloning.xtts_license_accepted=true` in der Config gesetzt oder beim Rendern `SERIES_ACCEPT_COQUI_LICENSE=1` gesetzt hast.

Ohne `00` sollten die anderen Schritte nicht gestartet werden.

### 01 - Projektstruktur erzeugen

`python 01_setup_project.py`

Dieser Schritt stellt sicher, dass die gesamte Projektstruktur und die Config-Datei vorhanden sind.

### 02 - Folge importieren

`python 02_import_episode.py`

Dieser Schritt:

- nimmt die erste noch nicht verarbeitete Datei aus `data/inbox/episodes`
- kopiert sie nach `data/raw/episodes`
- schreibt Metadaten nach `data/processed/metadata`
- markiert die Datei in `logs/processing_registry.json` als importiert

Wichtig:

- `02` verarbeitet immer nur die naechste unregistrierte Datei
- eine `mp4` direkt im Root-Ordner wird nicht automatisch verarbeitet

### 03 - Szenen schneiden

`python 03_split_scenes.py`

Dieser Schritt:

- nimmt die erste importierte Folge aus `data/raw/episodes`
- erkennt Szenen mit `scenedetect`
- exportiert je Szene eine `scene_XXXX.mp4`
- schreibt die Szenenliste als CSV nach `data/processed/scene_index`

Wenn keine Szene erkannt wird, faellt der Schritt auf feste Segmente anhand von `default_scene_seconds_fallback` zurueck.

Wichtig:

- wenn bereits ein Szenenordner existiert, beendet sich `03` ohne Neuberechnung
- wenn `delete_input_after_split` auf `true` steht, werden Inbox-Kopie und Arbeitskopie nach erfolgreichem Split geloescht

### 04 - Audio, Transkription, Sprecher-Cluster

`python 04_diarize_and_transcribe.py`

Dieser Schritt:

- nimmt den ersten Szenenordner unter `data/processed/scene_clips`
- exportiert Audio pro Szene nach `data/raw/audio`
- transkribiert jede Szene mit Whisper
- schneidet daraus Segmente
- berechnet einfache Voice-Embeddings
- bildet Sprecher-Cluster (`speaker_001`, `speaker_002`, ...)

Wichtige Ausgaben:

- `data/processed/speaker_segments/<folge>/...`
- `data/processed/speaker_transcripts/<folge>_segments.json`
- `data/processed/speaker_segments/<folge>/_speaker_clusters.json`

Wichtig:

- Whisper verwendet standardmaessig `large-v3`
- bei nutzbarer CUDA-GPU wird Whisper direkt auf GPU geladen
- dieser Schritt kann sehr lange dauern
- vorhandene Cache-Dateien werden wiederverwendet, solange die interne `process_version` passt

### 05 - Gesichter und Stimmen verknuepfen

`python 05_link_faces_and_speakers.py`

Dieser Schritt:

- liest die Sprechersegmente aus `04`
- erkennt Gesichter in den Szenen ueber `facenet-pytorch` mit OpenCV-Fallback
- bildet Face-Cluster (`face_001`, `face_002`, ...)
- konsolidiert Mehrfacherkennungen zuerst lokal innerhalb einer Szene
- ordnet sichtbare Face-Cluster zeitbasiert pro Dialogsegment zu
- verknuepft diese segmentnahen Face-Cluster mit Sprecher-Clustern
- uebernimmt manuell vergebene Figurennamen aus `character_map.json`
- ignoriert Cluster mit dem Namen `noface` kuenftig automatisch
- baut `character_map.json`, `voice_map.json`, `linked_segments.json` und `review_queue.json`

Wichtige technische Details:

- wenn eine CUDA-GPU nutzbar ist, laufen MTCNN und Face-Embeddings auf GPU
- wenn MTCNN in einzelnen Frames keine Box liefert, faellt `05` auf OpenCV-Haar-Cascades zurueck
- `sample_every_n_frames`, `max_faces_per_frame`, `max_scene_clusters`, `max_visible_faces_per_segment`, `segment_visibility_padding_seconds`, `min_face_size`, `embedding_threshold`, `scene_embedding_threshold` und `detection_confidence_threshold` kommen aus der Projekt-Config

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
- `--fresh`: loescht die bisherigen `05`-Artefakte vor dem Lauf
- `--episode`: waehlt gezielt einen Szenenordner

Sehr wichtig:

- `character_map.json` und `voice_map.json` sind aktuell globale Dateien
- `--fresh` setzt daher die aktuellen Maps komplett zurueck und nicht nur einen einzelnen Character
- wenn du ein Cluster spaeter auf `noface` setzt, wird es bei kuenftigen `05`-Laeufen nicht mehr als sichtbare Figur verwendet

### 06 - Trainingsdatensatz bauen

`python 06_build_dataset.py`

Dieser Schritt:

- liest `linked_segments`
- baut pro Szene einen kompakten Datensatz mit Figuren, Transkript, Keywords und Segmentdetails
- schreibt den Datensatz nach `data/datasets/video_training`

### 07 - Serienmodell trainieren und neue Folge erzeugen

`python 07_generate_episode.py`

Dieser Schritt:

- liest alle `*_dataset.json` aus `data/datasets/video_training`
- baut daraus ein lokales Serienmodell
- erzeugt eine neue Folge als Markdown
- erzeugt eine passende Shotlist als JSON
- filtert haeufige Fuellwoerter und typische Untertitel-/Artefaktbegriffe aus der Themenauswahl heraus

Wichtige Ausgaben:

- `generation/model/series_model.json`
- `generation/story_prompts/folge_XX.md`
- `generation/shotlists/folge_XX.json`

Wichtig:

- das "Training" ist aktuell ein lokales heuristisches Modell aus Statistiken, Sprecherbeispielen, Keywords und einer Markov-Chain
- dies ist kein grosses neuronales Generationsmodell

### 08 - Face-Cluster benennen und Review pruefen

`python 08_review_unknowns.py`

Dieser Schritt kann jetzt drei Dinge:

- ohne Zusatzparameter direkt die Face-Review fuer unbenannte Cluster starten
- offene Review-Faelle aus `review_queue.json` anzeigen
- Face-Cluster manuell benennen
- bereits benannte Face-Cluster gezielt umbenennen
- Face-Cluster mit `noface` markieren, damit sie kuenftig ignoriert werden

Wichtige Beispiele:

- `python 08_review_unknowns.py --list-faces`
- `python 08_review_unknowns.py`
- `python 08_review_unknowns.py --assign-face face_001 --name "Babe Carano"`
- `python 08_review_unknowns.py --assign-face face_014 --name "Mr. Sammich"`
- `python 08_review_unknowns.py --assign-face face_023 --name "Teague/Busboy"`
- `python 08_review_unknowns.py --rename-face face_023 --rename-to "Teague"`
- `python 08_review_unknowns.py --rename-face "Mr. Sammich" --rename-to "Mr. Sammich Sr."`
- `python 08_review_unknowns.py --assign-face face_041 --ignore`
- `python 08_review_unknowns.py --review-faces --open-previews`
- `python 08_review_unknowns.py --show-queue`

Bedeutung:

- beim einfachen Start werden alle automatisch benannten Face-Cluster nacheinander gezeigt
- `--list-faces` zeigt standardmaessig nur bereits benannte Figuren
- pro Cluster wird eine Montage mit Szene links und Ausschnitt rechts erzeugt und einzeln in einer blockierenden Vorschau gezeigt
- der naechste Face-Cluster wird erst nach einer echten Benennung oder `noface` geoeffnet
- waehrend der Review werden Beispielnamen wie `Babe Carano`, `Mr. Sammich`, `Teague/Busboy` und `noface` direkt eingeblendet
- `--assign-face ... --name ...` speichert einen dauerhaften Figurennamen in `character_map.json`
- `--rename-face ... --rename-to ...` benennt einen vorhandenen Charakter per Face-ID oder aktuellem Namen um
- `--assign-face ... --ignore` setzt den Namen intern auf `noface`
- `--review-faces` geht interaktiv durch automatisch benannte Cluster
- `--show-queue` zeigt nur die offene Sprecher-/Segment-Review an
- nach einer Namensvergabe werden `voice_map.json`, alle vorhandenen `linked_segments.json` und die `review_queue.json` direkt aktualisiert

### 09 - Serienbibel aktualisieren

`python 09_build_series_bible.py`

Dieser Schritt:

- liest `series_model.json`
- schreibt eine JSON- und Markdown-Serienbibel
- listet Hauptfiguren, Themen und Referenzszenen auf

Wichtige Ausgaben:

- `series_bible/episode_summaries/auto_series_bible.json`
- `series_bible/episode_summaries/auto_series_bible.md`

### 10 - Draft-Video rendern

`python 10_render_episode.py`

Dieser Schritt:

- nimmt standardmaessig die neueste Shotlist
- kann ueber `SERIES_RENDER_EPISODE` gezielt auf eine Folge zeigen
- erzeugt Kartenbilder, TTS-Audio und MP4-Segmente
- sammelt fuer benannte Figuren automatisch Referenzaudio aus vorhandenen Sprechersegmenten
- versucht bei vorhandenem `TTS`-Paket XTTS-Voice-Cloning pro Figur
- nutzt XTTS aber nur nach expliziter Lizenzfreigabe ueber Config oder Environment-Variable
- erzeugt fuer benannte Figuren ein referenzbasiertes Talking-Head-/Lip-Sync-Preview aus den Face-Crops
- faellt fuer unbekannte Figuren oder fehlende Assets auf statische Karten + Standard-TTS zurueck
- setzt alles zu einem Draft-Video zusammen
- schreibt ausserdem ein Render-Manifest

Wenn der installierte FFmpeg-Encoder es unterstuetzt und GPU-Nutzung aktiviert ist, wird fuer das Rendern automatisch NVENC verwendet. Sonst faellt `10` auf CPU-Encoding zurueck.
Wenn XTTS lokal verfuegbar ist, wird fuer das Voice Cloning ebenfalls das bevorzugte Torch-Geraet verwendet. Die eingebaute Lip-Sync-Variante ist ein audio-reaktiver Preview-Modus auf Basis der erkannten Face-Crops und kein externes Wav2Lip-/SadTalker-Setup.

Beispiel fuer gezielten Render in PowerShell:

```powershell
$env:SERIES_RENDER_EPISODE='folge_02'
python 10_render_episode.py
Remove-Item Env:SERIES_RENDER_EPISODE
```

Beispiel fuer XTTS nach eigener Lizenzbestaetigung:

```powershell
$env:SERIES_ACCEPT_COQUI_LICENSE='1'
python 10_render_episode.py
Remove-Item Env:SERIES_ACCEPT_COQUI_LICENSE
```

Wichtige Ausgaben:

- `generation/renders/drafts/<folge>/<folge>_draft.mp4`
- `generation/renders/drafts/<folge>/<folge>_render_manifest.json`

Wichtig:

- das Ergebnis ist ein Storyboard-/Preview-Render mit optionalem Voice Cloning und referenzbasiertem Talking Head
- unbekannte oder nicht benannte Figuren bleiben bei statischen Karten
- ohne `TTS`-Paket bleibt der Voice-Cloning-Teil automatisch im Fallback
- ohne gesetzte XTTS-Lizenzfreigabe bleibt der Voice-Cloning-Teil ebenfalls bewusst im Fallback
- keine echte vollautomatische Video-Generierung auf Produktionsniveau

### 99 - Alles in einem Lauf

`python 99_process_next_episode.py`

Fuehrt diese Schritte nacheinander aus:

- `01_setup_project.py`
- `02_import_episode.py`
- `03_split_scenes.py`
- `04_diarize_and_transcribe.py`
- `05_link_faces_and_speakers.py`
- `06_build_dataset.py`
- `07_generate_episode.py`
- `09_build_series_bible.py`
- `10_render_episode.py`

### 11 - GitHub synchronisieren (optional)

`python 11_sync_to_github.py`

Dieser Schritt:

- initialisiert bei Bedarf ein lokales Git-Repository
- synchronisiert ausschliesslich Root-`*.py`-Skripte plus `README.md`
- ignoriert alle Projektordner und deren Inhalte vollstaendig
- ignoriert den lokalen Sync-Helfer selbst absichtlich, damit er nicht mit hochgeladen wird
- prueft vor dem Push, dass bei Script-Aenderungen auch `README.md` mitgeaendert wurde
- verwendet standardmaessig das Repository `https://github.com/tyskman11/ai-series-generator`
- verwendet fuer neue lokale Commits standardmaessig die E-Mail `baumscarry@gmail.com`
- legt bei vorhandenem `GITHUB_TOKEN` ein fehlendes GitHub-Repo automatisch an
- erstellt bei neuen Aenderungen einen Commit und pusht den aktuellen Stand nach GitHub

Wichtige Umgebungsvariablen:

- `GITHUB_TOKEN`: persoenliches GitHub-Token fuer Repo-Erstellung und Push ohne gespeicherte Credentials
- `GITHUB_OWNER`: optionaler Override fuer den Standard-Owner `tyskman11`
- `GITHUB_REPO`: optionaler Override fuer das Standard-Repo `ai-series-generator`
- `GITHUB_PRIVATE`: `true` oder `false` fuer neue Repositories
- `GIT_USER_NAME`: optionaler Git-Name fuer lokale Commits
- `GIT_USER_EMAIL`: optionale Git-E-Mail fuer lokale Commits, Standard `baumscarry@gmail.com`

Nuetzliche Beispiele:

- nur pruefen, was passieren wuerde: `python 11_sync_to_github.py --dry-run`
- mit gespeicherten Git-Credentials pushen: `python 11_sync_to_github.py`
- mit Token pushen: `set GITHUB_TOKEN=<token>` und danach `python 11_sync_to_github.py`

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

- Die meisten Einzel-Skripte arbeiten standardmaessig auf dem ersten verfuegbaren Szenenordner bzw. der ersten Folge.
- `05` hat zwar jetzt `--episode`, aber `04`, `06` und einige andere Schritte arbeiten weiterhin standardmaessig auf dem ersten verfuegbaren Ordner.
- `character_map.json` und `voice_map.json` sind global und noch nicht pro Folge getrennt.
- Die Sprecherzuordnung ist heuristisch, nicht diarization-grade production quality.
- Das Trainingsmodell in `07` ist regel-/datengetrieben und nicht mit einem grossen multimodalen KI-Modell vergleichbar.
- `10` rendert ein Preview-Draft, keine finale Episode.
- Das eingebaute Lip-Sync in `10` ist ein lokaler audio-reaktiver Preview-Modus, nicht dieselbe Qualitaet wie spezialisierte Deepfake-/Lip-Sync-Modelle.
- XTTS-Voice-Cloning funktioniert nur fuer Figuren mit brauchbarem Referenzaudio und installiertem `TTS`-Paket.
- XTTS braucht zusaetzlich eine von dir explizit bestaetigte Coqui-Lizenzfreigabe; ohne diese rendert `10` absichtlich weiter mit Fallback-TTS.
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
python 07_generate_episode.py
python 10_render_episode.py
```
