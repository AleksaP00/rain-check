# RainCheck: beleške za Claude

## Ko je korisnik i kako radimo

- Korisnik je **početnik**: završio je CS50P (osnove Pythona) i ovaj projekat radi da vežba naučeno.
- Razgovor se vodi na **srpskom** (latinica). README projekta je na engleskom.
- **Korisnik sam kuca sav kod.** Ne piši gotova rešenja. Objasni šta funkcija treba da radi, daj nagoveštaje, pa tek ako zapne, rešenje.
- Objašnjavaj **svaki korak i svaku komandu** (šta je alat, šta znači svaka opcija, zašto se tako radi).
- Radimo malim koracima, jedna funkcija po jedna. Posle svake funkcije: test, pa commit.
- Korisnik na ovom računaru koristi `python` (ne `python3`). Uvek u aktiviranom `.venv`.
- Komande za git i terminal korisnik kuca sam; ti mu ih daješ i objašnjavaš.
- Kad ispravljaš grešku, objasni zašto je greška nastala. Ne ispravljaj tiho.

## Projekat

Specifikacija je u dokumentu "RainCheck — CS50P finalni projekat za AI/ML put" (PDF, nije u repou).

Ukratko: CLI aplikacija (`project.py`) sa komandama `fetch`, `train`, `evaluate`, `predict`. Preuzima istoriju vremena iz Open-Meteo API-ja, trenira k-NN model napisan od nule (`model.py`: `StandardScaler`, `KNNClassifier`) i predviđa da li će sutra padati kiša.

Fajlovi: `project.py`, `model.py`, `test_project.py`, `test_model.py`, `requirements.txt`, `README.md`.

Pravila iz dokumenta:
- Svaka funkcija ima type hints i docstring. Svaki modul ima `logger = logging.getLogger(__name__)`.
- Testovi ne idu na internet (lažni API preko `monkeypatch`). Cilj je 12–15 testova.
- `data/` i `models/` nisu u git-u.
- `print` samo za rezultat koji korisnik traži, sve ostalo kroz `logging`.
- Očekivane greške daju jasnu poruku i `sys.exit(1)`, bez traceback-a.

## Git tok rada

- `main` je uvek ispravan. Svaka faza je svoja grana i svoj Pull Request.
- Koraci: `git switch main` → `git pull` → `git switch -c <grana>` → mali commiti (`git add -p` za postojeće fajlove) → `git push -u origin <grana>` → PR na GitHub-u → merge → `git switch main` → `git pull` → `git branch -d <grana>`.
- Poruke commita: Conventional Commits (`feat:`, `fix:`, `test:`, `docs:`, `chore:`), kratko, imperativ.
- Na kraju: `git tag v1.0.0` i `git push --tags`.

## Plan i stanje

Faze: 1) setup, 2) podaci (`feat/fetch-data`), 3) dataset (`feat/dataset`), 4) model (`feat/knn-model`), 5) CLI (`feat/cli`), 6) README (`docs/readme`).

- [x] Faza 1: setup (spojena u `main`, PR #1)
- [ ] Faza 2: podaci, grana `feat/fetch-data`
  - [x] `validate_date` + test
  - [ ] `slugify` + test (u radu)
  - [ ] `geocode` (+ `LocationNotFoundError`) + test
  - [ ] `fetch_history` + test
  - [ ] `save_csv` / `load_csv` + test
- [ ] Faze 3–6

Ažuriraj ovu listu kad se nešto završi.
