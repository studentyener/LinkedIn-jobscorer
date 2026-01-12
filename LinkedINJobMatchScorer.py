# -*- coding: utf-8 -*-
"""
Created on Mon Jan 12 09:54:49 2026

@author: sofiene
"""
# =========================
#Tous nos imports
# =========================

import csv
import json
import io
import os
import re
import tkinter as tk
from tkinter import filedialog, messagebox

import webbrowser
import urllib.parse

import ttkbootstrap as tb
from ttkbootstrap.constants import *

# Test de OpenAI (optionnel)
try:
    from openai import OpenAI  # pip install openai
    OPENAI_AVAILABLE = True
except Exception:
    OpenAI = None
    OPENAI_AVAILABLE = False


# =========================
# Fonctions CSV
# =========================
def sniff_delimiter(sample: str) -> str:
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=[",", ";", "\t", "|"])
        return dialect.delimiter
    except Exception:
        return ","


def read_csv_from_text(csv_text: str):
    csv_text = csv_text.strip("\ufeff").strip()
    if not csv_text:
        return [], []

    sample = csv_text[:4096]
    delim = sniff_delimiter(sample)

    f = io.StringIO(csv_text)
    reader = csv.DictReader(f, delimiter=delim)

    rows = []
    for r in reader:
        if r is None:
            continue
        clean = {}
        for k, v in r.items():
            if k is None:
                continue
            key = str(k).strip()
            val = "" if v is None else str(v).strip()
            clean[key] = val
        if any(v for v in clean.values()):
            rows.append(clean)

    headers = list(reader.fieldnames) if reader.fieldnames else (list(rows[0].keys()) if rows else [])
    return headers, rows


def read_csv_from_file(path: str):
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        text = f.read()
    return read_csv_from_text(text)


def rows_to_corpus(rows):
    parts = []
    for r in rows:
        for v in r.values():
            if isinstance(v, str) and v.strip():
                parts.append(v.strip())
    return " ".join(parts)


# =========================
# Traitement de texte
# =========================
STOPWORDS = {
    # FR
    "le", "la", "les", "un", "une", "des", "du", "de", "d", "dans", "sur", "sous", "avec", "sans", "et", "ou",
    "mais", "donc", "or", "ni", "car", "à", "au", "aux", "ce", "cet", "cette", "ces", "se", "sa", "son", "ses",
    "leur", "leurs", "mon", "ma", "mes", "ton", "ta", "tes", "nos", "vos", "pour", "par", "pas", "plus", "moins",
    "très", "tres", "être", "etre", "avoir", "faire", "afin", "ainsi", "comme", "chez", "vers", "entre", "pendant",
    "avant", "après", "apres", "tout", "tous", "toute", "toutes", "vous", "tu", "il", "elle", "ils", "elles", "on",
    "nous", "je", "j", "y", "en", "qui", "que", "quoi", "dont", "où", "ou", "si", "l", "c", "m", "t", "s",
    # EN
    "the", "a", "an", "and", "or", "but", "so", "to", "of", "in", "on", "for", "with", "without", "as", "at", "by",
    "from", "this", "that", "these", "those", "is", "are", "was", "were", "be", "been", "being", "have", "has",
    "had", "do", "does", "did", "i", "you", "he", "she", "they", "we", "it", "your", "our", "their", "my", "me",
    "us", "them", "who", "what", "which", "where", "when", "how", "not", "more", "less", "very"
}


def normalize_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9àâäçéèêëîïôöùûüÿñæœ\s\-\+/#\.]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def tokenize(s: str):
    s = normalize_text(s)
    raw_tokens = re.split(r"[ \t\n\r]+", s)
    tokens = []
    for t in raw_tokens:
        t = t.strip("-+/#.")
        if not t:
            continue
        if len(t) <= 2:
            continue
        if t in STOPWORDS:
            continue
        tokens.append(t)
    return tokens


def extract_linkedin_keywords_from_rows(rows):
    corpus = rows_to_corpus(rows)
    if not corpus.strip():
        return []
    return sorted(set(tokenize(corpus)))


def extract_job_keywords(job_text: str):
    return sorted(set(tokenize(job_text)))


def compute_match_score(linkedin_keywords, job_keywords):
    lk = set(linkedin_keywords)
    jk = set(job_keywords)

    if not lk:
        return 0.0, [], [], [], []

    inter = sorted(lk.intersection(jk))
    missing = sorted(lk.difference(jk))

    base = (len(inter) / len(lk)) * 100.0
    length_bonus = min(5.0, max(0.0, (len(jk) - 80) / 40.0))
    score = min(100.0, base + length_bonus)

    important_missing = [w for w in missing if len(w) >= 8][:30]
    important_matched = [w for w in inter if len(w) >= 8][:30]

    return score, inter, missing, important_matched, important_missing


# =========================
# Fonctions OPENAI
# =========================
SENSITIVE_COL_PATTERNS = [
    r"email", r"e-mail", r"mail",
    r"phone", r"telephone", r"tel",
    r"address", r"adresse",
    r"birth", r"birthday", r"naissance",
]


def sanitize_rows(rows):
    if not isinstance(rows, list):
        return []
    safe = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        clean = {}
        for k, v in r.items():
            key = str(k)
            low = key.lower()
            if any(re.search(pat, low) for pat in SENSITIVE_COL_PATTERNS):
                continue
            clean[key] = v
        safe.append(clean)
    return safe


def truncate_text(s: str, max_chars: int = 8000) -> str:
    s = s.strip()
    if len(s) <= max_chars:
        return s
    return s[:max_chars] + "\n\n...[TRONQUÉ]..."


def ai_job_match_from_csv(rows, job_text: str, model: str = "gpt-4.1-mini") -> dict:
    if not OPENAI_AVAILABLE:
        raise RuntimeError("openai n'est pas installé. Fais: pip install openai")

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY manquant (variable d'environnement).")

    client = OpenAI(api_key=api_key)

    rows_safe = sanitize_rows(rows)
    job_text = truncate_text(job_text, 8000)

    system = (
        "Tu es un expert recrutement/ATS et coach carrière. "
        "Tu analyses un export LinkedIn (CSV sous forme de lignes/colonnes) et une annonce (texte). "
        "Tu dois produire une analyse structurée et actionnable. "
        "Tu retournes STRICTEMENT un JSON valide et complet."
    )

    payload = {
        "task": "job_match_scorer",
        "language": "fr",
        "linkedin_csv_rows": rows_safe[:100],
        "note": "Si le CSV contient plusieurs types d'infos, déduis les compétences/expériences au mieux.",
        "job_posting_text": job_text,
        "expected_json_schema": {
            "summary": "string (5-8 lignes max)",
            "score": "number 0-100",
            "strengths": ["string"],
            "gaps": ["string"],
            "advice": ["string"],
            "keywords_to_add": ["string"]
        },
        "rules": [
            "Sois concret, actionnable et orienté recrutement.",
            "Obligatoire: strengths >= 5 items, gaps >= 5 items, advice >= 8 items, keywords_to_add >= 10 items si possible.",
            "Chaque item doit être court (1 phrase max) et spécifique (pas de blabla).",
            "Si une info manque côté profil, dis-le dans 'gaps' et propose une action dans 'advice'.",
            "Ne divulgue aucune donnée personnelle."
        ]
    }

    resp = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}
        ],
        text={"format": {"type": "json_object"}},
    )
    return json.loads(resp.output_text)


# =========================
# UI Application 
# =========================
class App(tb.Window):
    def __init__(self):
        super().__init__(themename="darkly")  # ✅ toujours dark
        self.title("LinkedIn Toolkit — CSV")
        self.geometry("1120x800")
        self.minsize(1120, 800)

        self.csv_headers = []
        self.csv_rows = []
        self.saved_path = None
        self.last_ai_result = None  # ✅ stocke la dernière analyse IA

        self.root_container = tb.Frame(self, padding=12)
        self.root_container.pack(fill=BOTH, expand=YES)

        self.sidebar = tb.Frame(self.root_container, width=250)
        self.sidebar.pack(side=LEFT, fill=Y, padx=(0, 12))

        self.content = tb.Frame(self.root_container)
        self.content.pack(side=LEFT, fill=BOTH, expand=YES)

        self._build_sidebar()
        self.show_import_page()

    # ---------- Layout helpers ----------
    def _clear_content(self):
        for w in self.content.winfo_children():
            w.destroy()

    def _require_csv(self) -> bool:
        if not self.csv_rows:
            messagebox.showerror("CSV manquant", "Charge d'abord ton CSV LinkedIn.")
            self.show_import_page()
            return False
        return True

    def _page_header(self, title: str, subtitle: str = ""):
        header = tb.Frame(self.content)
        header.pack(fill=X)
        tb.Label(header, text=title, font=("Segoe UI", 18, "bold")).pack(anchor=W)
        if subtitle:
            tb.Label(header, text=subtitle, font=("Segoe UI", 11)).pack(anchor=W, pady=(4, 10))
        tb.Separator(self.content).pack(fill=X, pady=(10, 12))

    # ---------- Sidebar ----------
    def _build_sidebar(self):
        tb.Label(self.sidebar, text="JobMatch", font=("Segoe UI", 20, "bold")).pack(anchor=W, pady=(0, 10))
        tb.Label(self.sidebar, text="LinkedIn CSV toolkit", font=("Segoe UI", 10)).pack(anchor=W, pady=(0, 14))

        tb.Button(self.sidebar, text="Importer CSV", bootstyle="primary",
                  command=self.show_import_page).pack(fill=X, pady=6)

        tb.Button(self.sidebar, text="Analyse des données", bootstyle="warning",
                  command=self.show_data_analysis_page).pack(fill=X, pady=6)

        tb.Button(self.sidebar, text="Job Match Scorer", bootstyle="success",
                  command=self.show_jobmatch_page).pack(fill=X, pady=6)

        tb.Button(self.sidebar, text="Trouver une offre", bootstyle="info",
                  command=self.show_find_offer_page).pack(fill=X, pady=6)

        tb.Separator(self.sidebar).pack(fill=X, pady=14)

        tb.Label(self.sidebar, text="Statut", font=("Segoe UI", 11, "bold")).pack(anchor=W)
        self.status_var = tk.StringVar(value="Aucun CSV chargé.")
        tb.Label(self.sidebar, textvariable=self.status_var, wraplength=220).pack(anchor=W, pady=(6, 0))

        tb.Separator(self.sidebar).pack(fill=X, pady=14)
        hint = "IA: OK" if OPENAI_AVAILABLE else "IA: openai non installé"
        tb.Label(self.sidebar, text=hint, font=("Segoe UI", 9)).pack(anchor=W)

    # =========================
    # Page 1: Import des données CSV
    # =========================
    def show_import_page(self):
        self._clear_content()
        self._page_header(
            "Importer tes données LinkedIn (CSV)",
            "Charge un fichier .csv (recommandé) ou colle ton CSV puis valide (stockage en mémoire)."
        )

        box = tb.Labelframe(self.content, text="CSV LinkedIn (texte)", padding=10)
        box.pack(fill=BOTH, expand=YES)

        self.csv_text = tk.Text(box, height=18, wrap="word")
        self.csv_text.pack(fill=BOTH, expand=YES)

        btns = tb.Frame(self.content)
        btns.pack(fill=X, pady=12)

        tb.Button(btns, text="Valider (en mémoire)", bootstyle="success",
                  command=self.validate_and_keep_csv).pack(side=LEFT, padx=(0, 8))

        tb.Button(btns, text="Charger un CSV…", bootstyle="secondary",
                  command=self.load_csv_file).pack(side=LEFT, padx=(0, 8))

        tb.Button(btns, text="Vider", bootstyle="danger",
                  command=lambda: self.csv_text.delete("1.0", "end")).pack(side=LEFT)

    def validate_and_keep_csv(self):
        raw = self.csv_text.get("1.0", "end").strip()
        if not raw:
            messagebox.showwarning("CSV vide", "Colle d'abord ton CSV LinkedIn (ou charge un fichier).")
            return

        headers, rows = read_csv_from_text(raw)
        if not rows or not headers:
            messagebox.showerror("CSV invalide", "Impossible de lire ton CSV (pas d'en-têtes ou pas de lignes).")
            return

        self.csv_headers = headers
        self.csv_rows = rows
        self.saved_path = None

        self.status_var.set(
            "CSV validé ✅ (en mémoire)\n"
            f"Lignes : {len(rows)}\n"
            f"Colonnes : {len(headers)}"
        )
        messagebox.showinfo("OK", "CSV validé ✅ (aucun fichier créé)")

    def load_csv_file(self):
        path = filedialog.askopenfilename(
            title="Charger un fichier CSV LinkedIn",
            filetypes=[("Fichier CSV", "*.csv")],
        )
        if not path:
            return

        try:
            headers, rows = read_csv_from_file(path)
        except Exception as e:
            messagebox.showerror("Erreur", f"Impossible de lire ce CSV.\n\nDétail : {e}")
            return

        if not rows or not headers:
            messagebox.showerror("CSV invalide", "Ce fichier CSV n'a pas d'en-têtes ou pas de lignes.")
            return

        self.csv_headers = headers
        self.csv_rows = rows
        self.saved_path = path
        self.status_var.set(f"CSV chargé ✅\n{path}\nLignes: {len(rows)}")

        preview_lines = []
        delim = ";"
        preview_lines.append(delim.join(headers))
        for r in rows[:200]:
            preview_lines.append(delim.join((r.get(h, "") or "") for h in headers))
        if len(rows) > 200:
            preview_lines.append("... (aperçu tronqué) ...")

        self.csv_text.delete("1.0", "end")
        self.csv_text.insert("1.0", "\n".join(preview_lines))

        messagebox.showinfo("OK", "CSV chargé ✅")

    # =========================
    # Partie 2: Analyse des données
    # =========================
    def show_data_analysis_page(self):
        if not self._require_csv():
            return

        self._clear_content()
        self._page_header(
            "Analyse des données (CSV)",
            "Aperçu + mots-clés détectés à partir de toutes les colonnes."
        )

        headers = self.csv_headers
        rows = self.csv_rows

        out = tb.Labelframe(self.content, text="Détails", padding=10)
        out.pack(fill=BOTH, expand=YES)

        self.analysis_text = tk.Text(out, height=18, wrap="word")
        self.analysis_text.pack(fill=BOTH, expand=YES)

        def find_col(*cands):
            for cand in cands:
                for h in headers:
                    if cand in h.lower():
                        return h
            return None

        col_name = find_col("name", "nom")
        col_headline = find_col("headline", "titre")
        col_location = find_col("location", "localisation", "ville")
        col_skill = find_col("skill", "competence", "skills")

        lines = []
        lines.append("📌 Aperçu colonnes")
        lines.append(", ".join(headers[:40]) + (" ..." if len(headers) > 40 else ""))
        lines.append("")

        lines.append("👤 Profil (déduit de la 1ère ligne)")
        first = rows[0] if rows else {}
        if col_name:
            lines.append(f"- Nom : {first.get(col_name, '(non fourni)')}")
        if col_headline:
            lines.append(f"- Headline : {first.get(col_headline, '(non fourni)')}")
        if col_location:
            lines.append(f"- Localisation : {first.get(col_location, '(non fourni)')}")
        if not (col_name or col_headline or col_location):
            lines.append("- (Aucune colonne 'name/headline/location' détectée, c’est OK.)")

        lines.append("")
        lines.append("🧠 Skills (top)")
        skills = []
        if col_skill:
            for r in rows:
                v = (r.get(col_skill, "") or "").strip()
                if v:
                    skills.append(v)
        else:
            skill_cols = [h for h in headers if ("skill" in h.lower() or "compet" in h.lower())]
            for r in rows:
                for h in skill_cols:
                    v = (r.get(h, "") or "").strip()
                    if v:
                        skills.append(v)

        seen = set()
        uniq_skills = []
        for s in skills:
            k = s.lower()
            if k not in seen:
                seen.add(k)
                uniq_skills.append(s)

        lines.append(", ".join(uniq_skills[:40]) if uniq_skills else "(aucun skill détecté)")
        if len(uniq_skills) > 40:
            lines.append(f"... (+{len(uniq_skills) - 40} autres)")

        lines.append("")
        lines.append("🔎 Keywords détectés (matching)")
        kw = extract_linkedin_keywords_from_rows(rows)
        lines.append(f"- Total : {len(kw)}")
        lines.append(", ".join(kw[:70]) + (" ..." if len(kw) > 70 else ""))

        lines.append("")
        lines.append("🧾 Exemples de lignes (3)")
        for i, r in enumerate(rows[:3], start=1):
            sample_pairs = []
            for h in headers[:10]:
                v = (r.get(h, "") or "").strip()
                if v:
                    sample_pairs.append(f"{h}={v}")
            lines.append(f"{i}) " + (" | ".join(sample_pairs) if sample_pairs else "(ligne vide)"))

        self.analysis_text.delete("1.0", "end")
        self.analysis_text.insert("1.0", "\n".join(lines))

        actions = tb.Frame(self.content)
        actions.pack(fill=X, pady=10)
        tb.Button(actions, text="Copier", bootstyle="info",
                  command=self.copy_analysis_to_clipboard).pack(side=LEFT, padx=(0, 8))
        tb.Button(actions, text="Exporter résumé (JSON)", bootstyle="secondary",
                  command=self.export_analysis_summary).pack(side=LEFT)

    def copy_analysis_to_clipboard(self):
        txt = self.analysis_text.get("1.0", "end").strip()
        self.clipboard_clear()
        self.clipboard_append(txt)
        messagebox.showinfo("Copié", "Résumé copié ✅")

    def export_analysis_summary(self):
        txt = self.analysis_text.get("1.0", "end").strip()
        payload = {
            "analysis_text": txt,
            "source_csv_path": self.saved_path or "(en mémoire - non enregistré)",
            "headers": self.csv_headers,
            "rows_count": len(self.csv_rows),
        }
        path = filedialog.asksaveasfilename(
            title="Exporter le résumé",
            defaultextension=".json",
            filetypes=[("Fichier JSON", "*.json")],
            initialfile="linkedin_csv_analysis_summary.json",
        )
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            messagebox.showinfo("OK", f"Exporté ✅\n{path}")
        except Exception as e:
            messagebox.showerror("Erreur", f"Impossible d'exporter.\n\nDétail : {e}")

    # =========================
    # Partie 3: Trouver une offre (Google)
    # =========================
    def show_find_offer_page(self):
        self._clear_content()
        self._page_header(
            "Trouver une offre (Google)",
            "L’app construit ta recherche et ouvre Google / Google Jobs dans ton navigateur."
        )

        form = tb.Labelframe(self.content, text="Critères", padding=10)
        form.pack(fill=X)

        self.keyword_var = tk.StringVar(value="Data Analyst")
        self.location_var = tk.StringVar(value="Strasbourg")
        self.contract_var = tk.StringVar(value="CDI")
        self.remote_var = tk.StringVar(value="Hybride")

        tb.Label(form, text="Mots-clés").grid(row=0, column=0, sticky=W, pady=6)
        tb.Entry(form, textvariable=self.keyword_var, width=38).grid(row=0, column=1, sticky=W, pady=6, padx=8)

        tb.Label(form, text="Localisation").grid(row=0, column=2, sticky=W, pady=6, padx=(20, 0))
        tb.Entry(form, textvariable=self.location_var, width=26).grid(row=0, column=3, sticky=W, pady=6, padx=8)

        tb.Label(form, text="Contrat").grid(row=1, column=0, sticky=W, pady=6)
        tb.Combobox(
            form,
            textvariable=self.contract_var,
            values=["Stage", "Alternance", "CDD", "CDI", "Freelance"],
            state="readonly",
            width=35
        ).grid(row=1, column=1, sticky=W, pady=6, padx=8)

        tb.Label(form, text="Télétravail").grid(row=1, column=2, sticky=W, pady=6, padx=(20, 0))
        tb.Combobox(
            form,
            textvariable=self.remote_var,
            values=["Sur site", "Hybride", "Full remote"],
            state="readonly",
            width=23
        ).grid(row=1, column=3, sticky=W, pady=6, padx=8)

        actions = tb.Frame(self.content)
        actions.pack(fill=X, pady=12)

        tb.Button(actions, text="Afficher le résumé", bootstyle="secondary",
                  command=self.preview_find_offer).pack(side=LEFT, padx=(0, 8))

        tb.Button(actions, text="Rechercher sur Google", bootstyle="success",
                  command=self.search_jobs_on_google).pack(side=LEFT, padx=(0, 8))

        tb.Button(actions, text="Google (Jobs)", bootstyle="info",
                  command=self.search_google_jobs_tab).pack(side=LEFT)

        out = tb.Labelframe(self.content, text="Résumé", padding=10)
        out.pack(fill=BOTH, expand=YES)

        self.find_result = tk.Text(out, height=12, wrap="word")
        self.find_result.pack(fill=BOTH, expand=YES)
        self.find_result.insert("1.0", "Clique sur “Rechercher sur Google” pour ouvrir la recherche.")

    def preview_find_offer(self):
        summary = (
            "✅ Paramètres\n\n"
            f"- Mots-clés : {self.keyword_var.get().strip()}\n"
            f"- Localisation : {self.location_var.get().strip()}\n"
            f"- Contrat : {self.contract_var.get().strip()}\n"
            f"- Télétravail : {self.remote_var.get().strip()}\n\n"
            "Astuce : clique sur “Rechercher sur Google” pour lancer la recherche."
        )
        self.find_result.delete("1.0", "end")
        self.find_result.insert("1.0", summary)

    def _build_google_query(self) -> str:
        keywords = self.keyword_var.get().strip()
        location = self.location_var.get().strip()
        contract = self.contract_var.get().strip()
        remote = self.remote_var.get().strip()

        q_parts = []
        if keywords:
            q_parts.append(keywords)

        q_parts.append("offre d'emploi OR recrutement OR job")
        if contract:
            q_parts.append(contract)

        if remote == "Full remote":
            q_parts.append('"télétravail" OR remote OR "full remote"')
        elif remote == "Hybride":
            q_parts.append('"hybride" OR "télétravail partiel" OR remote')
        else:
            q_parts.append('"sur site" OR présentiel')

        if location:
            q_parts.append(location)

        return " ".join(q_parts)

    def search_jobs_on_google(self):
        q = self._build_google_query()
        url = "https://www.google.com/search?q=" + urllib.parse.quote_plus(q)
        webbrowser.open(url)

        self.find_result.delete("1.0", "end")
        self.find_result.insert("1.0", f"🔎 Recherche Google ouverte ✅\n\nRequête :\n{q}\n\nURL :\n{url}")

    def search_google_jobs_tab(self):
        q = self._build_google_query() + " jobs"
        url = "https://www.google.com/search?q=" + urllib.parse.quote_plus(q) + "&ibp=htl;jobs"
        webbrowser.open(url)

        self.find_result.delete("1.0", "end")
        self.find_result.insert("1.0", f"🧭 Google Jobs ouvert ✅\n\nRequête :\n{q}\n\nURL :\n{url}")

    # =========================
    # Partie 4: Job match Scorer
    # =========================
    def show_jobmatch_page(self):
        if not self._require_csv():
            return

        self._clear_content()
        self._page_header(
            "Job Match Scorer",
            "Colle une annonce → score rapide (keywords) ou analyse IA. Bouton dédié pour les améliorations."
        )

        top = tb.Frame(self.content)
        top.pack(fill=X, pady=(0, 10))

        self.model_var = tk.StringVar(value="gpt-4.1-mini")
        tb.Label(top, text="Modèle IA :", font=("Segoe UI", 10, "bold")).pack(side=LEFT)
        tb.Entry(top, textvariable=self.model_var, width=20).pack(side=LEFT, padx=8)

        ia_status = "OK" if OPENAI_AVAILABLE else "openai non installé"
        tb.Label(top, text=f"IA: {ia_status}", font=("Segoe UI", 10)).pack(side=LEFT, padx=12)

        annonce = tb.Labelframe(self.content, text="Annonce (copier-coller)", padding=10)
        annonce.pack(fill=BOTH, expand=YES)

        self.job_text = tk.Text(annonce, height=12, wrap="word")
        self.job_text.pack(fill=BOTH, expand=YES)

        actions = tb.Frame(self.content)
        actions.pack(fill=X, pady=10)

        tb.Button(actions, text="Score rapide (keywords)", bootstyle="success",
                  command=self.compute_job_match_keywords).pack(side=LEFT, padx=(0, 8))
        tb.Button(actions, text="Analyse IA (ChatGPT)", bootstyle="primary",
                  command=self.compute_job_match_ai).pack(side=LEFT, padx=(0, 8))
        tb.Button(actions, text="Améliorations (IA)", bootstyle="info",
                  command=self.show_ai_improvements).pack(side=LEFT, padx=(0, 8))
        tb.Button(actions, text="Résumé annonce", bootstyle="secondary",
                  command=self.preview_job_post).pack(side=LEFT, padx=(0, 8))
        tb.Button(actions, text="Vider", bootstyle="danger",
                  command=lambda: self.job_text.delete("1.0", "end")).pack(side=LEFT)

        score_box = tb.Labelframe(self.content, text="Score", padding=10)
        score_box.pack(fill=X, pady=(6, 10))

        self.score_var = tk.DoubleVar(value=0.0)
        self.score_label_var = tk.StringVar(value="0.0%")

        tb.Label(score_box, text="Compatibilité :", font=("Segoe UI", 11, "bold")).pack(side=LEFT)
        self.score_bar = tb.Progressbar(
            score_box, maximum=100, variable=self.score_var, length=470,
            bootstyle="secondary-striped"
        )
        self.score_bar.pack(side=LEFT, padx=12)
        tb.Label(score_box, textvariable=self.score_label_var,
                 font=("Segoe UI", 12, "bold")).pack(side=LEFT)

        out = tb.Labelframe(self.content, text="Résultat", padding=10)
        out.pack(fill=BOTH, expand=YES)

        self.job_result_text = tk.Text(out, height=12, wrap="word")
        self.job_result_text.pack(fill=BOTH, expand=YES)
        self.job_result_text.insert("1.0", "Colle une annonce puis clique sur un bouton d'analyse.")

    def _set_score_bar_style(self, score: float):
        if score >= 75:
            style = "success-striped"
        elif score >= 50:
            style = "warning-striped"
        else:
            style = "danger-striped"
        self.score_bar.configure(bootstyle=style)

    def animate_score(self, target: float):
        target = max(0.0, min(100.0, float(target)))
        current = float(self.score_var.get())

        self._set_score_bar_style(target)

        if abs(target - current) < 0.6:
            self.score_var.set(target)
            self.score_label_var.set(f"{target:.1f}%")
            return

        step = 1.0 if target > current else -1.0
        new_val = current + step
        self.score_var.set(new_val)
        self.score_label_var.set(f"{new_val:.0f}%")
        self.after(8, lambda: self.animate_score(target))

    def preview_job_post(self):
        text = self.job_text.get("1.0", "end").strip()
        if not text:
            messagebox.showwarning("Annonce vide", "Colle une annonce d'abord.")
            return

        char_count = len(text)
        word_count = len(text.split())
        excerpt = text[:900] + ("..." if len(text) > 900 else "")

        summary = (
            "✅ Résumé annonce (local)\n\n"
            f"- Caractères : {char_count}\n"
            f"- Mots : {word_count}\n\n"
            "Extrait :\n"
            f"{excerpt}"
        )
        self.job_result_text.delete("1.0", "end")
        self.job_result_text.insert("1.0", summary)

    def compute_job_match_keywords(self):
        job = self.job_text.get("1.0", "end").strip()
        if not job:
            messagebox.showwarning("Annonce vide", "Colle une annonce d'abord.")
            return

        linkedin_keywords = extract_linkedin_keywords_from_rows(self.csv_rows)
        job_keywords = extract_job_keywords(job)

        score, matched, missing, imp_matched, imp_missing = compute_match_score(linkedin_keywords, job_keywords)
        self.animate_score(score)

        result = []
        result.append("⚡ Job Match (keywords - local)")
        result.append("")
        result.append(f"Score estimé : {score:.1f}%")
        result.append("")
        result.append(f"Keywords CSV détectés : {len(linkedin_keywords)}")
        result.append(f"Keywords annonce détectés : {len(job_keywords)}")
        result.append(f"Correspondances : {len(matched)}")
        result.append("")
        result.append("✅ Mots-clés trouvés (extrait) :")
        result.append(", ".join(matched[:60]) if matched else "(aucun)")
        result.append("")
        result.append("❌ Mots-clés CSV manquants (extrait) :")
        result.append(", ".join(missing[:60]) if missing else "(aucun)")
        result.append("")
        result.append("⭐ Focus (mots longs ≥ 8 caractères) :")
        result.append(f"- Matched : {', '.join(imp_matched) if imp_matched else '(aucun)'}")
        result.append(f"- Missing : {', '.join(imp_missing) if imp_missing else '(aucun)'}")

        self.job_result_text.delete("1.0", "end")
        self.job_result_text.insert("1.0", "\n".join(result))

    def compute_job_match_ai(self):
        job = self.job_text.get("1.0", "end").strip()
        if not job:
            messagebox.showwarning("Annonce vide", "Colle une annonce d'abord.")
            return

        model = (self.model_var.get().strip() or "gpt-4.1-mini")

        try:
            result = ai_job_match_from_csv(self.csv_rows, job, model=model)
            self.last_ai_result = result
        except Exception as e:
            self.last_ai_result = None
            messagebox.showwarning(
                "IA indisponible",
                "Impossible de faire l'analyse IA (quota / clé / réseau).\n"
                "Analyse locale (keywords) utilisée à la place.\n\n"
                f"Détail : {e}"
            )
            self.compute_job_match_keywords()
            return

        try:
            score = float(result.get("score", 0))
        except Exception:
            score = 0.0
        score = max(0.0, min(100.0, score))
        self.animate_score(score)

        lines = []
        lines.append("🤖 Analyse IA (ChatGPT)")
        lines.append("")
        lines.append(f"🎯 Score : {score:.1f}/100")
        lines.append("")
        lines.append("✅ Points forts")
        strengths = result.get("strengths", [])
        if isinstance(strengths, list) and strengths:
            for s in strengths[:10]:
                lines.append(f"• {s}")
        else:
            lines.append("• (aucun)")

        lines.append("")
        lines.append("❌ Manques / écarts")
        gaps = result.get("gaps", [])
        if isinstance(gaps, list) and gaps:
            for g in gaps[:12]:
                lines.append(f"• {g}")
        else:
            lines.append("• (aucun)")

        lines.append("")
        lines.append("🚀 Pistes d’amélioration")
        advice = result.get("advice", [])
        if isinstance(advice, list) and advice:
            for a in advice[:15]:
                lines.append(f"• {a}")
        else:
            lines.append("• (aucun)")

        lines.append("")
        lines.append("🔑 Mots-clés à ajouter")
        kws = result.get("keywords_to_add", [])
        if isinstance(kws, list) and kws:
            lines.append(", ".join(kws[:40]))
        else:
            lines.append("(aucun)")

        lines.append("")
        lines.append("🧾 Résumé")
        lines.append(result.get("summary", "(vide)"))

        self.job_result_text.delete("1.0", "end")
        self.job_result_text.insert("1.0", "\n".join(lines))

    def show_ai_improvements(self):
        if not self.last_ai_result:
            messagebox.showinfo(
                "Améliorations (IA)",
                "Aucune analyse IA disponible.\n\nLance d'abord “Analyse IA (ChatGPT)”."
            )
            return

        advice = self.last_ai_result.get("advice", [])
        kws = self.last_ai_result.get("keywords_to_add", [])

        lines = []
        lines.append("🚀 Améliorations proposées par l’IA")
        lines.append("")
        lines.append("📌 Actions concrètes")
        if isinstance(advice, list) and advice:
            for a in advice[:30]:
                lines.append(f"• {a}")
        else:
            lines.append("• (aucune)")

        lines.append("")
        lines.append("🔑 Mots-clés à ajouter (CV / LinkedIn)")
        if isinstance(kws, list) and kws:
            lines.append(", ".join(kws[:50]))
        else:
            lines.append("(aucun)")

        win = tb.Toplevel(self)
        win.title("Améliorations (IA)")
        win.geometry("820x520")

        box = tb.Frame(win, padding=10)
        box.pack(fill=BOTH, expand=YES)

        txt = tk.Text(box, wrap="word")
        txt.pack(fill=BOTH, expand=YES)
        txt.insert("1.0", "\n".join(lines))
        txt.configure(state="disabled")

        btns = tb.Frame(win, padding=10)
        btns.pack(fill=X)

        def copy():
            self.clipboard_clear()
            self.clipboard_append("\n".join(lines))
            messagebox.showinfo("Copié", "Améliorations copiées ✅")

        tb.Button(btns, text="Copier", bootstyle="secondary", command=copy).pack(side=LEFT)
        tb.Button(btns, text="Fermer", bootstyle="danger", command=win.destroy).pack(side=RIGHT)


if __name__ == "__main__":
    app = App()
    app.mainloop()
