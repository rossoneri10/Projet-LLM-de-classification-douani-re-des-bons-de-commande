"""
Pipeline de traitement des bons de commande / factures (PDF ou image)
=======================================================================

Étapes :
  1. Détection du type de fichier (PDF texte natif / PDF scanné / image)
  2. Extraction du texte (directe ou OCR)
  3. Extraction structurée des lignes produits via Mistral (Ollama, local)
  4. Correspondance des codes produits avec un référentiel (exacte puis floue)
  5. Rapport final (JSON) avec niveau de confiance par ligne

Dépendances à installer :
    pip install pymupdf pdf2image pytesseract pillow rapidfuzz requests opencv-python

Dépendances système :
    - Tesseract OCR installé sur la machine (avec le pack de langue "fra")
    - Poppler installé (requis par pdf2image)
    - Ollama installé et lancé, avec le modèle mistral disponible :
          ollama pull mistral

Utilisation :
    python traitement_factures.py chemin/vers/facture.pdf
    python traitement_factures.py chemin/vers/facture.jpg
"""

import sys
import os
import json
import re
import requests

import fitz  # PyMuPDF
from pdf2image import convert_from_path
import pytesseract
from PIL import Image
import cv2
import numpy as np
from rapidfuzz import process, fuzz


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OLLAMA_URL = "http://localhost:11434/api/generate"
MODELE = "qwen2.5:3b"
DPI_OCR = 300
LANGUE_OCR = "fra"

# Référentiel produits d'exemple. À remplacer par le chargement de vos
# fichiers réels (CSV, Excel, base de données...).
REFERENTIEL_PRODUITS = [
    {"code": "PRD-0012", "designation": "Vis inox M6x20"},
    {"code": "PRD-0045", "designation": "Câble électrique 2.5mm"},
    {"code": "PRD-0078", "designation": "Roulement à billes 6202"},
    {"code": "PRD-0103", "designation": "Joint torique 15mm"},
]


# ---------------------------------------------------------------------------
# 1. Détection du type de fichier
# ---------------------------------------------------------------------------

def has_native_text(pdf_path, seuil_caracteres=50):
    """Retourne True si le PDF contient du texte natif exploitable."""
    doc = fitz.open(pdf_path)
    texte = ""
    for page in doc:
        texte += page.get_text()
    doc.close()
    return len(texte.strip()) > seuil_caracteres


def est_image(fichier_path):
    return fichier_path.lower().endswith((".jpg", ".jpeg", ".png", ".tiff", ".bmp"))


# ---------------------------------------------------------------------------
# 2. Extraction du texte
# ---------------------------------------------------------------------------

def extraction_pdf_native(pdf_path):
    doc = fitz.open(pdf_path)
    texte = "\n".join(page.get_text() for page in doc)
    doc.close()
    return texte


def pretraiter_image(image_pil):
    """Améliore la qualité d'une image avant OCR : niveaux de gris,
    contraste et redressement (deskew)."""
    image_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    gris = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)

    # Contraste adaptatif
    gris = cv2.adaptiveThreshold(
        gris, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 11
    )

    # Redressement (deskew) basé sur les contours du texte
    coords = np.column_stack(np.where(gris < 255))
    if len(coords) > 0:
        angle = cv2.minAreaRect(coords)[-1]
        angle = -(90 + angle) if angle < -45 else -angle
        (h, w) = gris.shape
        centre = (w // 2, h // 2)
        matrice = cv2.getRotationMatrix2D(centre, angle, 1.0)
        gris = cv2.warpAffine(
            gris, matrice, (w, h), flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )

    return Image.fromarray(gris)


def ocr_image(image_pil):
    image_traitee = pretraiter_image(image_pil)
    return pytesseract.image_to_string(image_traitee, lang=LANGUE_OCR)


def pdf_vers_images(pdf_path):
    return convert_from_path(pdf_path, dpi=DPI_OCR)


def extraire_texte(fichier_path):
    """Point d'entrée unique : renvoie le texte brut quel que soit le
    format d'entrée (PDF texte, PDF scanné, image)."""
    if est_image(fichier_path):
        image = Image.open(fichier_path)
        return ocr_image(image)

    if has_native_text(fichier_path):
        return extraction_pdf_native(fichier_path)

    # PDF scanné (image encapsulée) : on OCRise chaque page
    images = pdf_vers_images(fichier_path)
    return "\n".join(ocr_image(img) for img in images)


# ---------------------------------------------------------------------------
# 3. Extraction structurée via Mistral (Ollama)
# ---------------------------------------------------------------------------

PROMPT_EXTRACTION = """Tu es un assistant d'extraction de données. Voici le texte
brut d'un bon de commande ou d'une facture, potentiellement issu d'un OCR
(donc avec possibles fautes de reconnaissance).

Extrais chaque ligne produit et retourne UNIQUEMENT un JSON valide, sans texte
autour, sans balises markdown, au format exact suivant :

[{{"code_produit": "...", "designation": "...", "quantite": ..., "prix_unitaire": ...}}]

Si une information est absente ou illisible, mets la valeur null.

Texte du document :
{texte}
"""


def appeler_mistral(prompt, format_json=True):
    payload = {
        "model": MODELE,
        "prompt": prompt,
        "stream": False,
    }
    if format_json:
        payload["format"] = "json"

    reponse = requests.post(OLLAMA_URL, json=payload, timeout=300)
    reponse.raise_for_status()
    return reponse.json()["response"]


def extraire_lignes_produits(texte_facture):
    prompt = PROMPT_EXTRACTION.format(texte=texte_facture)
    sortie_brute = appeler_mistral(prompt)

    # Filet de sécurité : au cas où le modèle ajoute du texte autour du JSON
    match = re.search(r"\[.*\]", sortie_brute, re.DOTALL)
    contenu_json = match.group(0) if match else sortie_brute

    try:
        donnees = json.loads(contenu_json)
    except json.JSONDecodeError:
        print("Attention : réponse du modèle non parsable en JSON :")
        print(sortie_brute)
        return []

    print("--- JSON brut renvoyé par Mistral (debug) ---")
    print(json.dumps(donnees, ensure_ascii=False, indent=2))
    print("---------------------------------------------")

    return normaliser_lignes(donnees)


def normaliser_lignes(donnees):
    """Ramène la sortie du modèle à une liste de dictionnaires, quelle que
    soit sa forme exacte (liste directe, dict englobant, chaînes JSON
    imbriquées, etc.)."""

    # Cas 1 : le modèle a englobé la liste dans une clé, ex. {"lignes": [...]}
    if isinstance(donnees, dict):
        for cle in ("lignes", "line_items", "items", "produits", "data"):
            if cle in donnees and isinstance(donnees[cle], list):
                donnees = donnees[cle]
                break
        else:
            # Dict unique représentant une seule ligne
            donnees = [donnees]

    if not isinstance(donnees, list):
        print("Attention : format de sortie inattendu, aucune ligne exploitable.")
        return []

    resultat = []
    for item in donnees:
        if isinstance(item, dict):
            resultat.append(item)
        elif isinstance(item, str):
            # Le modèle a parfois renvoyé chaque ligne comme une chaîne
            # JSON au lieu d'un objet JSON réel : on retente de la parser.
            try:
                sous_item = json.loads(item)
                if isinstance(sous_item, dict):
                    resultat.append(sous_item)
                    continue
            except json.JSONDecodeError:
                pass
            print(f"Ligne ignorée (format inattendu) : {item!r}")
        else:
            print(f"Ligne ignorée (type inattendu) : {item!r}")

    return resultat


# ---------------------------------------------------------------------------
# 4. Correspondance avec le référentiel produits
# ---------------------------------------------------------------------------

SEUIL_CONFIANCE_FLOUE = 80  # score rapidfuzz sur 100


def correspondance_exacte(code_extrait):
    for produit in REFERENTIEL_PRODUITS:
        if produit["code"].strip().upper() == str(code_extrait).strip().upper():
            return produit
    return None


def correspondance_floue(code_extrait, designation_extraite):
    codes_reference = [p["code"] for p in REFERENTIEL_PRODUITS]
    designations_reference = [p["designation"] for p in REFERENTIEL_PRODUITS]

    # On tente d'abord sur le code, puis sur la désignation
    meilleur_code = process.extractOne(
        str(code_extrait), codes_reference, scorer=fuzz.ratio
    )
    meilleure_designation = process.extractOne(
        str(designation_extraite), designations_reference, scorer=fuzz.token_sort_ratio
    )

    candidats = []
    if meilleur_code and meilleur_code[1] >= SEUIL_CONFIANCE_FLOUE:
        produit = next(p for p in REFERENTIEL_PRODUITS if p["code"] == meilleur_code[0])
        candidats.append((produit, meilleur_code[1]))
    if meilleure_designation and meilleure_designation[1] >= SEUIL_CONFIANCE_FLOUE:
        produit = next(
            p for p in REFERENTIEL_PRODUITS if p["designation"] == meilleure_designation[0]
        )
        candidats.append((produit, meilleure_designation[1]))

    if not candidats:
        return None

    candidats.sort(key=lambda c: c[1], reverse=True)
    return candidats[0]


PROMPT_DESAMBIGUISATION = """Tu dois déterminer quel produit du référentiel
correspond le mieux à une ligne extraite d'un bon de commande.

Ligne extraite du bon de commande :
  code : {code}
  désignation : {designation}

Candidats possibles dans le référentiel :
{candidats}

Réponds UNIQUEMENT avec un JSON au format :
{{"code_choisi": "...", "raison": "..."}}
Si aucun candidat ne correspond de façon plausible, mets code_choisi à null.
"""


def desambiguiser_avec_llm(code_extrait, designation_extraite, candidats):
    texte_candidats = "\n".join(
        f"- {c['code']} : {c['designation']}" for c in candidats
    )
    prompt = PROMPT_DESAMBIGUISATION.format(
        code=code_extrait, designation=designation_extraite, candidats=texte_candidats
    )
    sortie = appeler_mistral(prompt)
    try:
        return json.loads(sortie)
    except json.JSONDecodeError:
        return {"code_choisi": None, "raison": "Réponse LLM non parsable"}


def resoudre_ligne(ligne_extraite):
    code = ligne_extraite.get("code_produit")
    designation = ligne_extraite.get("designation")

    # 1. Correspondance exacte
    produit = correspondance_exacte(code) if code else None
    if produit:
        return {**ligne_extraite, "produit_reference": produit, "confiance": "exacte"}

    # 2. Correspondance floue
    resultat_floue = correspondance_floue(code, designation)
    if resultat_floue and resultat_floue[1] >= 92:
        # score très élevé : on considère la correspondance fiable
        return {
            **ligne_extraite,
            "produit_reference": resultat_floue[0],
            "confiance": f"floue ({resultat_floue[1]}%)",
        }

    # 3. Cas ambigu : on sollicite le LLM avec les meilleurs candidats
    candidats = [p for p in REFERENTIEL_PRODUITS]  # à affiner : top N par score
    decision = desambiguiser_avec_llm(code, designation, candidats)
    produit_choisi = next(
        (p for p in REFERENTIEL_PRODUITS if p["code"] == decision.get("code_choisi")),
        None,
    )
    return {
        **ligne_extraite,
        "produit_reference": produit_choisi,
        "confiance": "llm" if produit_choisi else "non_resolu",
        "raison_llm": decision.get("raison"),
    }


# ---------------------------------------------------------------------------
# 5. Pipeline complet
# ---------------------------------------------------------------------------

def traiter_facture(fichier_path):
    print(f"Traitement de : {fichier_path}")

    texte = extraire_texte(fichier_path)
    print("Texte extrait (aperçu) :")
    print(texte[:300], "...\n")

    lignes = extraire_lignes_produits(texte)
    print(f"{len(lignes)} ligne(s) produit détectée(s).\n")

    resultats = [resoudre_ligne(ligne) for ligne in lignes]

    rapport = {
        "fichier_source": os.path.basename(fichier_path),
        "nb_lignes": len(resultats),
        "lignes": resultats,
    }
    return rapport


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage : python traitement_factures.py chemin/vers/fichier")
        sys.exit(1)

    fichier = sys.argv[1]
    if not os.path.exists(fichier):
        print(f"Fichier introuvable : {fichier}")
        sys.exit(1)

    rapport_final = traiter_facture(fichier)

    nom_sortie = os.path.splitext(os.path.basename(fichier))[0] + "_rapport.json"
    with open(nom_sortie, "w", encoding="utf-8") as f:
        json.dump(rapport_final, f, ensure_ascii=False, indent=2)

    print(f"Rapport enregistré : {nom_sortie}")
