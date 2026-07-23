"""
Extraction de la nomenclature douanière (positions SH, désignations, taux de
droit d'importation) à partir des PDF de chapitres du tarif des douanes.
=============================================================================

Contrairement à traitement_factures.py (qui traite les bons de commande),
ce script construit UNE FOIS votre base de référence tarifaire à partir des
PDF de chapitres (Engrais.pdf, Produits chimiques organiques.pdf, etc.).

Étapes :
  1. Extraction du texte brut de chaque chapitre (pages tarifaires uniquement)
  2. Découpage en blocs par position tarifaire principale (ex: "31.01")
  3. Extraction structurée de chaque bloc via Mistral (Ollama, local)
  4. Consolidation en une base JSON unique : nomenclature_douaniere.json

Dépendances :
    pip install pymupdf requests

Utilisation :
    python extraction_nomenclature_douaniere.py dossier_des_pdf/
    (traite tous les .pdf du dossier et produit nomenclature_douaniere.json)
"""

import sys
import os
import re
import json
import glob

import fitz
import requests


OLLAMA_URL = "http://localhost:11434/api/generate"
MODELE = "qwen2.5:3b"


# ---------------------------------------------------------------------------
# 1. Extraction du texte brut par chapitre
# ---------------------------------------------------------------------------

def extraire_texte_chapitre(pdf_path):
    """Extrait le texte du PDF, en essayant de sauter les pages de notes
    légales pour ne garder que les pages du tableau tarifaire (celles qui
    contiennent 'Codification' et 'Désignation des Produits')."""
    doc = fitz.open(pdf_path)
    texte = ""
    dans_le_tableau = False
    for page in doc:
        texte_page = page.get_text()
        if "Codification" in texte_page and "Désignation" in texte_page:
            dans_le_tableau = True
        if dans_le_tableau:
            texte += texte_page
    doc.close()
    return texte


# ---------------------------------------------------------------------------
# 2. Découpage en blocs par position tarifaire principale
# ---------------------------------------------------------------------------

def decouper_en_blocs(texte, taille_max_caracteres=1500):
    """Découpe le texte à chaque position principale (ex: '31.01') pour
    obtenir des blocs de taille raisonnable pour Mistral. Si un bloc est
    trop long (rubrique avec beaucoup de sous-positions), il est
    re-découpé en morceaux plus petits, avec un découpage forcé par
    taille de caractères en dernier recours pour garantir qu'aucun bloc
    ne dépasse la limite, même si le format du chapitre diffère."""

    blocs_bruts = re.split(r'\n(?=\d{2}\.\d{2}\n)', texte)
    blocs_bruts = [b.strip() for b in blocs_bruts if b.strip()]

    blocs_niveau_2 = []
    for bloc in blocs_bruts:
        if len(bloc) <= taille_max_caracteres:
            blocs_niveau_2.append(bloc)
        else:
            sous_blocs = re.split(r'\n(?=\d{4}\.\d{2}\n)', bloc)
            blocs_niveau_2.extend(s.strip() for s in sous_blocs if s.strip())

    # Dernier recours : découpage forcé par taille, sur des frontières de
    # ligne, pour tout bloc qui resterait trop volumineux (formats de
    # chapitre différents, pas de coupure naturelle trouvée, etc.)
    blocs_finaux = []
    for bloc in blocs_niveau_2:
        if len(bloc) <= taille_max_caracteres:
            blocs_finaux.append(bloc)
            continue

        lignes = bloc.split("\n")
        morceau_courant = []
        taille_courante = 0
        for ligne in lignes:
            if taille_courante + len(ligne) > taille_max_caracteres and morceau_courant:
                blocs_finaux.append("\n".join(morceau_courant))
                morceau_courant = []
                taille_courante = 0
            morceau_courant.append(ligne)
            taille_courante += len(ligne) + 1
        if morceau_courant:
            blocs_finaux.append("\n".join(morceau_courant))

    return blocs_finaux


# ---------------------------------------------------------------------------
# 3. Extraction structurée via Mistral
# ---------------------------------------------------------------------------

PROMPT_NOMENCLATURE = """Tu es un assistant d'extraction de données douanières.
Voici un extrait du tarif des droits de douane à l'importation. Le texte
contient des positions tarifaires (codes à 4 chiffres comme "31.01"), des
sous-positions (codes à 6 chiffres comme "3101.00"), des codes complémentaires
(2 chiffres), des désignations de produits, des taux de droit d'importation
(en %) et des unités.

Extrais CHAQUE ligne tarifaire (chaque produit ou catégorie ayant un taux de
droit associé) et retourne UNIQUEMENT un JSON valide, sans texte autour, sans
balises markdown, au format exact suivant :

[
  {{
    "position": "31.01",
    "sous_position": "3101.00",
    "code": "10",
    "designation": "guano",
    "droit_importation_pct": 2.5,
    "unite": "kg"
  }}
]

Règles :
- Le champ "designation" doit reconstituer la description complète du produit
  (en combinant le contexte de la position/sous-position si la ligne est une
  simple précision comme "autres" ou "d'os").
- "droit_importation_pct" est un nombre (utilise un point, pas une virgule).
- ATTENTION : le texte contient parfois un chiffre isolé (1 ou 2 chiffres,
  par exemple "7" ou "5") juste avant un code à 4 ou 6 chiffres comme
  "8413.11". Ce chiffre isolé est un NUMÉRO DE RENVOI (note de bas de page),
  PAS un code tarifaire. Ne le mets JAMAIS dans "position", "sous_position"
  ou "code" — ignore-le complètement.
- N'INVENTE JAMAIS un code, une position ou une sous-position. Si l'information
  n'est pas explicitement présente dans le texte fourni, mets la valeur JSON
  null (pas la chaîne "null", la vraie valeur null). Il vaut mieux laisser un
  champ vide que de deviner ou de recopier un code d'une autre ligne.
- N'extrais QUE les informations réellement présentes dans le texte ci-dessous.
  Ne complète jamais avec des connaissances générales sur les douanes.
- Ignore les notes légales, en-têtes de page, numéros de page.

Texte à traiter :
{texte}
"""


def appeler_mistral(prompt, timeout=300):
    payload = {"model": MODELE, "prompt": prompt, "stream": False, "format": "json"}
    reponse = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
    reponse.raise_for_status()
    return reponse.json()["response"]


def normaliser_code_position(valeur):
    """Corrige les positions/sous-positions valides mais mal ponctuées
    (ex: '3301' -> '33.01', '330110' -> '3301.10')."""
    if not isinstance(valeur, str):
        return valeur
    v = valeur.strip()
    if re.fullmatch(r'\d{4}', v):
        return v[:2] + "." + v[2:]
    if re.fullmatch(r'\d{6}', v):
        return v[:4] + "." + v[4:]
    return v


def valider_contre_source(ligne, texte_source):
    """Rejette (met à null) tout code/position/sous-position qui n'apparaît
    pas littéralement dans le texte source du bloc — élimine les codes
    inventés par le modèle, même s'il a reçu l'instruction de ne pas le
    faire."""
    for champ in ("position", "sous_position", "code"):
        valeur = ligne.get(champ)
        if valeur is None:
            continue
        valeur = normaliser_code_position(valeur)
        ligne[champ] = valeur
        if str(valeur) not in texte_source:
            ligne[champ] = None
            ligne.setdefault("_champs_rejetes", []).append(champ)
    return ligne


def extraire_lignes_bloc(bloc_texte, position_chapitre_info=""):
    prompt = PROMPT_NOMENCLATURE.format(texte=bloc_texte)

    sortie_brute = None
    delais = [200, 400]  # 2 tentatives, plus de marge qu'en urgence
    for tentative, delai in enumerate(delais, 1):
        try:
            sortie_brute = appeler_mistral(prompt, timeout=delai)
            break
        except requests.exceptions.RequestException as e:
            if tentative < len(delais):
                print(f"\n  Timeout (tentative {tentative}/{len(delais)}), nouvel essai avec délai plus long...", end=" ")
            else:
                print(f"\n  Erreur réseau/timeout sur ce bloc ({position_chapitre_info}) : {e}")
                print("  Bloc ignoré après plusieurs tentatives, poursuite du traitement.")
                return []

    match = re.search(r"\[.*\]", sortie_brute, re.DOTALL)
    contenu_json = match.group(0) if match else sortie_brute

    try:
        donnees = json.loads(contenu_json)
    except json.JSONDecodeError:
        print(f"  Attention : bloc non parsable ({position_chapitre_info}) :")
        print("  " + sortie_brute[:200].replace("\n", " "))
        return []

    if isinstance(donnees, dict):
        donnees = [donnees]
    if not isinstance(donnees, list):
        return []

    lignes_valides = [d for d in donnees if isinstance(d, dict)]
    lignes_valides = [valider_contre_source(l, bloc_texte) for l in lignes_valides]
    return lignes_valides


# ---------------------------------------------------------------------------
# 4. Pipeline complet sur un chapitre
# ---------------------------------------------------------------------------

def traiter_chapitre(pdf_path):
    nom_fichier = os.path.basename(pdf_path)
    print(f"\n=== Chapitre : {nom_fichier} ===")

    texte = extraire_texte_chapitre(pdf_path)
    if not texte.strip():
        print("  Aucun texte tarifaire détecté, chapitre ignoré.")
        return []

    blocs = decouper_en_blocs(texte)
    print(f"  {len(blocs)} bloc(s) à traiter.")

    toutes_lignes = []
    for i, bloc in enumerate(blocs, 1):
        print(f"  Bloc {i}/{len(blocs)}...", end=" ", flush=True)
        lignes = extraire_lignes_bloc(bloc, f"{nom_fichier} bloc {i}")
        for l in lignes:
            l["fichier_source"] = nom_fichier
        toutes_lignes.extend(lignes)
        print(f"{len(lignes)} ligne(s)")

    print(f"  Total pour ce chapitre : {len(toutes_lignes)} ligne(s)")
    return toutes_lignes


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage : python extraction_nomenclature_douaniere.py dossier_des_pdf/")
        print("        ou : python extraction_nomenclature_douaniere.py fichier.pdf")
        sys.exit(1)

    cible = sys.argv[1]

    if os.path.isdir(cible):
        fichiers_pdf = sorted(glob.glob(os.path.join(cible, "*.pdf")))
    elif os.path.isfile(cible):
        fichiers_pdf = [cible]
    else:
        print(f"Chemin introuvable : {cible}")
        sys.exit(1)

    if not fichiers_pdf:
        print("Aucun fichier PDF trouvé.")
        sys.exit(1)

    nomenclature_complete = []
    nom_sortie = "nomenclature_douaniere.json"

    for pdf_path in fichiers_pdf:
        try:
            nomenclature_complete.extend(traiter_chapitre(pdf_path))
        except Exception as e:
            print(f"\nErreur inattendue sur {os.path.basename(pdf_path)} : {e}")
            print("Chapitre ignoré, poursuite avec le suivant.")

        # Sauvegarde progressive : si le script est interrompu, le travail
        # déjà fait n'est pas perdu.
        with open(nom_sortie, "w", encoding="utf-8") as f:
            json.dump(nomenclature_complete, f, ensure_ascii=False, indent=2)

    print(f"\n{len(nomenclature_complete)} ligne(s) tarifaire(s) au total.")
    print(f"Base enregistrée : {nom_sortie}")
