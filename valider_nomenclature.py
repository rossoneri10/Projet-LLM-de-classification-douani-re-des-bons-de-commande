"""
Validation de la base de nomenclature douanière extraite
============================================================

Vérifie la qualité d'un fichier nomenclature_douaniere.json produit par
extraction_nomenclature_douaniere.py : normalise les "null" mal typés,
détecte les positions au format invalide, et repère les codes suspects
(répétés sur des désignations trop différentes = probable hallucination).

Utilisation :
    python valider_nomenclature.py nomenclature_douaniere.json
"""

import sys
import json
import re
from collections import defaultdict


REGEX_POSITION_VALIDE = re.compile(r'^\d{2}\.\d{2}$')
REGEX_SOUS_POSITION_VALIDE = re.compile(r'^\d{4}\.\d{2}$')


def normaliser_valeur(v):
    """Convertit les chaînes 'null' / 'None' / '' en vraie valeur null."""
    if isinstance(v, str) and v.strip().lower() in ("null", "none", ""):
        return None
    return v


def valider(fichier_json):
    with open(fichier_json, "r", encoding="utf-8") as f:
        lignes = json.load(f)

    print(f"Total de lignes : {len(lignes)}\n")

    anomalies_position_invalide = []
    anomalies_code_suspect = defaultdict(list)
    lignes_normalisees = []

    for i, ligne in enumerate(lignes):
        ligne_normalisee = {k: normaliser_valeur(v) for k, v in ligne.items()}
        lignes_normalisees.append(ligne_normalisee)

        position = ligne_normalisee.get("position")
        if position and not (
            REGEX_POSITION_VALIDE.match(position) or REGEX_SOUS_POSITION_VALIDE.match(position)
        ):
            anomalies_position_invalide.append((i, position, ligne_normalisee.get("designation")))

        # Un même couple (position, code) censé désigner UN produit :
        # si les désignations associées sont très différentes, suspect
        cle = (position, ligne_normalisee.get("code"))
        anomalies_code_suspect[cle].append(ligne_normalisee.get("designation"))

    print(f"--- Positions au format invalide ({len(anomalies_position_invalide)}) ---")
    for i, position, designation in anomalies_position_invalide[:20]:
        print(f"  ligne {i} : position='{position}' -> {designation}")

    print(f"\n--- Codes potentiellement réutilisés à tort (>=3 désignations différentes) ---")
    for cle, designations in anomalies_code_suspect.items():
        designations_uniques = set(designations)
        if len(designations_uniques) >= 3:
            print(f"  position/code {cle} utilisé pour {len(designations_uniques)} désignations différentes :")
            for d in list(designations_uniques)[:5]:
                print(f"    - {d}")

    with open("nomenclature_douaniere_normalisee.json", "w", encoding="utf-8") as f:
        json.dump(lignes_normalisees, f, ensure_ascii=False, indent=2)
    print("\nFichier normalisé enregistré : nomenclature_douaniere_normalisee.json")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage : python valider_nomenclature.py nomenclature_douaniere.json")
        sys.exit(1)
    valider(sys.argv[1])
