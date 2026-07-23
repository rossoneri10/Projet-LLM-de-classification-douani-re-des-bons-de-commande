"""
Moteur de correspondance douanière
=====================================

Prend en entrée :
  - le rapport JSON d'un bon de commande (produit par traitement_factures.py)
  - la base de nomenclature douanière (produite par
    extraction_nomenclature_douaniere.py, filtrée par valider_nomenclature.py)

Pour chaque produit du bon de commande, trouve la position tarifaire (SH) la
plus probable par correspondance de désignation (les bons de commande n'ont
pas de code SH, contrairement au projet initial de catalogue interne — la
correspondance se fait donc sur le texte de la désignation, pas sur un code).

Calcule ensuite le montant du droit de douane et le prix total pour chaque
ligne.

Dépendances :
    pip install rapidfuzz requests

Utilisation :
    python moteur_correspondance_sh.py rapport_facture.json nomenclature_exploitable.json
"""

import sys
import os
import json
import re
import requests

from rapidfuzz import process, fuzz


OLLAMA_URL = "http://localhost:11434/api/generate"
MODELE = "mistral"

SEUIL_ACCEPTATION_AUTOMATIQUE = 85  # score rapidfuzz au-dessus duquel on accepte sans LLM
NB_CANDIDATS_POUR_LLM = 5


# ---------------------------------------------------------------------------
# Chargement du référentiel
# ---------------------------------------------------------------------------

def charger_referentiel(chemin_json):
    with open(chemin_json, "r", encoding="utf-8") as f:
        lignes = json.load(f)

    for l in lignes:
        l["sh_code"] = "".join(filter(None, [
            (l.get("position") or "").replace(".", ""),
            (l.get("sous_position") or "").replace(".", "")[-2:] if l.get("sous_position") else "",
            l.get("code") or "",
        ]))
    return lignes


# ---------------------------------------------------------------------------
# Correspondance floue par désignation
# ---------------------------------------------------------------------------

def trouver_candidats(designation_extraite, referentiel, n=NB_CANDIDATS_POUR_LLM):
    designations_ref = [l["designation"] for l in referentiel]
    resultats = process.extract(
        designation_extraite, designations_ref, scorer=fuzz.token_set_ratio, limit=n
    )
    candidats = []
    for designation_trouvee, score, index in resultats:
        candidats.append((referentiel[index], score))
    return candidats


# ---------------------------------------------------------------------------
# Désambiguïsation via Mistral/Qwen
# ---------------------------------------------------------------------------

PROMPT_DESAMBIGUISATION = """Tu dois déterminer quelle position tarifaire douanière
correspond le mieux à un produit d'un bon de commande.

Produit à classer : {designation}

Candidats possibles dans la nomenclature douanière (code SH puis désignation
officielle) :
{candidats}

Consignes importantes :
- Le produit à classer peut être désigné de façon différente de la nomenclature
  officielle (nom commercial, langue différente, abréviation). Cherche le lien
  logique, pas une correspondance mot pour mot.
- Une PIÈCE, un COMPOSANT ou une VARIANTE d'un produit (ex: "tête de pompe" est
  une pièce de "pompe") doit être rattaché à la position du produit général
  correspondant, sauf s'il existe un candidat plus spécifique pour les pièces.
- Choisis le candidat le plus plausible même si la correspondance n'est pas
  parfaite, du moment qu'elle est raisonnable d'un point de vue métier.
- Ne réponds null que si vraiment AUCUN candidat n'a de rapport avec le produit
  (ex: un produit de nettoyage face à des candidats sur l'électronique).

Réponds UNIQUEMENT avec un JSON au format :
{{"sh_code_choisi": "...", "raison": "..."}}
"""


def appeler_mistral(prompt, timeout=300):
    payload = {"model": MODELE, "prompt": prompt, "stream": False, "format": "json"}
    reponse = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
    reponse.raise_for_status()
    return reponse.json()["response"]


def desambiguiser_avec_llm(designation_extraite, candidats):
    texte_candidats = "\n".join(
        f"- {c['sh_code']} : {c['designation']} (taux {c['droit_importation_pct']}%)"
        for c, score in candidats
    )
    prompt = PROMPT_DESAMBIGUISATION.format(
        designation=designation_extraite, candidats=texte_candidats
    )
    try:
        sortie = appeler_mistral(prompt)
    except requests.exceptions.RequestException as e:
        print(f"    Timeout LLM pour '{designation_extraite}': {e}")
        return {"sh_code_choisi": None, "raison": "Timeout du modèle"}

    match = re.search(r"\{.*\}", sortie, re.DOTALL)
    contenu = match.group(0) if match else sortie
    try:
        return json.loads(contenu)
    except json.JSONDecodeError:
        return {"sh_code_choisi": None, "raison": "Réponse LLM non parsable"}


# ---------------------------------------------------------------------------
# Résolution d'une ligne du bon de commande
# ---------------------------------------------------------------------------

def resoudre_ligne(ligne_bon_commande, referentiel):
    designation = ligne_bon_commande.get("designation") or ""
    if not designation.strip():
        return {**ligne_bon_commande, "sh_code": None, "confiance": "non_resolu"}

    candidats = trouver_candidats(designation, referentiel)
    if not candidats:
        return {**ligne_bon_commande, "sh_code": None, "confiance": "non_resolu"}

    meilleur_produit, meilleur_score = candidats[0]

    if meilleur_score >= SEUIL_ACCEPTATION_AUTOMATIQUE:
        produit_ref = meilleur_produit
        confiance = f"floue ({meilleur_score}%)"
    else:
        decision = desambiguiser_avec_llm(designation, candidats)
        code_choisi = decision.get("sh_code_choisi")
        produit_ref = next((c for c, s in candidats if c["sh_code"] == code_choisi), None)
        confiance = "llm" if produit_ref else "non_resolu"

    resultat = {**ligne_bon_commande, "confiance": confiance}

    if produit_ref:
        quantite = ligne_bon_commande.get("quantite") or 0
        prix_unitaire = ligne_bon_commande.get("prix_unitaire") or 0
        taux = produit_ref.get("droit_importation_pct") or 0

        montant_ht = quantite * prix_unitaire
        montant_droit = montant_ht * taux / 100
        montant_total = montant_ht + montant_droit

        resultat.update({
            "sh_code": produit_ref["sh_code"],
            "designation_douaniere": produit_ref["designation"],
            "taux_droit_pct": taux,
            "montant_ht": round(montant_ht, 2),
            "montant_droit": round(montant_droit, 2),
            "montant_total_avec_droit": round(montant_total, 2),
        })
    else:
        resultat.update({
            "sh_code": None,
            "designation_douaniere": None,
            "taux_droit_pct": None,
        })

    return resultat


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage : python moteur_correspondance_sh.py rapport_facture.json nomenclature_exploitable.json")
        sys.exit(1)

    chemin_facture, chemin_referentiel = sys.argv[1], sys.argv[2]

    with open(chemin_facture, "r", encoding="utf-8") as f:
        rapport_facture = json.load(f)

    referentiel = charger_referentiel(chemin_referentiel)
    print(f"Référentiel chargé : {len(referentiel)} position(s) tarifaire(s).")

    lignes_facture = rapport_facture.get("lignes", rapport_facture)
    print(f"Traitement de {len(lignes_facture)} ligne(s) du bon de commande...\n")

    resultats = []
    for i, ligne in enumerate(lignes_facture, 1):
        print(f"  Ligne {i}/{len(lignes_facture)} : {ligne.get('designation', '?')[:50]}...")
        resultats.append(resoudre_ligne(ligne, referentiel))

    rapport_final = {
        "fichier_source": rapport_facture.get("fichier_source"),
        "nb_lignes": len(resultats),
        "lignes": resultats,
    }

    nom_sortie = "rapport_douanier.json"
    with open(nom_sortie, "w", encoding="utf-8") as f:
        json.dump(rapport_final, f, ensure_ascii=False, indent=2)

    print(f"\nRapport enregistré : {nom_sortie}")

    # Résumé rapide en console
    resolues = sum(1 for r in resultats if r.get("sh_code"))
    print(f"{resolues}/{len(resultats)} ligne(s) classée(s) avec succès.")
