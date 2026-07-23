"""
Interface de validation humaine des lignes extraites des bons de commande
===========================================================================

Charge un rapport JSON produit par traitement_factures.py, affiche chaque
ligne avec son niveau de confiance, et permet de valider, corriger ou
rejeter chaque correspondance avant export final.

Dépendances :
    pip install streamlit pandas

Lancement :
    streamlit run interface_validation.py
"""

import json
import os
from datetime import datetime

import streamlit as st
import pandas as pd


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Validation des bons de commande", layout="wide")

# Référentiel produits d'exemple, à remplacer par le chargement de vos
# fichiers réels (doit être cohérent avec celui utilisé dans traitement_factures.py)
REFERENTIEL_PRODUITS = [
    {"code": "PRD-0012", "designation": "Vis inox M6x20"},
    {"code": "PRD-0045", "designation": "Câble électrique 2.5mm"},
    {"code": "PRD-0078", "designation": "Roulement à billes 6202"},
    {"code": "PRD-0103", "designation": "Joint torique 15mm"},
]

COULEURS_CONFIANCE = {
    "exacte": "🟢",
    "llm": "🟠",
    "non_resolu": "🔴",
}


def icone_confiance(confiance):
    if confiance and confiance.startswith("floue"):
        return "🟡"
    return COULEURS_CONFIANCE.get(confiance, "⚪")


# ---------------------------------------------------------------------------
# Chargement du rapport
# ---------------------------------------------------------------------------

st.title("Validation des bons de commande")

fichier_charge = st.file_uploader("Charger un rapport JSON", type=["json"])

if "lignes" not in st.session_state:
    st.session_state.lignes = None
    st.session_state.nom_source = None

if fichier_charge is not None:
    contenu = json.load(fichier_charge)
    st.session_state.lignes = contenu["lignes"]
    st.session_state.nom_source = contenu.get("fichier_source", fichier_charge.name)

if st.session_state.lignes is None:
    st.info("Chargez un rapport JSON généré par le script de traitement pour commencer.")
    st.stop()

st.caption(f"Source : {st.session_state.nom_source} — {len(st.session_state.lignes)} ligne(s)")


# ---------------------------------------------------------------------------
# Filtres
# ---------------------------------------------------------------------------

col_filtre1, col_filtre2 = st.columns([1, 3])
with col_filtre1:
    afficher_seulement_a_verifier = st.checkbox(
        "Afficher seulement les lignes à vérifier", value=True
    )

options_produits = [f"{p['code']} — {p['designation']}" for p in REFERENTIEL_PRODUITS]
options_produits_avec_vide = ["— Aucune correspondance —"] + options_produits


# ---------------------------------------------------------------------------
# Affichage et validation ligne par ligne
# ---------------------------------------------------------------------------

for i, ligne in enumerate(st.session_state.lignes):
    confiance = ligne.get("confiance", "non_resolu")
    deja_validee = ligne.get("valide_par_humain", False)

    if afficher_seulement_a_verifier and confiance == "exacte":
        continue
    if afficher_seulement_a_verifier and deja_validee:
        continue

    with st.container(border=True):
        col_info, col_action = st.columns([2, 2])

        with col_info:
            st.markdown(
                f"**{icone_confiance(confiance)} {ligne.get('designation', '—')}** "
                f"(code extrait : `{ligne.get('code_produit', '—')}`)"
            )
            st.caption(
                f"Quantité : {ligne.get('quantite', '—')} · "
                f"Prix unitaire : {ligne.get('prix_unitaire', '—')} · "
                f"Confiance : {confiance}"
            )
            produit_ref = ligne.get("produit_reference")
            if produit_ref:
                st.caption(f"Correspondance actuelle : {produit_ref['code']} — {produit_ref['designation']}")
            if ligne.get("raison_llm"):
                st.caption(f"Raisonnement du modèle : {ligne['raison_llm']}")

        with col_action:
            valeur_actuelle = (
                f"{produit_ref['code']} — {produit_ref['designation']}"
                if produit_ref else "— Aucune correspondance —"
            )
            index_defaut = (
                options_produits_avec_vide.index(valeur_actuelle)
                if valeur_actuelle in options_produits_avec_vide else 0
            )
            choix = st.selectbox(
                "Produit correct",
                options_produits_avec_vide,
                index=index_defaut,
                key=f"choix_{i}",
            )

            bouton_col1, bouton_col2 = st.columns(2)
            with bouton_col1:
                if st.button("Valider", key=f"valider_{i}", type="primary"):
                    if choix == "— Aucune correspondance —":
                        ligne["produit_reference"] = None
                    else:
                        code_choisi = choix.split(" — ")[0]
                        ligne["produit_reference"] = next(
                            p for p in REFERENTIEL_PRODUITS if p["code"] == code_choisi
                        )
                    ligne["confiance"] = "validee_humain"
                    ligne["valide_par_humain"] = True
                    st.rerun()
            with bouton_col2:
                if st.button("Rejeter la ligne", key=f"rejeter_{i}"):
                    ligne["produit_reference"] = None
                    ligne["confiance"] = "rejetee_humain"
                    ligne["valide_par_humain"] = True
                    st.rerun()


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

st.divider()

nb_validees = sum(1 for l in st.session_state.lignes if l.get("valide_par_humain"))
st.progress(
    nb_validees / len(st.session_state.lignes) if st.session_state.lignes else 0,
    text=f"{nb_validees} / {len(st.session_state.lignes)} ligne(s) traitée(s)",
)

if st.button("Exporter le rapport validé"):
    rapport_final = {
        "fichier_source": st.session_state.nom_source,
        "date_validation": datetime.now().isoformat(),
        "nb_lignes": len(st.session_state.lignes),
        "lignes": st.session_state.lignes,
    }
    nom_export = f"{os.path.splitext(st.session_state.nom_source)[0]}_valide.json"
    st.download_button(
        "Télécharger le JSON validé",
        data=json.dumps(rapport_final, ensure_ascii=False, indent=2),
        file_name=nom_export,
        mime="application/json",
    )

    # Aperçu tabulaire
    df = pd.DataFrame([
        {
            "code_extrait": l.get("code_produit"),
            "designation_extraite": l.get("designation"),
            "code_final": (l.get("produit_reference") or {}).get("code"),
            "designation_finale": (l.get("produit_reference") or {}).get("designation"),
            "statut": l.get("confiance"),
        }
        for l in st.session_state.lignes
    ])
    st.dataframe(df, use_container_width=True)
