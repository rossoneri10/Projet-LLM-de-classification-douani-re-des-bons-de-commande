# Classification douanière automatique par LLM local

Système d'analyse automatique des bons de commande / factures, qui identifie
pour chaque produit sa position tarifaire douanière (code SH), son taux de
droit d'importation, et calcule le montant de droit correspondant — le tout
en s'appuyant sur un LLM tournant **entièrement en local** via [Ollama](https://ollama.com)
(aucune donnée envoyée à un service externe).

## Contexte

Projet réalisé dans le cadre d'un stage, pour automatiser la classification
douanière des bons de commande reçus par l'université (fournisseurs comme
Fisher Scientific), en s'appuyant sur la nomenclature douanière officielle.

## Architecture

```
Bon de commande (PDF/image)
        │
        ▼
 [1] Extraction du texte (OCR si scanné)
        │
        ▼
 [2] Extraction structurée des lignes produits (LLM)
        │
        ▼
 [3] Correspondance floue + LLM contre la nomenclature douanière
        │
        ▼
 [4] Calcul du code SH, taux et montant de droit
        │
        ▼
   rapport_douanier.json
```

La base de référence (nomenclature douanière) est construite une fois, en
amont, à partir des PDF de chapitres du tarif des douanes (voir
`extraction_nomenclature_douaniere.py`).

## Scripts

| Script | Rôle |
|---|---|
| `traitement_factures.py` | Extrait le texte d'un bon de commande (PDF natif, PDF scanné ou image) et en extrait les lignes produits structurées via LLM. |
| `extraction_nomenclature_douaniere.py` | Construit la base de référence tarifaire à partir des PDF de chapitres du tarif des douanes. |
| `valider_nomenclature.py` | Détecte les anomalies dans la base extraite (positions invalides, valeurs mal typées) et produit une version normalisée. |
| `moteur_correspondance_sh.py` | Fait correspondre chaque ligne d'un bon de commande à une position tarifaire (code SH), calcule taux et montant de droit. |
| `interface_validation.py` | Interface Streamlit de relecture humaine des correspondances incertaines. |

## Installation

```bash
pip install pymupdf pdf2image pytesseract pillow rapidfuzz requests opencv-python streamlit pandas
```

Dépendances système :
- [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki) (avec le pack de langue française) — pour les bons de commande scannés
- [Poppler](https://github.com/oschwartz10612/poppler-windows/releases) — requis par `pdf2image`
- [Ollama](https://ollama.com) avec au moins un modèle installé :
  ```bash
  ollama pull mistral
  ollama pull qwen2.5:3b
  ```

## Utilisation

**1. Construire la base de référence tarifaire** (une seule fois, à partir des PDF de chapitres du tarif des douanes) :
```bash
python extraction_nomenclature_douaniere.py dossier_des_chapitres/
python valider_nomenclature.py nomenclature_douaniere.json
```

**2. Traiter un bon de commande :**
```bash
python traitement_factures.py facture.pdf
python moteur_correspondance_sh.py facture_rapport.json nomenclature_exploitable.json
```

**3. (Optionnel) Relecture humaine des cas incertains :**
```bash
streamlit run interface_validation.py
```

## Choix de modèle

- `qwen2.5:3b` : utilisé pour l'extraction en volume (nomenclature douanière, bons de commande) — plus rapide sur CPU.
- `mistral` (7B) : utilisé pour la désambiguïsation dans le moteur de correspondance — meilleur raisonnement sur les cas ambigus (ex: reconnaître qu'une "tête de pompe" est une pièce de "pompe").

## Résultat validé

Sur un bon de commande contenant une "Tête de pompe", le système a identifié
le code SH **8413** (pompes pour liquides, taux 2,5%), cohérent avec le tarif
douanier réel du fournisseur (84139100).

## Limites connues

- La base de référence n'est extraite que partiellement depuis les PDF sources (l'extraction par LLM local sur CPU rate parfois des lignes, notamment sur les gros chapitres).
- Le système tourne entièrement en local sur CPU ; les temps de traitement dépendent fortement du matériel.
- La précision du code SH est parfois limitée au niveau "position" (4 chiffres) plutôt qu'à la sous-position complète (10 chiffres).
- Les frais de service (fret, frais d'approche...) ressortent volontairement en `non_resolu` : ce ne sont pas des produits classables.

## Prochaines étapes

- Compléter l'extraction des chapitres restants de la nomenclature
- Affiner la précision au niveau sous-position
- Étendre l'interface de validation humaine pour couvrir le nouveau format (code SH, pas seulement code produit interne)
