# API d'Analyse de Sentiments

API Flask qui évalue le sentiment de tweets à l'aide d'un modèle de
régression logistique (scikit-learn) entraîné sur des données annotées
stockées dans MySQL. Pour chaque tweet, l'API renvoie un score continu
entre **-1** (très négatif) et **1** (très positif).

## Architecture

| Fichier | Rôle |
|---|---|
| `app.py` | API Flask : endpoint `POST /analyze`, `/health`, doc Swagger `/docs` |
| `train_model.py` | Entraîne le modèle, génère les matrices de confusion, sauvegarde le modèle |
| `db.py` | Connexion MySQL et lecture de la table `tweets` |
| `db/init/` | Schéma SQL et données annotées de départ (chargés au démarrage du conteneur MySQL) |
| `scripts/retrain.sh` | Réentraînement automatisable (avec logs horodatés) |
| `scripts/install_cron.sh` | Installe le cron hebdomadaire de réentraînement |

## Prérequis

- Python ≥ 3.14
- [uv](https://docs.astral.sh/uv/) pour la gestion des dépendances
- Docker (pour la base MySQL)

## Installation

```bash
# 1. Cloner puis installer les dépendances
uv sync

# 2. Créer le fichier d'environnement à partir de l'exemple
cp .env.example .env
```

Variables d'environnement (voir `.env.example`) :

| Variable | Défaut | Description |
|---|---|---|
| `MYSQL_HOST` | `127.0.0.1` | Hôte MySQL |
| `MYSQL_PORT` | `3306` | Port MySQL |
| `MYSQL_USER` | `sentiments_user` | Utilisateur |
| `MYSQL_PASSWORD` | `sentiments_password` | Mot de passe |
| `MYSQL_DATABASE` | `sentiments_db` | Base de données |

## Base de données

Le `docker-compose.yml` lance MySQL et charge automatiquement le schéma
(`db/init/01_schema.sql`) et les données annotées (`db/init/02_seed_data.sql`)
au premier démarrage.

```bash
docker compose up -d
```

Structure de la table `tweets` :

| Colonne | Type | Description |
|---|---|---|
| `id` | INT (PK) | Identifiant unique |
| `text` | TEXT | Contenu du tweet |
| `positive` | TINYINT(1) | 1 si positif, 0 sinon |
| `negative` | TINYINT(1) | 1 si négatif, 0 sinon |

## Entraînement du modèle

```bash
uv run python train_model.py
```

Ce script :
- lit les tweets annotés depuis MySQL,
- entraîne une régression logistique (TF-IDF + `LogisticRegression`),
- sauvegarde le modèle et le vectorizer dans `model/`,
- génère les matrices de confusion dans `reports/`.

> ⚠️ L'API charge le modèle depuis `model/`. Il faut donc entraîner le modèle
> **au moins une fois** avant de démarrer l'API.

## Lancer l'API

```bash
uv run python app.py
```

L'API écoute sur `http://127.0.0.1:5000`.

| Endpoint | Méthode | Description |
|---|---|---|
| `/analyze` | POST | Analyse une liste de tweets |
| `/health` | GET | Vérifie que l'API est opérationnelle |
| `/docs` | GET | Documentation interactive (Swagger UI) |

### Exemple d'utilisation

**Requête**

```bash
curl -X POST http://127.0.0.1:5000/analyze \
  -H "Content-Type: application/json" \
  -d '{"tweets": ["I absolutely love this!", "This is the worst experience ever"]}'
```

**Réponse**

```json
{
  "I absolutely love this!": 0.1688,
  "This is the worst experience ever": -0.3246
}
```

### Gestion des erreurs

L'endpoint renvoie un code `400` avec un message explicite dans les cas suivants :

| Cas | Message |
|---|---|
| Champ `tweets` manquant | `Le corps de la requête doit contenir un champ 'tweets'.` |
| `tweets` n'est pas une liste | `'tweets' doit être une liste de chaînes de caractères.` |
| Liste vide | `La liste de tweets ne peut pas être vide.` |
| Élément non-string | `Chaque élément de 'tweets' doit être une chaîne de caractères.` |
| Chaîne vide | `Les tweets ne peuvent pas être des chaînes vides.` |

## Réentraînement automatisé

Le modèle doit être réentraîné régulièrement avec les données les plus récentes
de la table `tweets`.

### Manuellement

```bash
./scripts/retrain.sh
```

La sortie est journalisée dans `logs/retrain.log`.

### Automatiquement (cron hebdomadaire)

Installe une tâche cron qui réentraîne le modèle chaque **lundi à 03h00** :

```bash
./scripts/install_cron.sh
```

Le script est idempotent : le relancer met à jour l'entrée existante sans la
dupliquer. Pour un réglage manuel, voir `scripts/crontab.example`.

Vérifier / retirer la tâche :

```bash
crontab -l                                     # lister
crontab -l | grep -v retrain.sh | crontab -    # retirer
```

## Rapport d'évaluation

Voir le rapport PDF (matrices de confusion, précision/rappel/F1, analyse et
recommandations) disponible dans le dépôt.
