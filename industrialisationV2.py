from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime, timezone, timedelta
import joblib

# Charger le modèle, le préprocesseur et la liste des features attendues
model = joblib.load("model.joblib")
preprocessor = joblib.load("preprocessor.joblib")
feature_names = joblib.load("feature_names.joblib")
print("feature_names in model:", feature_names)

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Création de l'application FastAPI
app = FastAPI()

# Configuration de CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://web-prediction-inondation-37.onrender.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Fonctions de récupération et d'agrégation des données météo et hydro ---

def fetch_meteo_data():
    """Récupère et agrège les données météo par département."""
    station_meteo = pd.read_csv('geo_station.csv', sep=',', skiprows=1,
                                  names=['typo', 'id_station', 'lat', 'lon', 'code_insee', 'code_postal'])
    meteo_stations = station_meteo[station_meteo['typo'].str.contains('meteo', case=False, na=False)].copy()
    meteo_stations['lat'] = pd.to_numeric(meteo_stations['lat'], errors='coerce')
    meteo_stations['lon'] = pd.to_numeric(meteo_stations['lon'], errors='coerce')
    meteo_stations = meteo_stations.dropna(subset=['lat', 'lon'])
    
    now = datetime.now()
    end_time = now + timedelta(hours=24)
    meteo_results = []
    
    for _, row in meteo_stations.iterrows():
        lat, lon, code_insee = row['lat'], row['lon'], row['code_insee']
        url = (f'https://www.infoclimat.fr/public-api/gfs/json?_ll={lat},{lon}'
               f'&_auth=CBIEEwB%2BXX9Sf1BnB3FXflgwV2JaLAMkC3cKaQ9qVClVPgBhAWFVM14wUC1Qfws9U34FZgswV2dQO1AoXC4FZAhiBGgAa106Uj1QNQcoV3xYYlc0WmUDPQtsCnIPfVQ%2BVTYAegFmVTdeL1AzUGMLOlN%2BBWcLNVdqUCxQKFwwBWEIaARmAGFdP1I%2BUDQHN1dlWHRXKFpjAzwLbwo%2BD2dUY1UzAGIBZ1UwXjRQZ1BlCzZTfgVhCzFXbFAyUDBcMgVnCG8EfwB8XUZSTlAvB3dXIVg%2BV3FaeANuCzYKOQ%3D%3D&_c=79a3845157a89ae350d7afdba9133c8c')
        try:
            response = requests.get(url)
            response.raise_for_status()
            json_data = response.json()
            for timestamp, data in json_data.items():
                if isinstance(data, dict):
                    timestamp_dt = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')
                    if now <= timestamp_dt <= end_time:
                        meteo_results.append({
                            'timestamp': timestamp,
                            'lat': lat,
                            'lon': lon,
                            'pluie': data.get('pluie', 0),
                            'temperature': data.get('temperature', {}).get('2m', None),
                            'vent_moyen': data.get('vent_moyen', {}).get('10m', None),
                            'vent_direction': data.get('vent_direction', {}).get('10m', None),
                            'humidite_moyen': data.get('humidite', {}).get('2m', None),
                            'code_insee': code_insee
                        })
        except Exception as e:
            logging.error(f"Erreur API météo pour la station {code_insee}: {e}")
    
    if meteo_results:
        meteo_df = pd.DataFrame(meteo_results)
        meteo_df['temperature_celsius'] = meteo_df['temperature'].apply(
            lambda x: x - 273.15 if x and x > 100 else x)
        insee_stats = meteo_df.groupby('code_insee').agg(
            nb_releves=('timestamp', 'nunique'),
            portee_prevision=('timestamp', 'max'),
            cumul_pluie_insee=('pluie', 'sum'),
            temp_moyenne_insee=('temperature_celsius', 'mean'),
            vent_moyen_insee=('vent_moyen', 'mean'),
            humidite_insee=('humidite_moyen', 'mean'),
            vent_direction_insee=('vent_direction', 'mean')
        ).reset_index()
        n_unique = insee_stats['cumul_pluie_insee'].nunique()
        q = min(10, n_unique)
        insee_stats['decile_cumul_pluie'] = pd.qcut(
            insee_stats['cumul_pluie_insee'], q=q, labels=False, duplicates="drop"
        ) + 1
        # Création d'une colonne "departement" à partir des 2 premiers caractères de code_insee
        insee_stats['departement'] = insee_stats['code_insee'].str[:2]
        dep_stats = insee_stats.groupby('departement').agg(
            pluie_dep=('cumul_pluie_insee', 'mean'),
            temp_dep=('temp_moyenne_insee', 'mean'),
            max_portee=('portee_prevision', 'max'),
            vent_moyen=('vent_moyen_insee', 'mean'),
            humidite=('humidite_insee', 'mean'),
            vent_direction=('vent_direction_insee', 'mean'),
            decile_pluie=('decile_cumul_pluie', 'mean')
        ).reset_index()
        dep_stats['max_portee_date'] = pd.to_datetime(dep_stats['max_portee']).dt.date
        return dep_stats
    else:
        logging.warning("Aucune donnée météo collectée.")
        return pd.DataFrame()

def fetch_hydro_data():
    """Récupère et agrège les données hydrométriques par département."""
    station_meteo = pd.read_csv('geo_station.csv', sep=',', skiprows=1,
                                  names=['typo', 'id_station', 'lat', 'lon', 'code_insee', 'code_postal'])
    hydro_stations = station_meteo[station_meteo['typo'].str.contains('hydro', case=False, na=False)].copy()
    hydro_stations['code_site'] = hydro_stations['id_station'].str[:8]
    hydro_stations = hydro_stations.dropna(subset=['lat', 'lon', 'code_insee']).reset_index(drop=True)
    
    decile_thresholds = [208, 424, 705, 1115, 1693, 2646, 4516, 8935, 25054]
    date_now = datetime.now(timezone.utc)
    date_debut = (date_now - timedelta(days=1)).strftime('%Y-%m-%dT00:00:00Z')
    date_fin = date_now.strftime('%Y-%m-%dT%H:%M:%SZ')
    
    for i, row in hydro_stations.iterrows():
        code_site = row['code_site']
        url = (
            f"https://hubeau.eaufrance.fr/api/v1/hydrometrie/observations_tr"
            f"?code_entite={code_site}&date_debut_obs={date_debut}"
            f"&date_fin_obs={date_fin}&grandeur_hydro=Q&size=100&sort=desc"
        )
        try:
            response = requests.get(url)
            response.raise_for_status()
            json_data = response.json()
            if json_data and 'data' in json_data and json_data['data']:
                max_obs = max(json_data['data'], key=lambda x: x.get('resultat_obs', float('-inf')))
                hydro_stations.loc[i, 'code_station'] = max_obs['code_station']
                hydro_stations.loc[i, 'resultat_obs'] = max_obs['resultat_obs']
                date_obs = pd.to_datetime(max_obs['date_obs'])
                hydro_stations.loc[i, 'date_obs'] = date_obs.strftime('%Y-%m-%d %H:%M:%S')
        except Exception as e:
            logging.error(f"Erreur API hydrométrie pour la station {code_site}: {e}")
    
    hydro_stations['resultat_obs_code_insee'] = hydro_stations.groupby('code_insee')['resultat_obs'].transform('max')
    hydro_stations['imputation'] = hydro_stations['resultat_obs_code_insee'].isna().astype(int)
    hydro_stations.loc[hydro_stations['imputation'] == 0, 'decile_insee'] = hydro_stations.loc[
        hydro_stations['imputation'] == 0, 'resultat_obs_code_insee'
    ].apply(lambda x: sum(x > t for t in decile_thresholds) + 1)
    hydro_stations['date_obs'] = pd.to_datetime(hydro_stations['date_obs'], errors='coerce')
    hydro_stations['decile_dep_moyen'] = hydro_stations.groupby(hydro_stations['code_insee'].str[:2])['decile_insee'].transform('mean')
    # Créer la colonne "departement" à partir de code_insee
    hydro_stations['departement'] = hydro_stations['code_insee'].str[:2]
    
    hydro_valid = hydro_stations[hydro_stations['imputation'] == 0].copy()
    if not hydro_valid.empty:
        filtered = hydro_valid[['date_obs', 'resultat_obs_code_insee', 'decile_insee', 'departement']].dropna().drop_duplicates()
        filtered['date_sans_heures'] = filtered['date_obs'].dt.date
        hydro_dep = filtered.groupby('departement').agg(
            debit_dep=('resultat_obs_code_insee', 'mean'),
            decile_dep=('decile_insee', 'mean'),
            portee_date_debit=('date_sans_heures', 'max')
        ).reset_index()
        hydro_dep['portee_date_debit'] = pd.to_datetime(hydro_dep['portee_date_debit']) + timedelta(days=1)
        return hydro_dep
    else:
        logging.warning("Aucune donnée hydrométrique valide collectée.")
        return pd.DataFrame()

# --- Fonction d'enrichissement par les données statiques ---



def enrich_with_static_data(resultat: pd.DataFrame) -> pd.DataFrame:
    """
    Enrichit le DataFrame issu de la fusion (resultat)
    avec les données statiques des communes, zones inondables et zones de montagne.
    Les fichiers CSV doivent être dans le répertoire de l'application.
    """
    # Assurer que 'departement' existe dans resultat ; sinon le créer à partir de 'code_insee'
    if 'departement' not in resultat.columns and 'code_insee' in resultat.columns:
        resultat['departement'] = resultat['code_insee'].astype(str).str.zfill(5).str[:2]
    
    # Sauvegarder la colonne "departement" avant de faire le merge
    dept_temp = resultat['departement'].copy()
    
    # Chargement des fichiers statiques
    communes_df = pd.read_csv('communes_de_france.csv', sep=';')
    communes_df['code_insee'] = communes_df['code_insee'].astype(str).str.zfill(5)
    communes_df['dep_commune'] = communes_df['code_insee'].str[:-3]
    
    zones_inondables_df = pd.read_csv('atlas_zones_inondables.csv')
    zones_montagne_df = pd.read_csv('zones_montagne.csv', sep=';', encoding='latin1')
    
    # Fusion avec communes_df : utiliser 'departement' de resultat pour correspondre à 'dep_commune'
    df = pd.merge(resultat, communes_df, left_on='departement', right_on='dep_commune', how='left')
    df = df.reset_index(drop=True)
    
    # Réinjecter la colonne "departement" sauvegardée
    df['departement'] = dept_temp
    
    # Fusion avec les zones inondables
    zones_inondables_df['code_commune'] = zones_inondables_df['code_commune'].astype(str)
    df = pd.merge(df, zones_inondables_df, left_on='code_insee', right_on='code_commune', how='left')
    df = df.reset_index(drop=True)
    df['zone_inondable'] = df['code_commune'].notnull().astype(int)
    
    # Fusion avec les zones de montagne
    df = pd.merge(df, zones_montagne_df, left_on='code_insee', right_on='CodeCommune', how='left')
    df['montagne'] = df['Montagne'].apply(lambda x: 1 if x == 'M' else 0)
    df['ZoneDefavoriseeSimple1'] = df['ZoneDefavoriseeSimple'].apply(lambda x: 1 if x == 'ZDS' else 0)
    
    # Création de colonnes supplémentaires
    df['jour'] = np.cos(2 * np.pi * datetime.now().timetuple().tm_yday / 365)
    df['mois'] = datetime.now().month
    df['saison_inondation'] = df['mois'].apply(lambda x: 0 if x in [1, 2, 3, 10, 11, 12] else 1)
    
    # IMPORTANT : Ne pas supprimer "departement"
    colonnes_a_supprimer = ['code_insee', 'code_postal', 'dep_commune', 
                             'code_commune', 'dep', 'CodeCommune', 'NomCommune', 'Montagne',
                             'ZoneDefavoriseeSimple', 'ZoneHandicapSpecifique', 'date_obs_elab',
                             'academie_nom_Versailles','academie_nom_Reims','academie_nom_Amiens',
                             'academie_nom_Corse','academie_nom_Limoges','academie_nom_Nice','academie_nom_Paris',
                             'max_portee','max_portee_date']
    df = df.drop(columns=colonnes_a_supprimer, errors='ignore')
    print(df.info())
    
    # Transformation en numériques (uniquement les colonnes déjà numériques)
    #df = df.select_dtypes(include=['number', 'bool'])
    bool_cols = df.select_dtypes(include=['bool']).columns
    df[bool_cols] = df[bool_cols].astype(int)
    
    df = df.reset_index(drop=True).drop_duplicates()
    return df




# --- Fonction de fusion et d'enrichissement global ---
def merge_data():
    """
    Fusionne les données hydrométriques et météorologiques puis
    les enrichit avec les données statiques.
    Cette fonction est censée être appelée une fois par jour.
    """
    meteo_df = fetch_meteo_data()
    hydro_df = fetch_hydro_data()
    
    if meteo_df.empty or hydro_df.empty:
        logging.error("Données insuffisantes pour fusionner.")
        return pd.DataFrame()
    
    # Fusion des données hydro et meteo sur la colonne "departement"
    merged_df = pd.merge(
        hydro_df,
        meteo_df,
        on="departement",
        how="inner"
    )
    
    # Enrichissement avec les données statiques
    final_df = enrich_with_static_data(merged_df)
    final_df=final_df.drop_duplicates()
    # Sauvegarder les inputs enrichis pour vérification
    final_df.to_csv('file1.csv', index=False)
    return final_df

# --- Fonction de pré-calcul des prédictions quotidiennes ---





import pandas as pd
import logging




# Mapping des colonnes de df_final1 vers celles attendues par le modèle
feature_rename_map = {
    "pluie_dep": "pluie_24h",
    "zone_inondable": "inondable",
    "montagne": "montagne1",
    "ZoneDefavoriseeSimple1": "zone_defavorisee_simple1",
    "debit_dep": "resultat_obs_elab"
}


def precompute_predictions():
    """
    Parcourt les données quotidiennes enrichies (daily_data),
    calcule la prédiction pour chaque combinaison unique de 
    'departement', 'zone_inondable', 'montagne' et 'ZoneDefavoriseeSimple1'
    et sauvegarde les résultats dans un fichier CSV.
    """
    if daily_data.empty:
        logging.error("Aucune donnée quotidienne disponible pour la prédiction.")
        return pd.DataFrame()

    predictions_list = []
    extreme_features_order = ['resultat_obs_elab', 'pluie_24h', 'vent_moyen']  # Variables à scaler

    # Group by la combinaison souhaitée
    group_cols = ['departement', 'inondable', 'montagne1'] #, 'zoneDefavoriseesimple1'
    for group, group_df in daily_data.groupby(group_cols):
        # group est un tuple contenant (departement, zone_inondable, montagne, ZoneDefavoriseeSimple1)
        # group_df correspond aux données de ce groupe
        if group_df.empty:
            logging.warning(f"Aucune donnée pour le groupe {group}.")
            continue

        dept_data = group_df.copy()

        # Renommer si besoin (utiliser votre mapping si nécessaire)
        dept_data = dept_data.rename(columns=feature_rename_map)

        # One-hot encoding sur 'academie_nom' et 'mois'
        df_encoded = pd.get_dummies(dept_data, columns=['academie_nom', 'mois'], prefix=['academie_nom', 'mois_lettre'])

        # Vérifier et ajouter les features manquantes attendues par le modèle
        missing_cols = set(feature_names) - set(df_encoded.columns)
        for col in missing_cols:
            df_encoded[col] = 0

        # Réorganiser les colonnes pour matcher exactement feature_names
        df_final = df_encoded.reindex(columns=feature_names, fill_value=0)

        # Vérifier et appliquer le transformateur "extreme" sur les features concernées
        if "extreme" in preprocessor.named_transformers_:
            available_extreme_features = [feat for feat in extreme_features_order if feat in df_final.columns]
            if available_extreme_features and df_final[available_extreme_features].shape[0] > 0:
                df_extreme = df_final[available_extreme_features].fillna(0)
                df_final[available_extreme_features] = preprocessor.named_transformers_["extreme"].transform(df_extreme)

        # Vérification finale
        if df_final.shape[0] == 0:
            logging.warning(f"Aucune donnée après reindex pour le groupe {group}.")
            continue

        # Faire la prédiction
        try:
            prediction = model.predict(df_final)
            pred_value = int(prediction[0])
        except Exception as e:
            logging.error(f"Erreur lors de la prédiction pour le groupe {group}: {e}")
            continue

        # Créer un dictionnaire qui inclut la combinaison de groupe
        prediction_info = {
            "departement": group[0],
            "inondable": group[1],
            "montagne1": group[2],
            "zoneDefavoriseesimple1": group[3],
            "prediction": pred_value #,
            #"annonce": "Risque faible d'inondation" if pred_value == 0 else "Attention, risque d'inondation important"
        }
        #predictions_list.append(prediction_info)
        # Calculer la probabilité maximale par département
max_proba_df = predictions_df.groupby("departement")["proba inondation en pourcentage"].max().reset_index()
max_proba_df = max_proba_df.rename(columns={"proba inondation en pourcentage": "max_proba"})
        # Fusionner cette information avec le DataFrame de prédictions
predictions_df = predictions_df.merge(max_proba_df, on="departement", how="left")

        # Mettre à jour le message "annonce" en fonction de la prédiction et de la probabilité maximale
def update_annonce(row):
    if row["prediction"] == 0:
        return (f"Risque faible d'inondation; dans votre département la zone la plus à risque "
                f"a {row['max_proba']}% de subir une inondation")
    else:
        return (f"Attention, risque d'inondation important; dans votre département la zone la plus à risque "
                f"a {row['max_proba']}% de subir une inondation")

predictions_df["annonce"] = predictions_df.apply(update_annonce, axis=1)


    # predictions_df = pd.DataFrame(predictions_list)
    # predictions_df.to_csv('predictions_daily.csv', index=False)
    # logging.info("Prédictions quotidiennes sauvegardées dans predictions_daily.csv")
    # return predictions_df


# def precompute_predictions():
#     """
#     Parcourt les données quotidiennes enrichies (daily_data),
#     calcule la prédiction pour chaque département et sauvegarde les résultats dans un fichier CSV.
#     """
#     if daily_data.empty:
#         logging.error("Aucune donnée quotidienne disponible pour la prédiction.")
#         return pd.DataFrame()

#     predictions_list = []
#     extreme_features_order = ['resultat_obs_elab', 'pluie_24h', 'vent_moyen']  # Variables à scaler

#     for dept in daily_data['departement'].unique():
#         # Filtrer les données pour le département
#         dept_data = daily_data[daily_data["departement"] == dept].copy()
#         if dept_data.empty:
#             logging.warning(f"Aucune donnée pour le département {dept}.")
#             continue

#         # Renommer les colonnes selon le mapping attendu par le modèle
#         dept_data = dept_data.rename(columns=feature_rename_map)

#         # Appliquer le one-hot encoding sur 'academie_nom' et 'mois'
#         # On suppose que ces colonnes existent dans dept_data
#         df_encoded = pd.get_dummies(dept_data, columns=['academie_nom', 'mois'], prefix=['academie_nom', 'mois_lettre'])

#         # Vérifier et ajouter les features manquantes attendues par le modèle
#         missing_cols = set(feature_names) - set(df_encoded.columns)
#         for col in missing_cols:
#             df_encoded[col] = 0

#         # Réorganiser les colonnes pour matcher exactement feature_names
#         df = df_encoded.reindex(columns=feature_names, fill_value=0)

#         # Avant de scaler, s'assurer qu'il n'y a pas de NaN dans les colonnes extrêmes
#         for feat in extreme_features_order:
#             if feat in df.columns:
#                 df[feat] = df[feat].fillna(0)

#         # Appliquer le transformateur "extreme" uniquement si la sélection n'est pas vide
#         if "extreme" in preprocessor.named_transformers_:
#             available_extreme_features = [feat for feat in extreme_features_order if feat in df.columns]
#             # Vérifier si le DataFrame contient au moins une ligne pour ces features
#             if df[available_extreme_features].shape[0] > 0:
#                 df_extreme = df[available_extreme_features]
#                 # Si le DataFrame de transformation est vide ou si toutes les valeurs sont NaN, on passe
#                 if df_extreme.shape[0] == 0:
#                     logging.warning(f"Les features extrêmes sont vides pour le département {dept}.")
#                 else:
#                     df[available_extreme_features] = preprocessor.named_transformers_["extreme"].transform(df_extreme)

#         # Vérifier que df n'est pas vide
#         if df.shape[0] == 0:
#             logging.warning(f"Aucune donnée après reindex pour le département {dept}.")
#             continue

#         # Vérifier s'il y a des NaN dans le DataFrame final (pour information)
#         if df.isna().sum().sum() > 0:
#             logging.warning(f"Des valeurs manquantes subsistent dans les features pour le département {dept}.")

#         # Faire la prédiction
#         try:
#             prediction = model.predict(df)
#             pred_value = int(prediction[0])
#         except Exception as e:
#             logging.error(f"Erreur lors de la prédiction pour le département {dept}: {e}")
#             continue

#         annonce = "Risque faible d'inondation" if pred_value == 0 else "Attention, risque d'inondation important"

#         predictions_list.append({
#             "departement": dept,
#             "prediction": pred_value,
#             "annonce": annonce
#         })

#     predictions_df = pd.DataFrame(predictions_list)
#     predictions_df.to_csv('predictions_daily.csv', index=False)
#     logging.info("Prédictions quotidiennes sauvegardées dans predictions_daily.csv")
#     return predictions_df

# Mise à jour des données quotidiennes et pré-calcul des prédictions lors du démarrage
daily_data = merge_data()
daily_predictions = precompute_predictions()


# --- Routes de l'API ---
@app.get("/")
def home():
    return {"message": "API de prédiction des risques d'inondation par département"}

# Schéma d'entrée pour la prédiction (seulement le département)
class PredictionInput(BaseModel):
    departement: str

@app.get("/update_daily_data")
def update_daily_data():
    """
    Met à jour manuellement les données quotidiennes et les prédictions.
    En production, cette mise à jour devrait être planifiée (ex. à 03h00).
    """
    global daily_data, daily_predictions
    daily_data = merge_data()
    daily_predictions = precompute_predictions()
    n_dep = len(daily_data) if not daily_data.empty else 0
    return {"message": "Données quotidiennes mises à jour", "n_departements": n_dep}

@app.post("/predict")
def predict(input_data: PredictionInput):
    """
    Renvoie la prédiction pour le département demandé à partir des prédictions pré-calculées.
    """
    dept = input_data.departement
    if daily_predictions.empty:
        return {"error": "Prédictions quotidiennes non disponibles."}
    
    result = daily_predictions[daily_predictions["departement"] == dept]
    if result.empty:
        return {"error": f"Aucune prédiction trouvée pour le département {dept}."}
    
    return {
        "departement": dept,
        "prediction_inondations": int(result["prediction"].iloc[0]),
        "annonce": result["annonce"].iloc[0]
    }
