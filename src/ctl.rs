//! Interrogation du serveur en marche, par HTTP.
//!
//! `ft ctl` fait la même chose en ligne de commande, mais il exige que le
//! moteur soit installé et activé dans l'environnement courant. Or ces
//! questions — le serveur répond-il, que tient-il en mémoire, quel modèle
//! sert-il — se posent précisément quand on doute de l'installation. Les poser
//! en HTTP les rend indépendantes de tout ce qui pourrait manquer par ailleurs.

use serde::Serialize;
use std::time::Duration;

/// L'URL de base du serveur.
///
/// L'hôte la donne dans `FREETOKEN_LOCARYN_ENDPOINT` quand il lance le moteur ;
/// le serveur MCP, lui, la reconstruit depuis le port du manifeste.
pub fn base_url() -> String {
    std::env::var("FREETOKEN_LOCARYN_ENDPOINT")
        .ok()
        .map(|u| u.trim_end_matches('/').to_string())
        .filter(|u| !u.is_empty())
        .unwrap_or_else(|| format!("http://127.0.0.1:{}", crate::DEFAULT_PORT))
}

fn client() -> Result<reqwest::Client, String> {
    reqwest::Client::builder()
        .timeout(Duration::from_secs(10))
        .build()
        .map_err(|e| e.to_string())
}

/// Un `GET` qui rend le corps en JSON, ou dit que le serveur ne répond pas.
async fn get(chemin: &str) -> Result<serde_json::Value, String> {
    let url = format!("{}{chemin}", base_url());
    let reponse = client()?.get(&url).send().await.map_err(|e| {
        format!(
            "le serveur ne répond pas sur {url} ({e}) — il n'est probablement pas démarré. \
             Choisissez un modèle dans Réglages → Moteur pour le lancer."
        )
    })?;
    let statut = reponse.status();
    let texte = reponse.text().await.unwrap_or_default();
    if !statut.is_success() {
        return Err(format!("{url} a répondu {statut} : {texte}"));
    }
    serde_json::from_str(&texte).or_else(|_| Ok(serde_json::json!({ "raw": texte })))
}

/// Ce que le serveur dit de lui-même : chargé ou en cours, et sur quel modèle.
pub async fn health() -> Result<serde_json::Value, String> {
    get("/health").await
}

/// Débit, latence, VRAM, occupation des files.
pub async fn stats() -> Result<serde_json::Value, String> {
    get("/v1/stats").await
}

/// L'état des réservoirs de cache : experts sur le GPU, jetons de KV.
pub async fn cache_status() -> Result<serde_json::Value, String> {
    get("/v1/cache/status").await
}

/// Les modèles réellement servis, tels que les clients doivent les nommer.
pub async fn models() -> Result<serde_json::Value, String> {
    get("/v1/models").await
}

/// Les dernières requêtes vues par le serveur.
pub async fn requests(limit: Option<u32>) -> Result<serde_json::Value, String> {
    match limit {
        Some(n) => get(&format!("/v1/requests?limit={n}")).await,
        None => get("/v1/requests").await,
    }
}

/// Ce qu'un redimensionnement de cache demande.
#[derive(Debug, Clone, Default, Serialize)]
pub struct CacheRebuild {
    /// Emplacements du cache d'experts sur le GPU.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub moe: Option<u64>,
    /// Capacité du cache KV, en jetons.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kv: Option<u64>,
}

impl CacheRebuild {
    pub fn is_empty(&self) -> bool {
        self.moe.is_none() && self.kv.is_none()
    }
}

/// Redimensionne les réservoirs sans redémarrer le serveur.
///
/// C'est ce qui permet de rendre de la VRAM à une autre tâche — une génération
/// d'image, par exemple — sans perdre le modèle déjà chargé, ce qui coûterait
/// plusieurs minutes de rechargement.
pub async fn cache_rebuild(demande: &CacheRebuild) -> Result<serde_json::Value, String> {
    if demande.is_empty() {
        return Err("indiquez au moins « moe » ou « kv »".into());
    }
    let url = format!("{}/v1/cache/rebuild", base_url());
    let reponse = client()?
        .post(&url)
        .json(demande)
        .send()
        .await
        .map_err(|e| format!("le serveur ne répond pas sur {url} ({e})"))?;
    let statut = reponse.status();
    let texte = reponse.text().await.unwrap_or_default();
    if !statut.is_success() {
        return Err(format!("{url} a répondu {statut} : {texte}"));
    }
    serde_json::from_str(&texte).or_else(|_| Ok(serde_json::json!({ "raw": texte })))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// L'URL de base tient compte de ce que l'hôte a passé, sans double barre
    /// oblique : `…1919//health` répond 404 et fait croire le serveur mort.
    #[test]
    fn l_url_de_base_perd_sa_barre_finale() {
        // SAFETY: test mono-thread sur une variable qui n'est lue qu'ici.
        unsafe {
            std::env::set_var("FREETOKEN_LOCARYN_ENDPOINT", "http://127.0.0.1:1919/");
        }
        assert_eq!(base_url(), "http://127.0.0.1:1919");
        unsafe {
            std::env::remove_var("FREETOKEN_LOCARYN_ENDPOINT");
        }
        assert_eq!(base_url(), "http://127.0.0.1:1919");
    }

    #[test]
    fn un_redimensionnement_vide_est_refuse() {
        assert!(CacheRebuild::default().is_empty());
    }

    /// Seuls les réservoirs demandés sont envoyés : un `null` ferait remettre
    /// un réservoir à zéro chez le moteur.
    #[test]
    fn seuls_les_reservoirs_demandes_partent() {
        let json = serde_json::to_string(&CacheRebuild {
            moe: Some(4096),
            kv: None,
        })
        .unwrap();
        assert_eq!(json, r#"{"moe":4096}"#);
    }
}
