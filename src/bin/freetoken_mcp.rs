//! Serveur MCP stdio de l'extension FreeToken.
//!
//! `stdout` est réservé au JSON-RPC ; tout diagnostic passe par `stderr` sous
//! peine de casser le protocole.
//!
//! Ces outils répondent aux questions que le modèle et l'utilisateur se posent
//! au sujet du moteur : est-il installé, que manque-t-il, que tient-il en
//! mémoire, comment lui rendre de la VRAM. Le **choix du modèle et le
//! démarrage** ne sont pas ici : ils appartiennent à l'écran Réglages → Moteur
//! du socle, qui enregistre le modèle actif et supervise le processus. Un outil
//! qui lancerait le serveur en parallèle donnerait deux propriétaires au même
//! port.

use locaryn_plugin_freetoken as ft;
use serde_json::{json, Value};
use std::io::Write;
use std::time::Duration;
use tokio::io::{AsyncBufReadExt, BufReader};

const VERSION: &str = "1.0.0";

#[tokio::main]
async fn main() {
    let mut lignes = BufReader::new(tokio::io::stdin()).lines();
    while let Ok(Some(ligne)) = lignes.next_line().await {
        if ligne.trim().is_empty() {
            continue;
        }
        let reponse = match serde_json::from_str::<Value>(&ligne) {
            Ok(requete) => handle_request(requete).await,
            Err(erreur) => error_response(Value::Null, -32700, format!("JSON invalide : {erreur}")),
        };
        if reponse.is_null() {
            continue;
        }
        if let Ok(serialise) = serde_json::to_string(&reponse) {
            println!("{serialise}");
            let _ = std::io::stdout().flush();
        }
    }
}

async fn handle_request(requete: Value) -> Value {
    let id = requete.get("id").cloned().unwrap_or(Value::Null);
    let methode = requete
        .get("method")
        .and_then(Value::as_str)
        .unwrap_or_default();
    match methode {
        "initialize" => success(
            id,
            json!({
                "protocolVersion": "2025-06-18",
                "capabilities": { "tools": {} },
                "serverInfo": { "name": "morph-freetoken", "version": VERSION }
            }),
        ),
        "tools/list" => success(id, tools_list()),
        "tools/call" => {
            let params = requete.get("params").cloned().unwrap_or_else(|| json!({}));
            let nom = params
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or_default();
            let args = params
                .get("arguments")
                .cloned()
                .unwrap_or_else(|| json!({}));
            match call_tool(nom, args).await {
                Ok(valeur) => success(id, text_content(valeur)),
                Err(erreur) => error_response(id, -32000, erreur),
            }
        }
        notification if notification.starts_with("notifications/") => Value::Null,
        _ => error_response(id, -32601, format!("méthode MCP inconnue : {methode}")),
    }
}

fn tools_list() -> Value {
    json!({
        "tools": [
            {
                "name": "freetoken_status",
                "description": "Dit si le moteur peut tourner sur cette machine et ce qui manque : système, pilote NVIDIA, compilateur CUDA, version installée, cartes visibles, mémoire. À appeler en premier quand le moteur ne démarre pas.",
                "inputSchema": { "type": "object", "properties": {} }
            },
            {
                "name": "freetoken_install",
                "description": "Installe ou met à jour le moteur dans un environnement Python dédié à l'extension, à la version que cette extension a validée. Long : plusieurs minutes, et plusieurs gigaoctets téléchargés.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "version": {
                            "type": "string",
                            "description": "Version du paquet à installer. Par défaut celle que l'extension a validée."
                        }
                    }
                }
            },
            {
                "name": "freetoken_list_models",
                "description": "Liste les checkpoints que ce moteur sait servir, dans la bibliothèque de poids de l'utilisateur et dans le dossier de l'extension, en signalant ceux déjà convertis au format de chargement rapide.",
                "inputSchema": { "type": "object", "properties": {} }
            },
            {
                "name": "freetoken_server_health",
                "description": "Interroge le serveur en marche : est-il chargé, sur quel modèle, où en est le chargement. Répond aussi quels modèles il sert et sous quel nom les nommer dans une requête.",
                "inputSchema": { "type": "object", "properties": {} }
            },
            {
                "name": "freetoken_server_stats",
                "description": "Débit, latence, occupation de la VRAM et des files du serveur en marche. Utile pour expliquer une génération lente.",
                "inputSchema": { "type": "object", "properties": {} }
            },
            {
                "name": "freetoken_cache_status",
                "description": "L'état des réservoirs de cache du serveur : emplacements d'experts sur le GPU, capacité du cache KV, réutilisation de préfixe.",
                "inputSchema": { "type": "object", "properties": {} }
            },
            {
                "name": "freetoken_cache_resize",
                "description": "Redimensionne les réservoirs de cache sans redémarrer le serveur, donc sans recharger les poids. C'est ainsi qu'on rend de la VRAM à une autre tâche (une génération d'image, par exemple) tout en gardant le modèle chargé.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "moe": {
                            "type": "integer",
                            "description": "Nombre d'emplacements du cache d'experts sur le GPU."
                        },
                        "kv": {
                            "type": "integer",
                            "description": "Capacité du cache KV, en jetons."
                        }
                    }
                }
            },
            {
                "name": "freetoken_bench_bandwidth",
                "description": "Mesure la bande passante de la mémoire système contre celle du bus PCIe, et écrit le profil que le moteur relit pour choisir tout seul comment répartir les experts. À lancer une fois par machine. Long : plusieurs minutes.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "dtype": {
                            "type": "string",
                            "description": "Ne mesurer que ces formats d'experts, séparés par des virgules (ex. « nvfp4,bf16 »)."
                        }
                    }
                }
            },
            {
                "name": "freetoken_convert_checkpoint",
                "description": "Convertit un checkpoint Hugging Face au format de chargement rapide du moteur, dans le dossier de l'extension. Facultatif : le moteur charge aussi les checkpoints d'origine, mais plus lentement à chaque démarrage. Long, et double l'espace disque occupé.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "model": {
                            "type": "string",
                            "description": "Nom du checkpoint dans la bibliothèque de poids, chemin, ou identifiant de dépôt Hugging Face."
                        },
                        "name": {
                            "type": "string",
                            "description": "Nom du dossier de sortie. Par défaut, celui du checkpoint suivi de « -ftw »."
                        }
                    },
                    "required": ["model"]
                }
            },
            {
                "name": "freetoken_stop_server",
                "description": "Arrête le serveur et rend la VRAM. Le socle le relancera au prochain message si ce moteur reste le moteur actif.",
                "inputSchema": { "type": "object", "properties": {} }
            }
        ]
    })
}

async fn call_tool(nom: &str, args: Value) -> Result<Value, String> {
    let settings = ft::Settings::load();
    let runtime = ft::Runtime::detect(&settings);
    match nom {
        "freetoken_status" => {
            let sortie = ft::run(&ft::probe_invocation(&runtime), Duration::from_secs(180)).await?;
            let sonde = ft::parse_probe(&sortie.stdout);
            let sante = ft::ctl::health().await.ok();
            Ok(json!({
                "runtime": if runtime.is_wsl() { "wsl2" } else { "linux-natif" },
                "pret": sonde.ready(),
                "manquants": sonde.missing,
                "systeme": sonde.os,
                "pilote": sonde.driver_version,
                "cartes": sonde.gpus,
                "memoire_systeme_gio": sonde.ram_gb,
                "python": sonde.python,
                "uv": sonde.uv,
                "nvcc": sonde.nvcc,
                "version_moteur": sonde.engine_version.or(sonde.engine_on_path),
                "version_epinglee": ft::PINNED_VERSION,
                "serveur": sante,
                "endpoint": ft::ctl::base_url()
            }))
        }
        "freetoken_install" => {
            let version = args
                .get("version")
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|v| !v.is_empty())
                .unwrap_or(ft::PINNED_VERSION);
            let invocation = ft::install_invocation(&runtime, version);
            // Une installation qui compile des roues CUDA prend des dizaines de
            // minutes sur une machine modeste ; couper à deux minutes laisserait
            // un environnement à moitié posé.
            let sortie = ft::run(&invocation, Duration::from_secs(3 * 3600)).await?;
            if !sortie.success {
                return Err(format!(
                    "installation échouée (code {:?}).\n{}\n{}",
                    sortie.code,
                    sortie.stdout.trim(),
                    sortie.stderr.trim()
                ));
            }
            Ok(json!({
                "installe": true,
                "version_demandee": version,
                "version_rapportee": sortie.stdout.lines().last().unwrap_or("").trim(),
                "environnement": ft::venv_dir().display().to_string()
            }))
        }
        "freetoken_list_models" => {
            let modeles = ft::list_servable_models();
            Ok(json!({
                "modeles": modeles,
                "bibliotheque": ft::models_dir().map(|p| p.display().to_string()),
                "dossier_extension": ft::converted_dir().display().to_string(),
                "note": "Un identifiant de dépôt Hugging Face fonctionne aussi comme modèle : le moteur télécharge lui-même."
            }))
        }
        "freetoken_server_health" => {
            let sante = ft::ctl::health().await?;
            let modeles = ft::ctl::models().await.unwrap_or(Value::Null);
            Ok(json!({ "sante": sante, "modeles_servis": modeles }))
        }
        "freetoken_server_stats" => ft::ctl::stats().await,
        "freetoken_cache_status" => ft::ctl::cache_status().await,
        "freetoken_cache_resize" => {
            let demande = ft::ctl::CacheRebuild {
                moe: args.get("moe").and_then(Value::as_u64),
                kv: args.get("kv").and_then(Value::as_u64),
            };
            ft::ctl::cache_rebuild(&demande).await
        }
        "freetoken_bench_bandwidth" => {
            let dtype = args.get("dtype").and_then(Value::as_str);
            let sortie = ft::run(
                &ft::bench_invocation(&runtime, dtype),
                Duration::from_secs(3600),
            )
            .await?;
            if !sortie.success {
                return Err(format!(
                    "mesure échouée (code {:?}).\n{}\n{}",
                    sortie.code,
                    sortie.stdout.trim(),
                    sortie.stderr.trim()
                ));
            }
            Ok(json!({
                "mesure": true,
                "sortie": sortie.stdout.trim(),
                "note": "Le profil est relu au prochain démarrage : avec « Répartition des experts » sur « auto », le moteur suit cette mesure."
            }))
        }
        "freetoken_convert_checkpoint" => {
            let model = args
                .get("model")
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|m| !m.is_empty())
                .ok_or("« model » est requis")?;
            let defaut = format!(
                "{}-ftw",
                model.rsplit(['/', '\\']).next().unwrap_or("checkpoint")
            );
            let nom = args
                .get("name")
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|n| !n.is_empty())
                .unwrap_or(&defaut);
            // Le nom vient d'un appel d'outil : le confiner au dossier de
            // l'extension évite qu'un « ../.. » fasse écrire ailleurs.
            if nom.contains('/') || nom.contains('\\') || nom.contains("..") {
                return Err("« name » doit être un simple nom de dossier".into());
            }
            let sortie_dir = ft::converted_dir().join(nom);
            std::fs::create_dir_all(&sortie_dir).map_err(|e| e.to_string())?;
            let invocation = ft::convert_invocation(&runtime, model, &sortie_dir.to_string_lossy());
            let sortie = ft::run(&invocation, Duration::from_secs(6 * 3600)).await?;
            if !sortie.success {
                return Err(format!(
                    "conversion échouée (code {:?}).\n{}\n{}",
                    sortie.code,
                    sortie.stdout.trim(),
                    sortie.stderr.trim()
                ));
            }
            Ok(json!({
                "converti": true,
                "source": model,
                "dossier": sortie_dir.display().to_string(),
                "note": "Ce dossier apparaît maintenant comme modèle sélectionnable dans Réglages → Moteur."
            }))
        }
        "freetoken_stop_server" => {
            let port = std::env::var("FREETOKEN_LOCARYN_ENDPOINT")
                .ok()
                .and_then(|u| u.rsplit(':').next().map(str::to_string))
                .and_then(|p| p.trim_end_matches('/').parse::<u16>().ok())
                .unwrap_or(ft::DEFAULT_PORT);
            let sortie = ft::run(
                &ft::kill_stale_invocation(&runtime, port),
                Duration::from_secs(60),
            )
            .await?;
            Ok(json!({
                "arrete": true,
                "port": port,
                "commande": sortie.command
            }))
        }
        autre => Err(format!("outil inconnu : {autre}")),
    }
}

// ============================================================================
// Enveloppes JSON-RPC
// ============================================================================

fn text_content(valeur: Value) -> Value {
    let texte = match &valeur {
        Value::String(s) => s.clone(),
        autre => serde_json::to_string_pretty(autre).unwrap_or_else(|_| autre.to_string()),
    };
    json!({ "content": [{ "type": "text", "text": texte }] })
}

fn success(id: Value, resultat: Value) -> Value {
    json!({ "jsonrpc": "2.0", "id": id, "result": resultat })
}

fn error_response(id: Value, code: i32, message: impl Into<String>) -> Value {
    json!({
        "jsonrpc": "2.0",
        "id": id,
        "error": { "code": code, "message": message.into() }
    })
}
