//! Extension FreeToken — le moteur d'inférence Mixture-of-Experts, piloté
//! depuis Locaryn.
//!
//! FreeToken tourne sous **Linux x86_64**, avec un GPU NVIDIA, le pilote r580+
//! et CUDA 13 pour la compilation à la volée de ses noyaux. Sous Windows, il
//! n'existe pas de version native : ce module passe donc par WSL2, traduit les
//! chemins de la machine hôte vers ceux de la distribution, et dit précisément
//! ce qui manque quand ça ne marche pas.
//!
//! Tout ce qui est propre au moteur vit ici, dans l'extension. Le socle ne sait
//! que lancer une liste d'arguments et sonder une URL : c'est ce qui lui permet
//! d'accueillir un autre moteur demain sans que rien de FreeToken n'y soit
//! écrit.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::Duration;
use tokio::process::Command;

pub mod ctl;

/// Le paquet Python du moteur, et la version que cette extension a validée.
pub const PACKAGE: &str = "freetoken";
pub const PINNED_VERSION: &str = "0.1.2";
/// Ce que FreeToken installe comme point d'entrée en ligne de commande.
pub const CLI: &str = "ft";
/// Port d'écoute par défaut du serveur, celui du manifeste.
pub const DEFAULT_PORT: u16 = 1919;

// ============================================================================
// Réglages
// ============================================================================

/// Les réglages de l'extension, tels que l'écran des réglages les enregistre.
///
/// L'hôte donne le chemin du fichier dans `LOCARYN_EXTENSION_CONFIG_FILE`. Les
/// champs numériques y arrivent en texte — c'est ce que le formulaire écrit —
/// d'où la lecture tolérante plutôt qu'un typage qui échouerait au premier
/// réglage modifié.
#[derive(Debug, Clone, Default)]
pub struct Settings {
    pub moe_backend: Option<String>,
    pub memory_ratio: Option<f32>,
    pub max_output_tokens: Option<u32>,
    pub gpu: Option<String>,
    pub wsl_distro: Option<String>,
    pub extra_args: Vec<String>,
}

impl Settings {
    /// Lit les réglages. Un fichier absent ou illisible donne les valeurs par
    /// défaut : l'absence de réglages n'est pas une erreur, c'est le premier
    /// lancement.
    pub fn load() -> Self {
        let Some(path) = std::env::var_os("LOCARYN_EXTENSION_CONFIG_FILE").map(PathBuf::from)
        else {
            return Self::default();
        };
        let Ok(text) = std::fs::read_to_string(&path) else {
            return Self::default();
        };
        let Ok(value) = serde_json::from_str::<serde_json::Value>(&text) else {
            eprintln!(
                "[freetoken] réglages illisibles ({}) — valeurs par défaut",
                path.display()
            );
            return Self::default();
        };
        let texte = |cle: &str| -> Option<String> {
            value
                .get(cle)
                .map(|v| match v {
                    serde_json::Value::String(s) => s.clone(),
                    autre => autre.to_string(),
                })
                .map(|s| s.trim().trim_matches('"').to_string())
                .filter(|s| !s.is_empty())
        };
        Self {
            moe_backend: texte("moe_backend").filter(|b| b != "auto"),
            memory_ratio: texte("memory_ratio").and_then(|s| s.replace(',', ".").parse().ok()),
            max_output_tokens: texte("max_output_tokens").and_then(|s| s.parse().ok()),
            gpu: texte("gpu"),
            wsl_distro: texte("wsl_distro"),
            extra_args: texte("extra_args")
                .map(|s| s.split_whitespace().map(str::to_string).collect())
                .unwrap_or_default(),
        }
    }
}

// ============================================================================
// Où tourne le moteur
// ============================================================================

/// Comment atteindre le moteur sur cette machine.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Runtime {
    /// Linux : le moteur tourne directement, depuis l'environnement Python
    /// géré par l'extension ou depuis le chemin du système.
    Native,
    /// Windows : le moteur tourne dans une distribution WSL2, atteinte par
    /// `wsl.exe`. `None` = la distribution par défaut.
    Wsl { distro: Option<String> },
}

impl Runtime {
    /// Ce que cette machine impose. Sous Windows, seul WSL peut faire tourner
    /// le moteur ; ailleurs on suppose Linux, et l'appelant vérifie le reste.
    pub fn detect(settings: &Settings) -> Self {
        if cfg!(windows) {
            Runtime::Wsl {
                distro: settings.wsl_distro.clone(),
            }
        } else {
            Runtime::Native
        }
    }

    pub fn is_wsl(&self) -> bool {
        matches!(self, Runtime::Wsl { .. })
    }
}

/// Le dossier privé de l'extension, donné par l'hôte.
pub fn extension_data_dir() -> PathBuf {
    std::env::var_os("LOCARYN_EXTENSION_DATA_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| std::env::temp_dir().join("locaryn-freetoken"))
}

/// La bibliothèque de poids de l'utilisateur, donnée par l'hôte.
///
/// Sans elle, l'extension ne verrait que son dossier privé — vide au premier
/// lancement — et annoncerait qu'aucun modèle n'est installé alors que tout est
/// déjà téléchargé.
pub fn models_dir() -> Option<PathBuf> {
    std::env::var_os("LOCARYN_MODELS_DIR").map(PathBuf::from)
}

/// L'environnement Python que l'extension gère pour le moteur.
///
/// Il vit dans le dossier privé de l'extension, pas dans celui de
/// l'application : le paquet et ses noyaux CUDA pèsent plusieurs gigaoctets, et
/// une extension retirée doit pouvoir emporter les siens.
pub fn venv_dir() -> PathBuf {
    extension_data_dir().join("venv")
}

/// Le `ft` de l'environnement géré, s'il est là.
fn venv_cli(runtime: &Runtime) -> Option<PathBuf> {
    let candidat = match runtime {
        // Sous WSL, l'environnement est créé dans le dossier privé vu depuis
        // Linux ; le test d'existence se fait donc à l'intérieur, pas ici.
        Runtime::Wsl { .. } => return None,
        Runtime::Native => venv_dir().join("bin").join(CLI),
    };
    candidat.is_file().then_some(candidat)
}

// ============================================================================
// Traduction des chemins
// ============================================================================

/// Traduit un chemin Windows vers son équivalent dans WSL.
///
/// `D:\Documents\modeles` devient `/mnt/d/Documents/modeles`. Un chemin déjà
/// POSIX est rendu tel quel : les poids peuvent très bien vivre dans la
/// distribution elle-même, et le réécrire les rendrait introuvables.
pub fn to_wsl_path(path: &str) -> String {
    let p = path.trim();
    if p.starts_with('/') {
        return p.to_string();
    }
    let octets = p.as_bytes();
    if octets.len() >= 2 && octets[1] == b':' && (octets[0] as char).is_ascii_alphabetic() {
        let lettre = (octets[0] as char).to_ascii_lowercase();
        let reste = p[2..].replace('\\', "/");
        let reste = reste.trim_start_matches('/');
        return format!("/mnt/{lettre}/{reste}");
    }
    p.replace('\\', "/")
}

/// Le chemin d'un modèle, exprimé pour le moteur.
///
/// Un identifiant de dépôt Hugging Face (`Qwen/Qwen3.6-35B-A3B`) est laissé
/// intact : `ft serve --model` le reconnaît et télécharge lui-même. Un chemin
/// est traduit quand le moteur tourne dans WSL.
pub fn model_for_engine(model: &str, runtime: &Runtime) -> String {
    let m = model.trim();
    if m.is_empty() {
        return String::new();
    }
    if looks_like_hf_repo_id(m) {
        return m.to_string();
    }
    match runtime {
        Runtime::Wsl { .. } => to_wsl_path(m),
        Runtime::Native => m.to_string(),
    }
}

/// `propriétaire/nom`, sans schéma d'URL ni séparateur de chemin Windows.
pub fn looks_like_hf_repo_id(value: &str) -> bool {
    if value.contains("://") || value.contains('\\') || value.contains(' ') {
        return false;
    }
    if value.starts_with('/') || value.starts_with('.') {
        return false;
    }
    let octets = value.as_bytes();
    if octets.len() > 2 && octets[1] == b':' {
        return false;
    }
    let mut parts = value.split('/');
    let (owner, name) = (parts.next().unwrap_or(""), parts.next().unwrap_or(""));
    parts.next().is_none()
        && !owner.is_empty()
        && !name.is_empty()
        && owner
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'))
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'))
}

// ============================================================================
// Construction des commandes
// ============================================================================

/// Une commande prête à lancer : le programme et ses arguments, jamais une
/// ligne de shell interprétée par l'hôte.
#[derive(Debug, Clone)]
pub struct Invocation {
    pub program: String,
    pub args: Vec<String>,
}

impl Invocation {
    /// La commande, telle qu'on l'écrirait à la main — pour les journaux et les
    /// messages d'erreur. Un moteur qui ne démarre pas se diagnostique en
    /// relançant sa commande, encore faut-il la connaître.
    pub fn display(&self) -> String {
        let mut out = String::from(&self.program);
        for a in &self.args {
            out.push(' ');
            if a.contains(' ') {
                out.push('"');
                out.push_str(a);
                out.push('"');
            } else {
                out.push_str(a);
            }
        }
        out
    }

    pub fn to_command(&self) -> Command {
        let mut cmd = Command::new(&self.program);
        cmd.args(&self.args);
        cmd
    }
}

/// Enveloppe une ligne de commande Linux dans l'invocation qui l'atteint.
///
/// Sous Linux, c'est la commande elle-même. Sous Windows, `wsl.exe` la fait
/// exécuter par un shell de connexion dans la distribution : c'est ce shell qui
/// charge le `PATH` où `uv`, `python` et `nvcc` ont été installés — sans lui, la
/// commande échoue en annonçant que rien n'est installé.
pub fn wrap(runtime: &Runtime, shell_line: &str) -> Invocation {
    match runtime {
        Runtime::Native => Invocation {
            program: "/bin/sh".to_string(),
            args: vec!["-lc".to_string(), shell_line.to_string()],
        },
        Runtime::Wsl { distro } => {
            let mut args: Vec<String> = Vec::new();
            if let Some(d) = distro.as_ref().filter(|d| !d.trim().is_empty()) {
                args.push("-d".to_string());
                args.push(d.clone());
            }
            args.push("--".to_string());
            args.push("bash".to_string());
            args.push("-lc".to_string());
            args.push(shell_line.to_string());
            Invocation {
                program: "wsl.exe".to_string(),
                args,
            }
        }
    }
}

/// Le préfixe qui active l'environnement Python géré, quand il existe.
///
/// Le moteur peut aussi avoir été installé par l'utilisateur lui-même (`uv pip
/// install freetoken` dans son propre environnement) : dans ce cas le préfixe
/// est vide et `ft` est cherché sur le chemin.
fn activate_prefix(runtime: &Runtime) -> String {
    match runtime {
        Runtime::Native => match venv_cli(runtime) {
            Some(_) => format!(
                ". {}/bin/activate 2>/dev/null || true; ",
                shell_quote(&venv_dir().to_string_lossy())
            ),
            None => String::new(),
        },
        Runtime::Wsl { .. } => {
            // Le test d'existence a lieu dans la distribution, pas ici : un
            // chemin Windows ne dit rien de ce qui existe côté Linux.
            let venv = to_wsl_path(&venv_dir().to_string_lossy());
            format!(
                "if [ -f {v}/bin/activate ]; then . {v}/bin/activate; fi; ",
                v = shell_quote(&venv)
            )
        }
    }
}

/// Protège une valeur pour un shell POSIX.
pub fn shell_quote(value: &str) -> String {
    format!("'{}'", value.replace('\'', "'\\''"))
}

/// La ligne de commande de `ft serve`, réglages compris.
pub fn serve_line(settings: &Settings, runtime: &Runtime, port: u16, model: &str) -> String {
    let mut ligne = format!(
        "{}{} serve --model {} --host 127.0.0.1 --port {port}",
        activate_prefix(runtime),
        CLI,
        shell_quote(&model_for_engine(model, runtime))
    );
    if let Some(backend) = &settings.moe_backend {
        ligne.push_str(&format!(" --moe-backend {}", shell_quote(backend)));
    }
    if let Some(ratio) = settings.memory_ratio.filter(|r| (0.1..=0.95).contains(r)) {
        ligne.push_str(&format!(" --memory-ratio {ratio}"));
    }
    if let Some(max) = settings.max_output_tokens.filter(|m| *m > 0) {
        ligne.push_str(&format!(" --max-output-tokens {max}"));
    }
    if let Some(gpu) = &settings.gpu {
        ligne.push_str(&format!(" --gpu {}", shell_quote(gpu)));
    }
    for extra in &settings.extra_args {
        ligne.push(' ');
        ligne.push_str(&shell_quote(extra));
    }
    ligne
}

/// La commande complète pour lancer le serveur.
pub fn serve_invocation(
    settings: &Settings,
    runtime: &Runtime,
    port: u16,
    model: &str,
) -> Invocation {
    // `exec` remplace le shell par le serveur : un maillon de moins entre le
    // processus que l'hôte surveille et celui qui écoute vraiment, donc un
    // arrêt qui atteint sa cible.
    wrap(
        runtime,
        &format!("exec {}", serve_line(settings, runtime, port, model)),
    )
}

/// La commande qui installe ou met à jour le moteur dans l'environnement géré.
///
/// `uv` s'il est là — c'est ce que recommande le projet et c'est bien plus
/// rapide —, `python -m venv` + `pip` sinon. La version est épinglée : une
/// chaîne d'approvisionnement sans version installe autre chose à chaque fois.
pub fn install_invocation(runtime: &Runtime, version: &str) -> Invocation {
    let venv = match runtime {
        Runtime::Wsl { .. } => to_wsl_path(&venv_dir().to_string_lossy()),
        Runtime::Native => venv_dir().to_string_lossy().into_owned(),
    };
    let venv = shell_quote(&venv);
    let cible = shell_quote(&format!("{PACKAGE}[accel]=={version}"));
    let ligne = format!(
        "set -e; \
         if command -v uv >/dev/null 2>&1; then \
           uv venv {venv} >/dev/null; \
           VIRTUAL_ENV={venv} uv pip install --python {venv}/bin/python {cible}; \
         else \
           python3 -m venv {venv}; \
           {venv}/bin/python -m pip install --upgrade pip >/dev/null; \
           {venv}/bin/python -m pip install {cible}; \
         fi; \
         {venv}/bin/{CLI} --version"
    );
    wrap(runtime, &ligne)
}

/// La commande qui interroge la machine : pilote, cartes, outils présents.
pub fn probe_invocation(runtime: &Runtime) -> Invocation {
    let venv = match runtime {
        Runtime::Wsl { .. } => to_wsl_path(&venv_dir().to_string_lossy()),
        Runtime::Native => venv_dir().to_string_lossy().into_owned(),
    };
    let venv = shell_quote(&venv);
    // Chaque ligne est préfixée d'une clé : la sortie est lue par
    // `parse_probe`, pas par un humain, et une sortie sans clés serait un
    // format à devineriez à chaque version d'outil.
    let ligne = format!(
        "echo \"os=$(uname -s) $(uname -m)\"; \
         echo \"python=$(command -v python3 || echo -)\"; \
         echo \"uv=$(command -v uv || echo -)\"; \
         echo \"nvcc=$(command -v nvcc || echo -)\"; \
         echo \"venv={venv}\"; \
         if [ -x {venv}/bin/{CLI} ]; then echo \"ft_venv=$({venv}/bin/{CLI} --version 2>&1 | head -n1)\"; else echo 'ft_venv=-'; fi; \
         if command -v {CLI} >/dev/null 2>&1; then echo \"ft_path=$({CLI} --version 2>&1 | head -n1)\"; else echo 'ft_path=-'; fi; \
         if command -v nvidia-smi >/dev/null 2>&1; then \
           echo \"driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1)\"; \
           nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | while IFS= read -r l; do echo \"gpu=$l\"; done; \
         else echo 'driver=-'; fi; \
         echo \"ram_kb=$(awk '/MemTotal/ {{print $2}}' /proc/meminfo 2>/dev/null || echo 0)\""
    );
    wrap(runtime, &ligne)
}

/// La commande qui arrête un serveur resté en place sur ce port.
///
/// Sous WSL, l'arrêt du processus `wsl.exe` que surveille l'hôte ne garantit
/// pas la mort du serveur Linux. Sans ce nettoyage, changer de modèle laissait
/// l'ancien serveur occuper le port et la VRAM, et le nouveau échouait au
/// démarrage sans dire pourquoi.
pub fn kill_stale_invocation(runtime: &Runtime, port: u16) -> Invocation {
    let motif = shell_quote(&format!("{CLI} serve .*--port {port}"));
    wrap(
        runtime,
        &format!("pkill -f {motif} 2>/dev/null; pkill -f {motif} 2>/dev/null; exit 0"),
    )
}

/// La commande qui mesure la bande passante mémoire contre PCIe.
pub fn bench_invocation(runtime: &Runtime, dtype: Option<&str>) -> Invocation {
    let mut ligne = format!("{}{} bench bw", activate_prefix(runtime), CLI);
    if let Some(d) = dtype.filter(|d| !d.trim().is_empty()) {
        ligne.push_str(&format!(" --dtype {}", shell_quote(d)));
    }
    wrap(runtime, &ligne)
}

/// La commande qui convertit un checkpoint Hugging Face au format de
/// chargement rapide du moteur.
pub fn convert_invocation(runtime: &Runtime, source: &str, out: &str) -> Invocation {
    let ligne = format!(
        "{}{} checkpoint --model {} --out {}",
        activate_prefix(runtime),
        CLI,
        shell_quote(&model_for_engine(source, runtime)),
        shell_quote(&match runtime {
            Runtime::Wsl { .. } => to_wsl_path(out),
            Runtime::Native => out.to_string(),
        })
    );
    wrap(runtime, &ligne)
}

// ============================================================================
// Exécution
// ============================================================================

/// Ce qu'a donné une commande.
#[derive(Debug, Clone, Serialize)]
pub struct Outcome {
    pub command: String,
    pub success: bool,
    pub code: Option<i32>,
    pub stdout: String,
    pub stderr: String,
}

/// Lance une commande et attend sa fin.
///
/// `timeout` protège contre un `wsl.exe` qui ne rend jamais la main — cela
/// arrive quand la distribution démarre pour la première fois — plutôt que de
/// laisser un outil MCP suspendu sans explication.
pub async fn run(invocation: &Invocation, timeout: Duration) -> Result<Outcome, String> {
    let mut cmd = invocation.to_command();
    cmd.stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    hide_console(&mut cmd);
    let child = cmd
        .spawn()
        .map_err(|e| format!("{} : {e}", invocation.program))?;
    let sortie = tokio::time::timeout(timeout, child.wait_with_output())
        .await
        .map_err(|_| {
            format!(
                "délai dépassé ({} s) : {}",
                timeout.as_secs(),
                invocation.display()
            )
        })?
        .map_err(|e| e.to_string())?;
    Ok(Outcome {
        command: invocation.display(),
        success: sortie.status.success(),
        code: sortie.status.code(),
        stdout: String::from_utf8_lossy(&sortie.stdout).to_string(),
        stderr: String::from_utf8_lossy(&sortie.stderr).to_string(),
    })
}

/// Pas de fenêtre de console qui clignote sous Windows, et un groupe de
/// processus à tuer d'un bloc.
pub fn hide_console(cmd: &mut Command) {
    #[cfg(windows)]
    {
        // `tokio::process::Command` porte sa propre `creation_flags` sous
        // Windows : le trait de la bibliothèque standard n'a pas à être importé.
        cmd.creation_flags(0x0800_0008);
    }
    #[cfg(not(windows))]
    {
        let _ = cmd;
    }
}

// ============================================================================
// État de la machine
// ============================================================================

/// Ce que la machine offre, et ce qui manque.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Probe {
    /// `Linux x86_64` tel que rapporté par la distribution qui exécute.
    pub os: Option<String>,
    pub python: Option<String>,
    pub uv: Option<String>,
    /// Le compilateur CUDA, exigé pour la compilation à la volée des noyaux.
    pub nvcc: Option<String>,
    /// La version du moteur installée dans l'environnement géré.
    pub engine_version: Option<String>,
    /// La version du moteur trouvée sur le chemin du système, si l'utilisateur
    /// l'a installée lui-même.
    pub engine_on_path: Option<String>,
    pub driver_version: Option<String>,
    pub gpus: Vec<String>,
    pub ram_gb: Option<f32>,
    /// Ce qui empêche le moteur de tourner, en phrases lisibles. Vide : rien
    /// ne manque.
    pub missing: Vec<String>,
}

impl Probe {
    pub fn ready(&self) -> bool {
        self.missing.is_empty()
    }
}

/// Lit la sortie clé=valeur de [`probe_invocation`].
pub fn parse_probe(stdout: &str) -> Probe {
    let mut valeurs: HashMap<&str, String> = HashMap::new();
    let mut gpus = Vec::new();
    for ligne in stdout.lines() {
        let Some((cle, valeur)) = ligne.split_once('=') else {
            continue;
        };
        let valeur = valeur.trim();
        if valeur == "-" || valeur.is_empty() {
            continue;
        }
        if cle.trim() == "gpu" {
            gpus.push(valeur.to_string());
        } else {
            valeurs.insert(cle.trim(), valeur.to_string());
        }
    }
    let mut probe = Probe {
        os: valeurs.get("os").cloned(),
        python: valeurs.get("python").cloned(),
        uv: valeurs.get("uv").cloned(),
        nvcc: valeurs.get("nvcc").cloned(),
        engine_version: valeurs.get("ft_venv").cloned(),
        engine_on_path: valeurs.get("ft_path").cloned(),
        driver_version: valeurs.get("driver").cloned(),
        gpus,
        ram_gb: valeurs
            .get("ram_kb")
            .and_then(|v| v.parse::<f64>().ok())
            .filter(|kb| *kb > 0.0)
            .map(|kb| (kb / 1024.0 / 1024.0) as f32),
        missing: Vec::new(),
    };
    probe.missing = manquants(&probe);
    probe
}

/// Ce qui manque, dit en une phrase par manque et avec la commande qui le
/// répare. Un « échec de démarrage » sans cause n'apprend rien.
fn manquants(p: &Probe) -> Vec<String> {
    let mut out = Vec::new();
    if p.os.is_none() {
        out.push(
            "Aucun système Linux joignable. Sous Windows, installez WSL2 (« wsl --install ») \
             puis une distribution."
                .to_string(),
        );
        return out;
    }
    if p.os.as_deref().is_some_and(|os| !os.contains("Linux")) {
        out.push(format!(
            "Le moteur exige Linux x86_64 ; la distribution rapporte « {} ».",
            p.os.clone().unwrap_or_default()
        ));
    }
    if p.python.is_none() {
        out.push("Python 3 absent de la distribution (« apt install python3-venv »).".to_string());
    }
    if p.driver_version.is_none() || p.gpus.is_empty() {
        out.push(
            "Aucun GPU NVIDIA visible (« nvidia-smi » ne répond pas). Sous WSL2, le pilote \
             s'installe côté Windows, pas dans la distribution."
                .to_string(),
        );
    }
    if p.nvcc.is_none() {
        out.push(
            "Le compilateur CUDA « nvcc » est absent : le moteur compile ses noyaux au premier \
             usage et en a besoin. Installez le toolkit CUDA 13."
                .to_string(),
        );
    }
    if p.engine_version.is_none() && p.engine_on_path.is_none() {
        out.push(format!(
            "Le moteur n'est pas installé. Lancez l'outil « freetoken_install » (il pose \
             {PACKAGE}[accel]=={PINNED_VERSION} dans un environnement dédié à l'extension)."
        ));
    }
    out
}

/// Le dossier où l'extension range les checkpoints qu'elle convertit.
pub fn converted_dir() -> PathBuf {
    extension_data_dir().join("models")
}

/// Les checkpoints que ce moteur sait servir, dans la bibliothèque de poids de
/// l'utilisateur et dans le dossier de l'extension.
///
/// Un répertoire compte quand il porte un `config.json` (checkpoint
/// Transformers) ou un `freetoken_weight.json` (déjà converti).
pub fn list_servable_models() -> Vec<ServableModel> {
    let mut out = Vec::new();
    for racine in models_dir().into_iter().chain(Some(converted_dir())) {
        let Ok(entrees) = std::fs::read_dir(&racine) else {
            continue;
        };
        for entree in entrees.flatten() {
            let chemin = entree.path();
            if !chemin.is_dir() {
                continue;
            }
            let converti = chemin.join("freetoken_weight.json").exists();
            if !converti && !chemin.join("config.json").exists() {
                continue;
            }
            let Some(nom) = chemin.file_name().and_then(|n| n.to_str()) else {
                continue;
            };
            out.push(ServableModel {
                name: nom.to_string(),
                path: chemin.to_string_lossy().into_owned(),
                converted: converti,
                size_gb: taille_gb(&chemin),
            });
        }
    }
    out.sort_by(|a, b| a.name.to_lowercase().cmp(&b.name.to_lowercase()));
    out.dedup_by(|a, b| a.path == b.path);
    out
}

/// Un checkpoint servable.
#[derive(Debug, Clone, Serialize)]
pub struct ServableModel {
    pub name: String,
    pub path: String,
    /// Déjà au format de chargement rapide du moteur.
    pub converted: bool,
    pub size_gb: f32,
}

fn taille_gb(dir: &Path) -> f32 {
    fn cumul(dir: &Path, reste: u32) -> u64 {
        if reste == 0 {
            return 0;
        }
        let Ok(entrees) = std::fs::read_dir(dir) else {
            return 0;
        };
        entrees
            .flatten()
            .map(|e| {
                let p = e.path();
                if p.is_dir() {
                    cumul(&p, reste - 1)
                } else {
                    p.metadata().map(|m| m.len()).unwrap_or(0)
                }
            })
            .sum()
    }
    cumul(dir, 3) as f32 / 1_073_741_824.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn un_chemin_windows_devient_un_chemin_wsl() {
        assert_eq!(
            to_wsl_path(r"D:\Documents\modeles"),
            "/mnt/d/Documents/modeles"
        );
        assert_eq!(to_wsl_path(r"C:\a b\c"), "/mnt/c/a b/c");
        // Un chemin déjà POSIX ne bouge pas : les poids peuvent vivre dans la
        // distribution, et le réécrire les rendrait introuvables.
        assert_eq!(to_wsl_path("/home/moi/modeles"), "/home/moi/modeles");
    }

    #[test]
    fn un_depot_hugging_face_traverse_intact() {
        let wsl = Runtime::Wsl { distro: None };
        assert_eq!(
            model_for_engine("Qwen/Qwen3.6-35B-A3B", &wsl),
            "Qwen/Qwen3.6-35B-A3B"
        );
        assert_eq!(
            model_for_engine(r"D:\modeles\Qwen__Qwen3.6-35B-A3B", &wsl),
            "/mnt/d/modeles/Qwen__Qwen3.6-35B-A3B"
        );
    }

    #[test]
    fn ce_qui_ressemble_a_un_depot_et_ce_qui_n_y_ressemble_pas() {
        assert!(looks_like_hf_repo_id("nvidia/GLM-5.2-NVFP4"));
        assert!(!looks_like_hf_repo_id("/mnt/d/modeles/glm"));
        assert!(!looks_like_hf_repo_id(r"D:\modeles\glm"));
        assert!(!looks_like_hf_repo_id(
            "https://huggingface.co/nvidia/GLM-5.2-NVFP4"
        ));
        assert!(!looks_like_hf_repo_id("glm"));
        assert!(!looks_like_hf_repo_id("a/b/c"));
    }

    #[test]
    fn les_reglages_arrivent_dans_la_ligne_de_commande() {
        let settings = Settings {
            moe_backend: Some("hybrid".into()),
            memory_ratio: Some(0.8),
            max_output_tokens: Some(4096),
            gpu: Some("1".into()),
            wsl_distro: None,
            extra_args: vec!["--enable-cache-report".into()],
        };
        let ligne = serve_line(&settings, &Runtime::Native, 1919, "Qwen/Qwen3.6-35B-A3B");
        assert!(ligne.contains("--moe-backend 'hybrid'"));
        assert!(ligne.contains("--memory-ratio 0.8"));
        assert!(ligne.contains("--max-output-tokens 4096"));
        assert!(ligne.contains("--gpu '1'"));
        assert!(ligne.contains("--port 1919"));
        assert!(ligne.contains("'--enable-cache-report'"));
    }

    /// `auto` est le défaut du moteur : le passer explicitement reviendrait à
    /// figer un choix que le moteur fait mieux que nous à partir du profil
    /// mesuré.
    #[test]
    fn le_choix_automatique_ne_pose_aucun_drapeau() {
        let settings = Settings {
            moe_backend: None,
            ..Default::default()
        };
        let ligne = serve_line(&settings, &Runtime::Native, 1919, "un/modele");
        assert!(!ligne.contains("--moe-backend"));
    }

    /// Une part de VRAM absurde est ignorée plutôt que transmise : le moteur
    /// refuserait de démarrer, et la cause serait invisible.
    #[test]
    fn une_part_de_vram_hors_bornes_est_ignoree() {
        let settings = Settings {
            memory_ratio: Some(1.8),
            ..Default::default()
        };
        let ligne = serve_line(&settings, &Runtime::Native, 1919, "un/modele");
        assert!(!ligne.contains("--memory-ratio"));
    }

    #[test]
    fn sous_windows_la_commande_passe_par_wsl() {
        let inv = wrap(
            &Runtime::Wsl {
                distro: Some("Ubuntu".into()),
            },
            "echo bonjour",
        );
        assert_eq!(inv.program, "wsl.exe");
        assert_eq!(inv.args[0], "-d");
        assert_eq!(inv.args[1], "Ubuntu");
        assert_eq!(inv.args[2], "--");
        assert_eq!(inv.args[3], "bash");
    }

    #[test]
    fn une_apostrophe_ne_casse_pas_la_ligne_de_commande() {
        assert_eq!(shell_quote("d'accord"), "'d'\\''accord'");
    }

    #[test]
    fn la_sonde_se_lit_en_cles_valeurs() {
        let p = parse_probe(
            "os=Linux x86_64\n\
             python=/usr/bin/python3\n\
             uv=-\n\
             nvcc=/usr/local/cuda/bin/nvcc\n\
             ft_venv=ft 0.1.2\n\
             ft_path=-\n\
             driver=580.65.06\n\
             gpu=0, NVIDIA GeForce RTX 4050 Laptop GPU, 6141 MiB\n\
             ram_kb=32000000\n",
        );
        assert_eq!(p.os.as_deref(), Some("Linux x86_64"));
        assert_eq!(p.engine_version.as_deref(), Some("ft 0.1.2"));
        assert_eq!(p.engine_on_path, None);
        assert_eq!(p.gpus.len(), 1);
        assert!(p.ram_gb.is_some_and(|r| r > 29.0 && r < 32.0));
        assert!(p.ready(), "rien ne manque : {:?}", p.missing);
    }

    /// Ce qui manque est nommé, avec la commande qui le répare — pas un
    /// « moteur indisponible » que personne ne sait corriger.
    #[test]
    fn ce_qui_manque_est_nomme() {
        let p = parse_probe("os=Linux x86_64\npython=/usr/bin/python3\nram_kb=16000000\n");
        assert!(!p.ready());
        assert!(p.missing.iter().any(|m| m.contains("nvidia-smi")));
        assert!(p.missing.iter().any(|m| m.contains("nvcc")));
        assert!(p.missing.iter().any(|m| m.contains("freetoken_install")));
    }

    #[test]
    fn sans_systeme_la_sonde_dit_d_installer_wsl() {
        let p = parse_probe("");
        assert_eq!(p.missing.len(), 1);
        assert!(p.missing[0].contains("WSL2"));
    }

    #[test]
    fn l_installation_epingle_la_version() {
        let inv = install_invocation(&Runtime::Native, "0.1.2");
        assert!(inv.args.last().unwrap().contains("freetoken[accel]==0.1.2"));
    }

    #[test]
    fn le_nettoyage_cible_le_port_et_ne_signale_pas_d_erreur() {
        let inv = kill_stale_invocation(&Runtime::Native, 1919);
        let ligne = inv.args.last().unwrap();
        assert!(ligne.contains("--port 1919"));
        assert!(ligne.contains("exit 0"));
    }
}
