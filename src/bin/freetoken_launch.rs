//! Lanceur du serveur FreeToken — le programme que le socle nomme dans
//! `engine.lifecycle.start`.
//!
//! Le socle sait lancer une liste d'arguments et sonder une URL ; il ne sait
//! pas que ce moteur n'existe que sous Linux, qu'il faut passer par WSL2 sous
//! Windows, ni qu'un chemin `D:\…` doit devenir `/mnt/d/…`. Tout cela vit ici.
//!
//! Le lanceur remplace son propre processus par le serveur là où c'est
//! possible, pour qu'il reste un seul maillon entre ce que le socle surveille
//! et ce qui écoute vraiment. Sous Windows, `wsl.exe` est ce maillon, et le
//! lanceur nettoie d'abord tout serveur resté sur le même port : sinon un
//! changement de modèle laisse l'ancien occuper le port et la VRAM, et le
//! nouveau échoue sans dire pourquoi.

use locaryn_plugin_freetoken as ft;
use std::process::ExitCode;
use std::time::Duration;

#[tokio::main]
async fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("serve") => serve(&args[1..]).await,
        Some("stop") => stop(&args[1..]).await,
        Some("probe") => probe().await,
        Some("--help") | Some("-h") | None => {
            eprintln!(
                "locaryn-freetoken-launch — lance le serveur FreeToken pour Locaryn\n\
                 \n\
                   serve --port <n> --model <chemin|depot>   démarre le serveur\n\
                   stop  --port <n>                          arrête un serveur resté en place\n\
                   probe                                     dit ce qui manque sur cette machine\n"
            );
            ExitCode::SUCCESS
        }
        Some(autre) => {
            eprintln!("[freetoken] commande inconnue : {autre}");
            ExitCode::FAILURE
        }
    }
}

/// Lit `--clé valeur` sans dépendance : le lanceur ne reçoit que ce que le
/// manifeste lui passe, et une bibliothèque d'analyse d'arguments serait plus
/// grosse que le besoin.
fn flag(args: &[String], nom: &str) -> Option<String> {
    let mut it = args.iter();
    while let Some(a) = it.next() {
        if a == nom {
            return it.next().cloned();
        }
        if let Some(reste) = a.strip_prefix(&format!("{nom}=")) {
            return Some(reste.to_string());
        }
    }
    None
}

fn port_de(args: &[String]) -> u16 {
    flag(args, "--port")
        .and_then(|p| p.parse().ok())
        .unwrap_or(ft::DEFAULT_PORT)
}

async fn serve(args: &[String]) -> ExitCode {
    let settings = ft::Settings::load();
    let runtime = ft::Runtime::detect(&settings);
    let port = port_de(args);
    let model = flag(args, "--model").unwrap_or_default();

    if model.trim().is_empty() {
        eprintln!(
            "[freetoken] aucun modèle : le moteur ne démarre pas sans poids.\n\
             Choisissez un checkpoint dans Réglages → Moteur, ou dans le catalogue de modèles \
             (filtre « Mixture-of-Experts »)."
        );
        return ExitCode::FAILURE;
    }

    // Ce qui manque est dit avant, avec la commande qui le répare. Laisser le
    // moteur échouer trente secondes plus tard n'apprend rien à personne.
    match ft::run(&ft::probe_invocation(&runtime), Duration::from_secs(120)).await {
        Ok(sortie) => {
            let sonde = ft::parse_probe(&sortie.stdout);
            if !sonde.ready() {
                eprintln!("[freetoken] le moteur ne peut pas démarrer sur cette machine :");
                for manque in &sonde.missing {
                    eprintln!("  - {manque}");
                }
                if !sortie.stderr.trim().is_empty() {
                    eprintln!(
                        "[freetoken] sortie d'erreur de la sonde :\n{}",
                        sortie.stderr.trim()
                    );
                }
                return ExitCode::FAILURE;
            }
            eprintln!(
                "[freetoken] {} · pilote {} · {} GPU · moteur {}",
                sonde.os.unwrap_or_else(|| "?".into()),
                sonde.driver_version.unwrap_or_else(|| "?".into()),
                sonde.gpus.len(),
                sonde
                    .engine_version
                    .or(sonde.engine_on_path)
                    .unwrap_or_else(|| "?".into())
            );
        }
        Err(e) => {
            eprintln!(
                "[freetoken] impossible de sonder la machine : {e}\n\
                 Sous Windows, vérifiez que WSL2 est installé (« wsl --install ») et qu'une \
                 distribution démarre."
            );
            return ExitCode::FAILURE;
        }
    }

    // Un serveur resté sur ce port tiendrait l'ancien modèle et la VRAM.
    let _ = ft::run(
        &ft::kill_stale_invocation(&runtime, port),
        Duration::from_secs(30),
    )
    .await;

    let invocation = ft::serve_invocation(&settings, &runtime, port, &model);
    eprintln!("[freetoken] {}", invocation.display());

    #[cfg(unix)]
    return remplacer_par_le_serveur(&invocation);
    #[cfg(windows)]
    return surveiller_le_serveur(&invocation, &runtime, port).await;
}

/// Sous Linux, ce processus **devient** le serveur.
///
/// Un maillon de moins entre ce que le socle surveille et ce qui écoute
/// vraiment : son signal d'arrêt atteint alors directement le serveur, au lieu
/// de tuer un lanceur qui laisserait le serveur en vie.
#[cfg(unix)]
fn remplacer_par_le_serveur(invocation: &ft::Invocation) -> ExitCode {
    use std::os::unix::process::CommandExt;
    let mut cmd = std::process::Command::new(&invocation.program);
    cmd.args(&invocation.args);
    // `exec` ne rend la main que s'il échoue.
    let erreur = cmd.exec();
    eprintln!("[freetoken] {} : {erreur}", invocation.program);
    ExitCode::FAILURE
}

/// Sous Windows, `wsl.exe` reste un maillon : on attend sa fin, on relaie son
/// code de sortie, puis on s'assure que le serveur Linux ne survit pas à la
/// fenêtre qui l'a lancé — sans quoi il garderait le port et la VRAM.
#[cfg(windows)]
async fn surveiller_le_serveur(
    invocation: &ft::Invocation,
    runtime: &ft::Runtime,
    port: u16,
) -> ExitCode {
    let mut cmd = invocation.to_command();
    ft::hide_console(&mut cmd);
    cmd.stdin(std::process::Stdio::null());
    let statut = match cmd.spawn() {
        Ok(mut enfant) => enfant.wait().await,
        Err(e) => {
            eprintln!(
                "[freetoken] {} : {e}\n\
                 WSL2 semble absent. Installez-le avec « wsl --install » depuis un terminal \
                 administrateur, puis redémarrez.",
                invocation.program
            );
            return ExitCode::FAILURE;
        }
    };
    let _ = ft::run(
        &ft::kill_stale_invocation(runtime, port),
        Duration::from_secs(30),
    )
    .await;
    match statut {
        Ok(s) if s.success() => ExitCode::SUCCESS,
        Ok(s) => {
            eprintln!("[freetoken] le serveur s'est arrêté ({s})");
            ExitCode::FAILURE
        }
        Err(e) => {
            eprintln!("[freetoken] attente du serveur impossible : {e}");
            ExitCode::FAILURE
        }
    }
}

async fn stop(args: &[String]) -> ExitCode {
    let settings = ft::Settings::load();
    let runtime = ft::Runtime::detect(&settings);
    let port = port_de(args);
    match ft::run(
        &ft::kill_stale_invocation(&runtime, port),
        Duration::from_secs(30),
    )
    .await
    {
        Ok(_) => {
            eprintln!("[freetoken] serveur du port {port} arrêté (s'il tournait)");
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("[freetoken] arrêt impossible : {e}");
            ExitCode::FAILURE
        }
    }
}

async fn probe() -> ExitCode {
    let settings = ft::Settings::load();
    let runtime = ft::Runtime::detect(&settings);
    match ft::run(&ft::probe_invocation(&runtime), Duration::from_secs(120)).await {
        Ok(sortie) => {
            let sonde = ft::parse_probe(&sortie.stdout);
            match serde_json::to_string_pretty(&sonde) {
                Ok(json) => println!("{json}"),
                Err(e) => {
                    eprintln!("[freetoken] sérialisation impossible : {e}");
                    return ExitCode::FAILURE;
                }
            }
            if sonde.ready() {
                ExitCode::SUCCESS
            } else {
                ExitCode::FAILURE
            }
        }
        Err(e) => {
            eprintln!("[freetoken] sonde impossible : {e}");
            ExitCode::FAILURE
        }
    }
}
