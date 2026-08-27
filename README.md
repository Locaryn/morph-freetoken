# morph-freetoken

Extension Locaryn qui apporte **FreeToken** comme moteur d'inférence : un
serveur local taillé pour les modèles **Mixture-of-Experts** dont les poids
dépassent la mémoire de la carte graphique.

Les experts vivent en mémoire système ; seuls ceux dont un jeton a besoin
traversent le bus PCIe ou sont calculés par le processeur. C'est ce qui permet de
faire tourner un modèle de plusieurs centaines de milliards de paramètres sur une
machine à un seul GPU grand public.

FreeToken est un projet de [FlashML](https://github.com/FlashML-org/FreeToken),
sous licence Apache-2.0. Cette extension ne le redistribue pas : elle l'installe
depuis PyPI, à une version épinglée, et le pilote.

---

## Ce que l'extension apporte

| | |
|---|---|
| **Moteur** | Un moteur de conversation supervisé par Locaryn, dans Réglages → Moteur, à côté du runtime intégré |
| **Catalogue** | Le filtre « Mixture-of-Experts » et douze checkpoints vérifiés, ajoutés au catalogue de modèles de l'application |
| **Outils** | Diagnostic de la machine, installation, mesure de bande passante, état et redimensionnement des caches, conversion de checkpoint |
| **Compétence** | Explique au modèle quel outil répond à quelle question |

L'application n'apprend rien de FreeToken : elle lit la section `engine` du
manifeste — comment installer, comment lancer, comment sonder, quels formats de
poids sont servis. Tout ce qui est propre au moteur, y compris le passage par
WSL2 sous Windows, vit dans le programme que cette extension livre.

---

## Ce que la machine doit offrir

- **Linux x86_64**, GPU **NVIDIA**, pilote **r580+**, **CUDA 13** avec `nvcc`
  (le moteur compile ses noyaux au premier usage)
- **Python ≥ 3.10** — `uv` est utilisé s'il est présent, sinon `python -m venv`
- De la **mémoire système** : c'est elle qui héberge les experts, et c'est le
  vrai plafond. Un modèle de la classe 30B-A3B demande une trentaine de
  gigaoctets ; les modèles de plus de 100 milliards de paramètres en demandent
  plusieurs centaines.

**Sous Windows**, il n'existe pas de version native : l'extension passe par
**WSL2**. Installez une distribution (`wsl --install`) et le toolkit CUDA *à
l'intérieur* — le pilote, lui, s'installe côté Windows. L'outil
`freetoken_status` dit exactement ce qui manque, avec la commande qui le répare.

---

## Installation

Réglages → Extensions → Ajouter, puis :

```
github:Locaryn/morph-freetoken@v1.0.0
```

L'extension arrive désactivée. Accordez-lui ses permissions, activez-la, puis :

1. Lancez l'outil `freetoken_status` depuis le chat — il liste ce qui manque.
2. `freetoken_install` pose le moteur dans un environnement dédié à
   l'extension (plusieurs minutes, plusieurs gigaoctets).
3. `freetoken_bench_bandwidth`, une fois par machine : le moteur choisira
   ensuite tout seul comment répartir les experts.
4. Catalogue de modèles → filtre « Mixture-of-Experts » → installez un
   checkpoint.
5. Réglages → Moteur → « FreeToken » → **Utiliser**.

---

## Réglages

| Réglage | Effet |
|---|---|
| Répartition des experts | `auto` (suit la mesure), `offload`, `hybrid`, `cpu`, `fused` |
| Part de la VRAM libre utilisable | Descendez si un autre programme partage la carte |
| Jetons de sortie par défaut | Budget des requêtes qui n'en fixent pas |
| GPU | Index ou UUID de `nvidia-smi` ; vide = la première carte |
| Distribution WSL | Vide = la distribution par défaut ; sans effet sous Linux |
| Arguments supplémentaires | Ajoutés tels quels à `ft serve` |

---

## Les outils

| Outil | Ce qu'il fait |
|---|---|
| `freetoken_status` | Système, pilote, `nvcc`, cartes, mémoire, version installée, état du serveur |
| `freetoken_install` | Installe ou met à jour le moteur à la version épinglée |
| `freetoken_list_models` | Les checkpoints servables, et ceux déjà convertis |
| `freetoken_server_health` | Le serveur répond-il, sur quel modèle |
| `freetoken_server_stats` | Débit, latence, VRAM, files |
| `freetoken_cache_status` | Les réservoirs de cache |
| `freetoken_cache_resize` | Rend de la VRAM **sans** décharger le modèle |
| `freetoken_bench_bandwidth` | Mesure mémoire système contre PCIe, une fois par machine |
| `freetoken_convert_checkpoint` | Format de chargement rapide ; facultatif, double l'espace disque |
| `freetoken_stop_server` | Rend toute la mémoire |

Le **démarrage** du serveur n'est pas un outil : il appartient à Réglages →
Moteur, qui enregistre le modèle actif et supervise le processus. Deux
propriétaires du même port donnent un port occupé et un démarrage manqué.

---

## Construire depuis les sources

```bash
cargo build --release
cargo test
node scripts/check-catalogue.mjs --online
```

Les deux exécutables produits vont dans `bin/` :

- `locaryn-freetoken-launch` — nommé par `engine.lifecycle.start` ; c'est lui
  qui sait passer par WSL2 et traduire les chemins ;
- `locaryn-freetoken-mcp` — le serveur MCP des outils.

`bin/` est ignoré par Git : l'archive des sources d'un dépôt GitHub ne le
contient donc pas, et une extension installée depuis les sources s'active,
s'affiche et ne fait rien. La CI compile par plateforme et publie une archive
nommée avec l'OS et l'architecture ; l'application cherche ce paquet en premier.

---

## Licence

Apache-2.0. FreeToken lui-même est distribué par FlashML sous Apache-2.0 et
n'est pas redistribué ici. Les poids de chaque modèle du catalogue restent sous
la licence de leur dépôt, indiquée dans la fiche.
