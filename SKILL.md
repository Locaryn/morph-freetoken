---
name: freetoken-engine
description: Diagnostiquer, installer et régler le moteur d'inférence FreeToken — quand une conversation échoue faute de moteur, quand un modèle Mixture-of-Experts est lent, ou quand il faut rendre de la mémoire GPU sans décharger le modèle.
---

# Le moteur FreeToken

FreeToken sert localement des modèles **Mixture-of-Experts** dont les poids
dépassent la mémoire de la carte : les experts vivent en mémoire système, et
seuls ceux dont un jeton a besoin traversent le bus PCIe ou sont calculés par le
processeur. C'est ce qui permet de faire tourner un modèle de plusieurs centaines
de milliards de paramètres sur une machine dotée d'un seul GPU grand public.

Le moteur exige **Linux x86_64**, un **GPU NVIDIA** avec pilote **r580+** et le
**toolkit CUDA 13** (il compile ses noyaux au premier usage). Sous Windows, il
tourne dans **WSL2** ; l'extension s'en occupe et traduit les chemins.

## Ce qui appartient au socle, et ce qui appartient à ces outils

Le **choix du modèle et le démarrage du serveur** se font dans
**Réglages → Moteur** de l'application : c'est là que le modèle actif est
enregistré et que le processus est supervisé. Ne proposez pas de lancer le
serveur par un outil — deux propriétaires du même port, c'est un port occupé et
un démarrage qui échoue.

Ces outils servent à **comprendre et régler** ce que le socle pilote.

## Quel outil, pour quelle question

| La question posée | L'outil |
|---|---|
| « Pourquoi le moteur ne démarre pas ? », « qu'est-ce qui manque ? » | `freetoken_status` |
| « Installe le moteur », « mets-le à jour » | `freetoken_install` |
| « Quels modèles peut-il servir ? » | `freetoken_list_models` |
| « Le serveur tourne-t-il ? sur quel modèle ? » | `freetoken_server_health` |
| « Pourquoi est-ce lent ? » | `freetoken_server_stats` |
| « Combien de VRAM prend le cache ? » | `freetoken_cache_status` |
| « Rends de la VRAM sans perdre le modèle » | `freetoken_cache_resize` |
| « Optimise la répartition pour cette machine » | `freetoken_bench_bandwidth` |
| « Accélère le chargement au démarrage » | `freetoken_convert_checkpoint` |
| « Libère la carte » | `freetoken_stop_server` |

## Ordre à suivre quand rien ne marche

1. `freetoken_status`. Il répond par une liste de manques, chacun avec la
   commande qui le répare. **Lisez cette liste à l'utilisateur telle quelle**
   plutôt que de deviner : « pas de nvcc » et « pas de pilote NVIDIA » ne se
   corrigent pas au même endroit — sous WSL2 le pilote s'installe côté Windows,
   le toolkit dans la distribution.
2. Si seul le moteur manque, `freetoken_install`. Comptez plusieurs minutes et
   plusieurs gigaoctets ; dites-le avant de lancer.
3. Si tout est en place mais que le serveur ne répond pas, c'est le socle qui le
   démarre : renvoyez l'utilisateur vers Réglages → Moteur, où le journal du
   moteur est consultable.

## Régler la répartition des experts

Le réglage « Répartition des experts » vaut `auto` par défaut, et c'est le bon
choix — à condition d'avoir mesuré la machine. `freetoken_bench_bandwidth`
compare la bande passante de la mémoire système à celle du bus PCIe et écrit un
profil ; `auto` le relit ensuite pour choisir entre `offload` (les experts
manquants traversent le bus) et `hybrid` (une partie traverse, le reste est
calculé par le processeur).

Sans cette mesure, `auto` retient `offload`, ce qui n'est pas toujours le plus
rapide. Proposez la mesure une fois par machine, pas à chaque question sur la
lenteur.

`fused` garde tous les experts sur la carte : ne le proposez que si la VRAM
disponible dépasse la taille des experts, sinon le démarrage échoue.

## Rendre de la mémoire sans tout perdre

Recharger un modèle de cette taille coûte plusieurs minutes. Quand une autre
tâche a besoin de la carte — une génération d'image, par exemple —, préférez
`freetoken_cache_resize` avec un `moe` plus petit : le modèle reste chargé, les
experts en trop libèrent leur place, et la génération suivante sera un peu plus
lente au lieu d'attendre un rechargement complet.

`freetoken_stop_server` rend tout, et coûte un rechargement complet ensuite.

## Convertir un checkpoint

`freetoken_convert_checkpoint` réécrit un checkpoint au format de chargement
rapide du moteur. C'est **facultatif** : le moteur charge très bien les
checkpoints d'origine. Cela réduit le temps de démarrage, et **double l'espace
disque occupé** par ce modèle. Dites-le avant de proposer, et vérifiez la place
disponible pour un checkpoint de plusieurs dizaines de gigaoctets.

## Ce qu'il faut dire honnêtement

- Un modèle dont les experts pèsent plus que la mémoire système **ne tournera
  pas**, quelle que soit la répartition choisie. La mémoire indiquée par
  `freetoken_status` est le plafond réel.
- Sur une carte à 6 Go et 32 Go de mémoire système, un modèle de la classe
  30B-A3B est un objectif raisonnable ; les modèles de plusieurs centaines de
  milliards de paramètres exigent bien davantage de mémoire système. Ne promettez
  pas l'inverse parce que le catalogue les affiche.
- Les checkpoints multimodaux sont servis **en texte seulement**.
