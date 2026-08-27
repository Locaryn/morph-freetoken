// Le catalogue part tel quel chez l'utilisateur : l'hôte rejette en silence une
// entrée mal formée, et la ligne manquante ne se voit qu'à l'écran. Ce contrôle
// vérifie aussi que chaque dépôt répond — un dépôt privé ou sous licence à
// accepter répond 401, et l'installation échoue chez l'utilisateur, pas ici.
//
//   node scripts/check-catalogue.mjs          # forme seulement
//   node scripts/check-catalogue.mjs --online # + une requête par dépôt
import { readFileSync } from "node:fs";

const catalogue = JSON.parse(readFileSync(new URL("../dist/marketplace.json", import.meta.url)));
const manifeste = JSON.parse(readFileSync(new URL("../morph.json", import.meta.url)));

let erreurs = 0;
const fail = (message) => {
  console.error(`catalogue invalide : ${message}`);
  erreurs += 1;
};

if (catalogue.schemaVersion !== 1) fail("schemaVersion doit valoir 1");
if (!Array.isArray(catalogue.models) || catalogue.models.length === 0) fail("aucun modèle");
if (!Array.isArray(catalogue.categories) || catalogue.categories.length === 0) {
  fail("aucune catégorie");
}
if (catalogue.refreshUrl && !catalogue.refreshUrl.startsWith("https://")) {
  fail("refreshUrl doit être en https");
}

// La catégorie ne doit apparaître que si la capacité de l'extension est active.
// Sans cela, le filtre reste à l'écran après désactivation, et ne renvoie rien.
const capacites = manifeste.capabilities ?? [];
for (const categorie of catalogue.categories) {
  for (const requise of categorie.requires ?? []) {
    if (!capacites.includes(requise)) {
      fail(`catégorie ${categorie.id} : « ${requise} » n'est pas une capacité du manifeste`);
    }
  }
}

const requis = [
  "id",
  "name",
  "brand",
  "description",
  "license",
  "releaseDate",
  "releaseYear",
  "capabilities",
  "variants",
];

const identifiants = new Set();
const depots = new Set();

for (const model of catalogue.models) {
  for (const cle of requis) {
    if (model[cle] === undefined) fail(`${model.id ?? "?"} : champ « ${cle} » manquant`);
  }
  if (identifiants.has(model.id)) fail(`identifiant en double : ${model.id}`);
  identifiants.add(model.id);
  if (!Array.isArray(model.variants) || model.variants.length === 0) {
    fail(`${model.id} : aucune variante`);
    continue;
  }
  for (const variante of model.variants) {
    if (!/^https:\/\/huggingface\.co\/[^/]+\/[^/]+$/.test(variante.tag ?? "")) {
      // Ce moteur sert des répertoires de checkpoint, pas des fichiers isolés :
      // la source est donc une adresse de dépôt, jamais un `resolve/main/…`.
      fail(`${model.id} : « tag » doit être l'adresse d'un dépôt Hugging Face`);
      continue;
    }
    depots.add(variante.tag.replace("https://huggingface.co/", ""));
    if (typeof variante.params !== "number") fail(`${model.id} : params doit être un nombre`);
    if (typeof variante.storageGb !== "number") fail(`${model.id} : storageGb doit être un nombre`);
    if (!Array.isArray(variante.quants) || variante.quants.length === 0) {
      fail(`${model.id} : « quants » manquant`);
    }
  }
}

if (process.argv.includes("--online")) {
  for (const depot of [...depots].sort()) {
    const reponse = await fetch(`https://huggingface.co/api/models/${depot}`);
    if (!reponse.ok) {
      fail(`${depot} répond ${reponse.status} — il ne s'installera pas chez l'utilisateur`);
      continue;
    }
    const fiche = await reponse.json();
    // `gated` demande d'accepter une licence sur le site : le téléchargement
    // échoue alors avec un 401 que rien dans l'application ne peut réparer.
    if (fiche.gated) fail(`${depot} exige d'accepter une licence (gated: ${fiche.gated})`);
  }
}

if (erreurs > 0) process.exit(1);
console.log(
  `catalogue valide : ${catalogue.models.length} modèles, ${depots.size} dépôts${
    process.argv.includes("--online") ? " (vérifiés en ligne)" : ""
  }`,
);
