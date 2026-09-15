# Plan technique — Prototype de traduction FR→EN (Seq2Seq RNN/GRU)

> Document de cadrage pour agents de code. Chaque étape liste : objectif, entrées,
> sorties attendues, et **points de contrôle** à vérifier avant de passer à la suite.
> Contrainte : projet de 2 semaines, périmètre volontairement resserré.

---

## Contexte & décisions déjà prises

- **Tâche** : traduction automatique FR→EN, pipeline Seq2Seq complet (prototype pédagogique, pas industriel).
- **Dataset** : `Helsinki-NLP/tatoeba`, paire `lang1="en", lang2="fr"` (corpus brut, un seul split `train` à découper soi-même).
- **Profil du corpus** (EDA déjà faite) : 264 905 paires, phrases courtes (médiane 6 mots EN et FR), queue longue ~1 % > 19 mots.
- **Architectures comparées** : **RNN simple vs GRU** (pas de LSTM, écarté par contrainte de temps).
- **Objectif de la comparaison** : mesurer *à quel point* l'avantage connu du GRU est visible **sur ce dataset**, pas prouver que GRU > RNN (déjà connu).

---

## Étape 0 — Chargement & inspection

**Objectif** : charger le corpus, confirmer la structure.

- Charger `Helsinki-NLP/tatoeba` (en/fr).
- Extraire deux colonnes : `fr` (source), `en` (cible).
- **Points de contrôle** :
  - 264 905 paires environ.
  - 0 valeur manquante.
  - Afficher 5 exemples pour confirmer l'alignement FR↔EN.

---

## Étape 1 — Nettoyage

**Objectif** : retirer le bruit, dans cet ordre.

1. Retirer les **chaînes vides** (~5) : `len(fr.strip()) == 0 or len(en.strip()) == 0`.
2. Retirer les paires **en == fr** (~15) : identiques dans les deux langues (bruit, noms propres, nombres).
3. Retirer les paires au **ratio de longueur aberrant** (~268) : `wordRatio` hors d'un intervalle raisonnable (ex. garder `0.5 <= ratio <= 2.0`), cible les mauvais alignements.
4. Appliquer un **max_len généreux** en mots (~40) pour couper les aberrations (max observé 209 mots) **sans amputer la traîne longue utile**.

- **Points de contrôle** :
  - Perte totale < 1 % du corpus (attendu).
  - Logger le nombre de paires retirées à chaque sous-étape.
  - Après nettoyage : plus aucune paire vide, plus aucune en==fr.

---

## Étape 2 — Déduplication

**Objectif** : retirer les doublons exacts (rempart anti-fuite de premier niveau).

- Retirer les **paires exactement dupliquées** (~508) : même (fr, en).
- **Ne pas** dédupliquer sur une seule langue ici (les cibles EN partagées sont gérées par le split, pas supprimées — plusieurs sources FR → une même cible EN est légitime et utile).

- **Points de contrôle** :
  - ~508 paires retirées.
  - `df.duplicated(subset=['fr','en']).sum() == 0` après opération.

---

## Étape 3 — Split train/val/test (LE point critique)

**Objectif** : découper sans fuite de cible **et** avec proportion de phrases longues contrôlée dans chaque part.

### 3a. Non-fuite : groupes par cible EN
- **Grouper par phrase cible EN.** Chaque groupe = toutes les paires partageant une même phrase anglaise.
- L'assignation train/val/test se fait **par groupe entier** : un groupe ne peut jamais chevaucher deux parts.
- Raison : la fuite qui compte est celle de la **cible** (ce que le modèle génère et sur quoi il est scoré). Une cible vue à l'entraînement puis re-testée gonfle artificiellement le BLEU.
- Fuite de source FR : **non traitée** (choix pragmatique assumé, fuite plus légère).

### 3b. Stratification par longueur (au niveau des GROUPES)
- Attribuer à chaque groupe une **longueur** = nombre de mots de sa cible EN.
- Définir 2 strates via un seuil **paramétrable, défaut 18 mots** :
  - strate **courte** : longueur ≤ 18
  - strate **longue** : longueur > 18
- Appliquer le ratio de split **à l'intérieur de chaque strate séparément**, au niveau des groupes → train/val/test ont la **même proportion de longues**.

### 3c. Split en deux temps (pour 3 parts)
- Ratios **paramétrables**, défaut suggéré : **80 / 10 / 10** ou **70 / 15 / 15** (donner un peu de marge au test pour un bin long bien peuplé).
- Temps 1 : (train+val) vs test — stratifié + par groupe.
- Temps 2 : train vs val — stratifié + par groupe.
- Outil : `train_test_split(..., stratify=<strate_du_groupe>)` appliqué à la **liste des groupes**, OU `StratifiedGroupKFold`. Préférer l'approche « stratifier la liste des groupes puis assigner » (plus lisible, plus déboggable).

### 3d. Points de contrôle (OBLIGATOIRES)
- **Test de non-fuite** : `set(cibles_EN_train) ∩ set(cibles_EN_test) == ∅` (idem train/val et val/test). Doit renvoyer **zéro**.
- **Proportion de longues** : vérifier qu'elle est ~identique dans train, val, test.
- **Compte absolu de longues dans le test** : doit être suffisant pour des métriques stables (viser ≥ quelques centaines). Si insuffisant, ajuster le seuil ou les ratios.
- **Vérification bonus** (à logger) : taille moyenne des groupes dans strate courte vs longue. Attendu : les longues sont quasi toutes des singletons (peu de doublons) → confirme que la déduplication par groupe n'affecte en pratique que les courtes.
- Ratio 80/20 en *paires* flottera légèrement (groupes de tailles variables) — acceptable, ne pas sur-corriger.

---

## Étape 4 — Vocabulaire (mots entiers)

**Objectif** : deux vocabulaires séparés, construits sur le TRAIN seul.

- **Décision actée** : tokenisation **mots entiers**, coupée à **95 % de couverture** des occurrences.
  - EN : ~3 700 mots. FR : ~7 000 mots.
  - Justifié par la couverture (au-delà : 7,5× plus de vocab pour +5 % de couverture ; hapax = 38 % du vocab pour ~0,6 % de couverture).
- **Deux vocabulaires distincts** : source FR, cible EN.
- **4 tokens spéciaux** aux indices 0–3 : `<pad>`=0, `<unk>`=1, `<sos>`=2, `<eos>`=3.
- Seuil via `min_freq` OU via taille cible (top-K par fréquence) — les deux équivalents ; choisir celui qui atteint ~95 % de couverture.
- **Construit sur le split TRAIN uniquement** (anti-fuite). Les mots test/val absents du train → `<unk>` (comportement réaliste).
- Encodage d'une phrase : `[<sos>] + indices + [<eos>]`, OOV → `<unk>`.

- **Points de contrôle** :
  - Indices spéciaux corrects (0–3).
  - Couverture réelle atteinte ~95 % (logger la valeur exacte).
  - Vocabulaire FR nettement > vocabulaire EN (asymétrie morphologique attendue).
  - Encoder/décoder une phrase test → round-trip correct.

> Base de code déjà existante : `vocab_builder.py` (à adapter).

---

## Étape 5 — Numérisation & padding

**Objectif** : transformer les phrases encodées en tenseurs prêts pour le modèle.

- Fixer `max_len` **en tokens** (pas en mots) — re-mesurer la distribution de longueur *après* encodage. Ici mots entiers → max_len ≈ max_mots + 2 bornes.
- Padder à `max_len` avec `<pad>`, tronquer au-delà.
- Créer les `DataLoader` (train/val/test), batchs, avec masque de padding.

- **Points de contrôle** :
  - Aucune séquence ne dépasse `max_len`.
  - Le `<pad>` est bien ignoré dans le calcul de la loss (voir étape 7).

---

## Étape 5bis — Caractérisation par config de tokenisation (avant Optuna)

**Objectif** : produire, **une fois par config de tokenisation** (mots-entiers 95 %, sous-mots Unigram, sous-mots BPE…), un rapport de stats descriptives qui servira à *expliquer* les résultats et à alimenter la soutenance (discussion 12c/d + piste Zipf demandée par le prof).

> **Où** : en amont d'Optuna, PAS dans la boucle d'essais. Ces stats dépendent du tokeniseur et du split, pas des hyperparamètres — les recalculer par essai serait redondant.
> **Sur quoi** : sur le **TRAIN** (cohérent avec la construction du vocabulaire). Optionnel : rapporter aussi val/test pour vérifier l'homogénéité du split.

### Tableau des 5 stats clés (par config)
Ce tableau raconte le **compromis** de chaque tokenisation d'un coup d'œil :
- **Longueur en tokens** : moyenne, médiane, p95, p99, max. (La variable qui gouverne le décrochage RNN et la charge du vecteur de contexte.)
- **Ratio tokens/mots** (facteur de fragmentation) : `nb_tokens / nb_mots`. 1.0 = mots entiers ; >1 = découpage sous-mots.
- **Taille du vocabulaire effectif** : nb de tokens distincts réellement utilisés.
- **Taux d'OOV** (`<unk>`) : ~0 pour les sous-mots, non-nul pour les mots entiers.
- **Couverture** : % des tokens du texte couverts par le vocabulaire.

### Comparaison Zipf BPE vs Unigram (piste soulevée par le prof)
Produire **une figure comparative** (pas une courbe pleine par config, redondant — Zipf apparaît à toute granularité). Ce qu'on cherche = les **différences fines** entre les deux mécanismes (BPE fusionne bottom-up par fréquence ; Unigram élague top-down par vraisemblance) :

- **Panneau 1 — Zipf superposé** : courbes BPE vs Unigram sur le même graphe log-log, avec pentes ajustées. Observer surtout le comportement de la **queue** (tokens rares), où les deux méthodes divergent le plus.
- **Panneau 2 — Distribution des longueurs de tokens** (histogramme, en caractères) BPE vs Unigram. C'est là que la différence de mécanisme se voit le mieux : Unigram tend vers des tokens plus longs / plus « morphémiques », BPE vers des fusions plus opportunistes. **Souvent plus informatif que le Zipf lui-même.**
- **Panneau 3 (optionnel) — Fragmentation** : tokens/mot par méthode, notamment sur les mots rares (relié à la longueur de séquence → décrochage RNN).

> **À faire si possible** : demander au prof ce qu'il s'attend à voir sur le Zipf (probablement : la distribution de fréquence comme *diagnostic de l'équilibre* du vocabulaire — ni trop concentré = OOV, ni trop plat = séquences trop longues). Aligner l'analyse sur son intention.
>
> **Conclusion honnête acceptée** : si les distributions Zipf s'avèrent très proches sur ce corpus simple, le dire — « la principale différence BPE/Unigram est sur la longueur des tokens, pas sur la forme Zipf » est une vraie réponse argumentée. C'est la démarche d'analyse qui est évaluée, pas un résultat spectaculaire.

### Stat liée aux runs (à mesurer plus tard, au niveau config)
- **Temps moyen par epoch** par config : les sous-mots allongent les séquences → entraînement plus lent. Utile pour discuter le compromis. Se mesure au niveau de la config, pas de l'essai Optuna.

- **Points de contrôle** :
  - Un rapport de stats stocké/versionné **par config**, avant de lancer Optuna dessus.
  - Vérifier la cohérence : ratio tokens/mots = 1.0 pour mots entiers, >1 pour sous-mots.
  - Vérifier l'homogénéité train/val/test si les trois sont rapportés (pas de décalage de distribution introduit par le split).

---

## Étape 6 — Ossature du modèle (encodeur-décodeur)

**Objectif** : UNE ossature qui accueille toutes les variantes sans réécriture.

### Encodeur
- `embedding → cellule récurrente`.
- **Cellule paramétrable** : `nn.RNN` ou `nn.GRU` (même interface, changement d'un paramètre).
- **Renvoyer TOUS les états cachés** (pas seulement le dernier) → indispensable pour greffer l'attention plus tard sans refonte.

### Décodeur
- `embedding → cellule récurrente → couche linéaire de sortie`.
- **Paramètre `teacher_forcing_ratio` intégré dès le départ** :
  - `1.0` → teacher forcing pur (entraînement de base).
  - décroissant → scheduled sampling.
  - `0.0` → génération libre (proche inférence).
- À chaque pas : tire un aléa < ratio → nourrit avec le vrai token (référence) ; sinon → nourrit avec sa propre prédiction précédente.

### Attention (Banana Attention — Bahdanau additive) — greffable
- Module d'alignement séparé, activable par flag.
- À chaque pas du décodeur : calcule des scores entre l'état courant du décodeur et **tous** les états de l'encodeur → softmax → vecteur de contexte pondéré → concaténé à l'entrée du décodeur.
- Se branche **sans toucher** encodeur ni boucle principale (d'où « renvoyer tous les états » à l'encodeur).

- **Points de contrôle** :
  - Forward passe sur un batch jouet sans erreur de dimension.
  - Bascule RNN↔GRU = 1 paramètre.
  - Bascule attention on/off = 1 flag.
  - `teacher_forcing_ratio` à 0 et à 1 donnent des comportements distincts vérifiables.

---

## Étape 7 — Boucle d'entraînement

**Objectif** : entraîner proprement, loss correcte.

- Loss : `CrossEntropyLoss(ignore_index=<pad>)` — **le padding ne doit pas compter** dans la loss.
- Optimiseur : Adam (lr = hyperparamètre).
- Gradient clipping (les RNN en ont besoin — explosion de gradient).
- Suivre par epoch : train loss, val loss, **perplexité**.
- Sauvegarder le meilleur modèle (sur val loss).

- **Points de contrôle** :
  - La train loss descend (sinon bug de pipeline).
  - Le `<pad>` est bien exclu (vérifier `ignore_index`).
  - Pas de NaN (si NaN → clipping/lr).

---

## Étape 8 — Recherche d'hyperparamètres (Optuna)

**Objectif** : régler chaque architecture à son optimum (option A).

### Principe
- **Option A** : un run Optuna **séparé par architecture** (RNN, puis GRU). Chacun réglé à son meilleur → comparaison « meilleur plafond vs meilleur plafond ».
- Formulation rapport : « chaque architecture optimisée indépendamment pour une comparaison à armes égales ; on mesure l'écart de *plafond* atteignable, pas à réglages identiques ».

### Efficacité
- Recherche sur un **sous-ensemble ~20 % stratifié par longueur** (garder des phrases longues dedans !).
- Activer le **pruning** (coupe les essais ratés tôt).
- Espace de recherche **restreint aux vrais hyperparamètres** (ceux qui ne changent pas les données) :
  - **learning rate** (prioritaire, ~80 % de l'effet)
  - dimension cachée
  - dropout
  - taille d'embedding
- **NE PAS** mettre tokeniseur/vocabulaire dans l'espace de recherche (ils changent les données — voir étape 8bis).

### Limite à documenter
- Les hyperparamètres « 20 % » sont un bon point de départ, pas l'optimum exact du dataset complet (léger risque de sur-régularisation).

- **Points de contrôle** :
  - Le sous-ensemble 20 % contient bien des phrases longues.
  - Logger tous les essais (`study.trials_dataframe()`).
  - Récupérer les meilleurs hyperparamètres par architecture.

---

## Étape 8bis — Exploration tokenisation (optionnel, pour la soutenance)

**Objectif** : montrer qu'on s'est penché sur l'aspect technique des tokeniseurs, SANS prétendre à une comparaison contrôlée.

- Intégrer plusieurs approches de tokenisation comme configurations **fixées par run** (pas dans l'espace Optuna) :
  - mots entiers 95 % (référence)
  - un tokeniseur **sous-mots** (Unigram via SentencePiece, pour cohérence avec MarianMT)
- Cadrage explicite : **exploration**, pas comparaison rigoureuse (le temps n'a pas permis d'entraînement complet de chaque variante ; les effets tokeniseur/réglages seraient confondus dans Optuna).
- Livrable soutenance : tableau/box-plot des scores par type de tokeniseur (depuis `trials_dataframe()`), pour **en parler techniquement** (BPE vs Unigram vs mots, gestion OOV, `▁` et dummy prefix, allongement des séquences).
- **S'appuyer sur les stats descriptives de l'étape 5bis** (longueurs en tokens, fragmentation, Zipf comparatif, longueurs de tokens) pour *expliquer* les écarts de score observés, plutôt que de les constater. C'est le lien entre « caractérisation de la tokenisation » (5bis) et « effet sur le modèle » (ici).

- **Points de contrôle** :
  - Ne PAS présenter comme « quel tokeniseur gagne ».
  - Présenter comme « exploration de l'effet, matière à discussion ».

---

## Étape 9 — Entraînement final

**Objectif** : entraîner les modèles retenus jusqu'au bout, dataset complet.

- RNN et GRU, hyperparamètres fixés (issus de leurs Optuna respectifs).
- Dataset **complet** (pas le 20 %).
- Entraînement jusqu'à convergence (early stopping sur val loss).
- **Variantes à ajouter si le temps** (par ordre de priorité) :
  1. **GRU + Banana Attention** (cœur pédagogique, au barème Q13) — comparé au GRU seul.
  2. **Scheduled sampling** (fait décroître `teacher_forcing_ratio` selon un planning) — prolonge le teacher forcing.

- **Points de contrôle** :
  - Courbes de loss/perplexité sauvegardées par modèle.
  - Modèles sauvegardés pour l'évaluation.

---

## Étape 10 — Évaluation (partie valorisée)

**Objectif** : mesurer la vraie qualité de traduction, par bins de longueur.

### Métriques (sur traductions RÉELLEMENT générées, PAS en teacher forcing)
- **SacreBLEU** : n-grammes, standardisé, sévère sur les synonymes.
- **METEOR** : intègre synonymes + racinisation, plus souple.
- **BERTScore** : similarité sémantique par embeddings contextuels (le plus lourd).
- Via la bibliothèque `evaluate` de Hugging Face.

### Découpage
- Évaluer **séparément par bin de longueur** : courtes (≤18) / longues (>18).
- C'est sur le bin **long** que l'écart RNN/GRU et l'effet de l'attention se révèlent (invisibles sur les courtes, corpus sous le seuil de décrochage ~15-20 mots).

### Analyses attendues (ce qui fait la note)
1. **Désaccords entre les 3 métriques** : où BLEU pénalise un synonyme que METEOR/BERTScore acceptent.
2. **Biais du teacher forcing (Q14a)** : l'accuracy sur tokens pendant l'entraînement est gonflée (le modèle reçoit toujours le bon mot précédent) → ne reflète pas la qualité en inférence → d'où la nécessité de métriques sur texte généré.
3. **Évolution par epoch** : cross-entropy loss, perplexité, SacreBLEU/BERTScore.
4. **Tests de prédiction** : afficher phrase source / traduction produite / référence (Q14d).

- **Points de contrôle** :
  - Les métriques sont calculées sur des sorties **générées** (génération libre), pas en teacher forcing.
  - Résultats ventilés par bin de longueur.
  - Si l'écart reste faible même sur le bin long → l'analyser (« corpus sous le seuil de décrochage »), pas le forcer.

---

## Attendu réaliste (à garder en tête)

- Sur Tatoeba (médiane 6 mots), les écarts **architecture** et **attention** seront **faibles sur les phrases courtes**.
- La littérature (Bahdanau 2015) situe la séparation des courbes vers **15-20 mots**.
- Le **bin de test long** est la fenêtre pour montrer ces écarts.
- Un **résultat serré bien expliqué** vaut un résultat positif : « même les phrases longues du corpus restent proches du seuil » est une conclusion mature.

---

## Priorités (si le temps manque)

1. ✅ RNN vs GRU (socle, 2 prototypes au barème)
2. ✅ Banana Attention (GRU avec/sans — cœur pédagogique Q13)
3. ✅ Évaluation 3 métriques + analyse
4. ⏳ Scheduled sampling
5. 🗑️ Trappe assumée : vocab complet vs 95 % (déjà tranché analytiquement), LSTM, duel BPE vs Unigram

---

## Fuites & pièges à ne jamais oublier

- **Vocabulaire construit sur le TRAIN seul.**
- **Split par groupe sur la cible EN** (test de non-fuite = 0).
- **Phrases longues gardées dans le TRAIN** (sinon le test long mesure du sous-apprentissage, pas l'architecture).
- **Loss ignore le `<pad>`.**
- **Métriques finales sur texte généré**, pas en teacher forcing.
- **max_len en tokens** pour le padding (pas en mots).
