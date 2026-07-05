# Projektnotizen — Geometric Encoder

## Überblick
tum.ai Research Track Projekt: geometric deep learning für crash simulation
surrogate models. Ziel: evtl. ICLR/ICML Publikation.

- Supervisor: Julius Riel
- Team: Victor Racu & Alex Burg → Physics Encoder (DEC-basiert:
  Discrete Exterior Calculus, 0-/1-/2-forms auf Vertices/Edges/Faces)
- Silja → Geometric Encoder: hierarchischer, multi-scale, geometry-aware
  Context, der die Physics-Tokens entlang der Verarbeitungsschritte
  informieren soll. Später Anbindung an Alex' DEC Encoder via Cross-Attention.

## Setup
- Server: ubuntu@129.70.51.6
- Projekt: /mnt/1t/mit-project/physicsnemo
- Dataset: /mnt/1t/mit-project/Dataset/full_body/
- venv: source /mnt/1t/mit-project/venv/bin/activate
- Branch: silja/geometric-encoding
- Eigene Dateien: examples/structural_mechanics/full_body_crash/

## Dataset
- Dodge Neon full body frontal crash, VTKHDF Format (vtkhdf_reader.py)
- ~1.1M Nodes/Sim, 284 strukturelle Teile, 17 Zeitschritte
- Felder: positions, displacement, velocity, stress_vm, plastic_strain,
  specific_energy, part_id
- Gültige Sims: 0002, 0003, 0005, 0007, 0008, 0010 (0010 = Val)
- Design-Parameter variieren: velocity_kmh, front_support_scale,
  lower_rail_subframe_scale

## Baseline: GeoTransolver
- Code: physicsnemo/experimental/models/geotransolver/
  (context_projector.py, gale.py, geotransolver.py)
- Config: crash/conf/model/geotransolver_one_shot.yaml
  (geometry_dim=3, slice_num=128, n_layers=6, include_local_features=true)
- Schwächen die adressiert werden: Kontext wird nur EINMAL gebaut und bleibt
  über alle GALE-Layer statisch; Ball-Queries sehen nur rohe Positionen,
  keine strukturelle/part-Info

## Aktueller Stand (eigene Dateien)
- geometric_encoder.py: GeometricEncoder — Ball Queries bei 3 Radien
  [0.1, 0.25, 1.0] (O(N²), N≤~8000), 7 Stats/Radius (mean_dist, std_dist,
  density, 3 PCA Eigenwerte, same_part_frac) + Part-Embedding
  (n_parts=300, dim=8) → MLP → (B,N,64) Context
- geo_transolver_enhanced.py: EnhancedGeoTransolver — hängt den
  Encoder-Output an den geometry-Tensor vor ContextProjector; part_id
  optional (Fallback: alle Nodes = "unknown" part)
- run_comparison.py: trainiert Baseline vs. Stats-only (ohne part_id) vs.
  Stats+part_id auf 5 Train-Sims + 1 Val-Sim, One-Shot Displacement
  Prediction (pos[t]-pos[0] für alle 17 Steps), 4096 Nodes/Sim, 200 Epochen,
  wandb + results.csv

## Ergebnisse bisher (200 Epochen, val relative L2)
- Baseline (nur XYZ):       0.1157
- Stats only (kein part_id): 0.1170
- Stats + part_id:           0.1167

Befund: Enhanced hilft früh (Epoche 10-40, ~-1.2pp), Baseline holt bis zur
Konvergenz auf. Vermutung: Datensatz zu klein (nur 5 Sims, 4096 zufällig
gesampelte Nodes/Sim), um den Vorteil des Encoders zu zeigen.

## Laufende Änderung (Stand 2026-07-05)
Ersetze uniformes stratified subsampling (über alle 284 Teile) durch
Subsampling beschränkt auf crash-relevante Teile — die meisten der 284
Teile (Kofferraum hinten, Türen, Räder, Verkleidung) verformen sich bei
einem Frontalcrash kaum und verwässern das Signal.

Entscheidung: datengetriebene Definition statt Namens-Allowlist — Teile
nach peak plastic strain über die Trainings-Sims ranken, Top-K behalten
(aktuell CRASH_TOP_K_PARTS=60, noch nicht kalibriert).
Implementiert in run_comparison.py: compute_crash_relevant_parts() /
subsample_from_raw().

Zum Gegenchecken: eigene Namens-Kandidatenliste (Rails/Längsträger,
A-/B-Pillar, Firewall, Bumper, Hood, Floor) — sobald das echte Ranking auf
den 5 Trainings-Sims läuft, damit abgleichen.

## Roadmap
- Sofort: Ranking auf echten 5 Trainings-Sims laufen lassen, um
  CRASH_TOP_K_PARTS zu kalibrieren; Vergleich mit physics-informed
  Sampling neu laufen lassen
- Größer: 50+ Simulationen und volle Node-Anzahl (~50k statt 4096) nötig
  für überzeugende Ergebnisse; Ablation-Studie welche Features am meisten
  helfen; Anbindung an Alex' DEC Physics Encoder via Cross-Attention

## Paper-Argument (ICLR)
"Existing methods either inject geometry once (FIGConv) or inject static
context repeatedly (GeoTransolver). We show that enriching ball queries
with GAOT-style statistical descriptors + part-aware features, injected
as a separate geometric encoder, improves convergence speed on crash
simulations where part boundaries are physically meaningful."

## Referenz-Papers (Details vor Zitieren nochmal prüfen)
- GAOT: MAGNO Encoder, Stats (n_neighbors, mean_dist, var_dist,
  centroid_offset, PCA Eigenwerte). Ablation: Geometry-Embedding nur im
  Encoder (nicht Encoder+Decoder) ist besser; Stats > PointNet-Style.
- PGOT: "Geometric Aliasing" (Token-Aggregation als Tiefpassfilter, löscht
  hochfrequente Randdetails). SpecGeo-Attention + TaylorDecomp-FFN
  (linearer/nichtlinearer Pfad, gelerntes Gate).
- FIGConvNet: implizites Grid + faktorisierte 2D-CNN-Grids, Geometrie nur
  einmal am Anfang injiziert. Stark bei Aerodynamik, schwach bei
  variierenden Randbedingungen.
- ReGUNet: Graph U-Net für Crash (B-Pillar Seitenaufprall), Coarsening
  3500→90 Nodes über 3 Level, rekurrenter Rollout über 12 Steps.
  0.32mm Testfehler, -51% vs. GNN-Baseline.
- NVIDIA GeoTransolver+FLARE (arXiv 2605.27758): testet GeoTransolver auf
  gleicher Bumper-Beam-Datenfamilie. FLARE ersetzt Self-Attention durch
  Low-Rank-Routing (O(NM), 2x weniger Speicher, bessere Genauigkeit).
  One-shot > autoregressiv/teacher-forcing für Stabilität. Muon Optimizer:
  -33% Fehler vs. GeoTransolver.
- Transolver-3: skaliert auf 160M-Zellen-Meshes via schnellerem
  Slice/Deslice, Geometry Slice Tiling, amortisiertem Training,
  Physical State Caching.
- DiT / PDE-Transformer: AdaLN-konditionierter Diffusion-Transformer für
  PDEs, braucht reguläre Grids — passt nicht zu unstrukturiertem Mesh.
