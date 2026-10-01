# Template difficulty probe — 2026-10-01

**Question:** do the 20 templates make the text task hard enough for base Qwen3.5-4B +
grammar to leave room for the fine-tune? The phase 0 spike warned that a clean
one-column resume is already near the ceiling.

**Setup:** `configs/data-probe.yaml` (run `probe`), 200 resumes over all 20 templates
with the teacher on, base `cvx-text-base-q4_k_m.gguf`, grammar mode only, every split.
Grammar schema with every key required (ADR 0007). ~10 resumes per template, so a single
row moves a template's figure by several points: read the ranking, not the decimals.

```
uv run python scripts/01_generate_cvs.py --config configs/data-probe.yaml --smoke
uv run python scripts/02_build_dataset.py --config configs/qwen35-4b-text.yaml --data-config configs/data-probe.yaml --smoke
uv run python scripts/06_evaluate.py --config configs/qwen35-4b-text.yaml --data-config configs/data-probe.yaml --smoke --which base --modes grammar --splits train,val,test_seen,test_unseen
```

## Result (base, grammar, F1 % / scalars exact %)

| template | n | scalars | exp | exp dates | edu | skills | lang |
|---|--:|--:|--:|--:|--:|--:|--:|
| cartoes | 9 | 98.1 | 37.3 | 72.7 | 100.0 | 98.2 | 100.0 |
| duas_colunas | 9 | 100.0 | 70.8 | 85.3 | 66.7 | 77.8 | 100.0 |
| colunas_fluidas | 10 | 98.3 | 73.1 | 63.2 | 69.6 | 75.9 | 92.3 |
| lateral_direita | 11 | 93.9 | 74.1 | 85.0 | 100.0 | 61.9 | 100.0 |
| lateral | 15 | 96.7 | 79.6 | 91.9 | 100.0 | 61.2 | 94.1 |
| rodape | 7 | 97.6 | 80.0 | 100.0 | 80.0 | 95.1 | 100.0 |
| caixa_resumo | 6 | 97.2 | 83.3 | 100.0 | 100.0 | 100.0 | 100.0 |
| tabela_grade | 9 | 98.1 | 85.0 | 91.2 | 100.0 | 96.8 | 100.0 |
| perfil_lateral | 6 | 94.4 | 88.9 | 100.0 | 80.0 | 97.6 | 83.3 |
| tabela | 10 | 100.0 | 89.7 | 94.2 | 100.0 | 100.0 | 94.7 |
| linha_do_tempo | 12 | 100.0 | 90.0 | 64.8 | 100.0 | 98.2 | 94.7 |
| cabecalho_faixa | 8 | 100.0 | 91.3 | 97.6 | 100.0 | 100.0 | 100.0 |
| etiquetas | 14 | 98.8 | 91.7 | 95.5 | 91.7 | 83.2 | 100.0 |
| secoes_margem | 9 | 98.1 | 93.3 | 96.4 | 94.7 | 98.9 | 95.7 |
| academico | 11 | 97.0 | 93.5 | 96.6 | 100.0 | 95.9 | 94.7 |
| executivo | 11 | 100.0 | 95.5 | 97.6 | 100.0 | 99.2 | 85.7 |
| linha_unica | 9 | 100.0 | 95.8 | 97.8 | 100.0 | 95.8 | 100.0 |
| tecnico | 9 | 100.0 | 96.3 | 100.0 | 100.0 | 91.6 | 85.7 |
| classico | 14 | 100.0 | 100.0 | 90.0 | 100.0 | 100.0 | 100.0 |
| rotulado | 11 | 98.5 | 100.0 | 93.2 | 100.0 | 100.0 | 92.3 |
| **all** | 200 | 98.4 | 84.6 | 90.4 | 95.0 | 89.5 | 94.8 |

## Reading

- **Layout, not content, is the difficulty.** Side-by-side regions (cards, equal
  columns, flowing columns, sidebars) cost 20-60 points of experience F1; one-column
  layouts stay at 93-100. Failures are the intended ones: `cartoes` merges two cards
  into one job ("Auxiliar de logística Conferente", two employers concatenated);
  `linha_do_tempo` mis-pairs stacked start/end dates; sidebars bleed skills into the
  main column.
- **Some errors are not layout-specific:** the base often keeps "— Cidade" inside
  `empresa` (seen in `rodape`). Training should fix these everywhere.
- **Scalars are saturated (94-100) on every layout.** Contact fields will not separate
  base from tuned; experiences, education dates and skills will.
- **Held-out set** (`cvx.splits`, 20% of 20 by hash): `colunas_fluidas`,
  `lateral_direita`, `linha_do_tempo`, `tecnico` — two hard multi-column layouts, the
  date-pairing layout and an easy one. Base exp F1 on them ≈ 73 / 74 / 90 / 96.
- **Scan noise is not a text-modality lever:** a scanned PDF has no text layer for
  pdfplumber. It belongs to the vision path (phase 3, `rasterize.py` augmentation).

## Data fix found on the way

~21% of teacher descriptions (80/375 in the smoke run) led with a job title
("Gerente de restaurante: ..."), twice another job's. `teacher._strip_title` now drops
the prefix and the prompt asks not to repeat the title. Descriptions are not scored,
but a wrong title in the prose would teach the model to read titles from it.
