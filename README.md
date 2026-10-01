# poc-finetune-cv-extraction

Extração de currículos em PDF (pt-BR) para JSON estruturado com modelos pequenos
ajustados via QLoRA (Unsloth), exportados para GGUF e servidos com llama.cpp numa
RTX 3070 Ti de 8GB.

Um mesmo modelo base, o **Qwen3.5-4B** (multimodal nativo), em dois caminhos que competem
no mesmo conjunto de teste:

| Modalidade | Entrada |
|---|---|
| texto | texto extraído do PDF |
| visão | imagens das páginas |

Os dados de treino são **sintéticos**: o gabarito JSON é gerado primeiro (a partir das
personas da NVIDIA) e renderizado em PDF com templates variados, então o rótulo é exato
por construção. O número principal vem de um pequeno conjunto de currículos reais,
coletados com consentimento.

```bash
make setup
make smoke MOD=text       # pipeline inteiro em 200 currículos
make test
```

Decisões e motivos: [`docs/adr/`](docs/adr/README.md). Guia operacional: [`CLAUDE.md`](CLAUDE.md).

**Status:** esqueleto — schema, prompt, métricas e splits implementados e testados;
geração e estágios de GPU são stubs (ver fases no `CLAUDE.md`).
