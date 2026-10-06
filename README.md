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

## Resultados

Resumo do run `full` (Qwen3.5-4B, Q4_K_M, RTX 3070 Ti). Relatório completo, com curva de
perda, splits, campos e templates: [`docs/RESULTS.md`](docs/RESULTS.md) (gerado por
`make report`).

**Treino** (QLoRA r=16, 1 época, 272 passos, ~4,35k currículos):

| Modalidade | Tempo | VRAM pico | Perda inicial → final | Perda média |
|---|--:|--:|--:|--:|
| texto | 56 min | 3,97 GB | 0,106 → 0,002 | 0,0063 |
| visão | 149 min | 5,36 GB | 0,108 → 0,002 | 0,0070 |

**Qualidade** no `test_unseen` (1.187 currículos de templates nunca vistos no treino),
decodificação com gramática JSON. Base → ajustado, em %:

| Modalidade | Escalares | Experiências F1 | Formação F1 | Habilidades F1 | Idiomas F1 | Alucinação |
|---|--:|--:|--:|--:|--:|--:|
| texto | 98,5 → **99,7** | 81,7 → **97,2** | 91,0 → **98,3** | 83,0 → **98,9** | 94,1 → **99,7** | 4,7 → 4,9 |
| visão | 95,7 → **99,6** | 91,6 → **99,6** | 99,3 → **99,8** | 97,9 → **99,2** | 89,5 → **99,9** | 5,7 → **5,2** |

**Currículo escaneado** (`test_unseen_scan`: os mesmos 1.187 currículos impressos e
escaneados — papel tingido, leve rotação, blur, JPEG), com gramática:

| Modalidade | Escalares | Experiências F1 | Formação F1 | Habilidades F1 | Alucinação |
|---|--:|--:|--:|--:|--:|
| texto | 12,4 → 4,5 | 0,0 → 0,0 | 0,0 → 0,0 | 0,0 → 0,9 | 0,0 → 97,9 |
| visão | 96,9 → **99,8** | 90,6 → **99,5** | 99,4 → **99,6** | 97,6 → **99,1** | 5,6 → **5,2** |

**Tempo de processamento** por currículo (média, 4 requisições simultâneas, com gramática):

| Modalidade | Base | Ajustado | Ganho | Tokens de saída (base → ajustado) |
|---|--:|--:|--:|--:|
| texto | 8,8 s | **5,6 s** | 1,57x | 530 → 343 |
| visão | 11,2 s | **7,6 s** | 1,47x | 525 → 345 |

Em resumo:

- O ajuste fino melhora as duas modalidades em todos os campos; o maior ganho está nas
  listas (experiências +15,5 pontos no texto, idiomas +10,4 na visão).
- Sem gramática, o modelo base não produz JSON válido (0%); o ajustado chega a 99,7–99,9%,
  ou seja, aprendeu o formato e dispensa a gramática.
- O ajustado é ~1,5x mais rápido porque gera ~35% menos tokens com a mesma vazão.
- Visão ajustada é a mais precisa (experiências 99,6 vs 97,2 no texto, sobretudo em layouts
  multicoluna); texto é ~26% mais rápido na inferência e treina em ~1/3 do tempo.
- A alucinação fica estável em ~5% nos dois modelos.
- **Escaneado, a visão não perde nada** (99,5 em experiências contra 99,6 no limpo). O
  texto não tem o que ler: sem OCR, o `pdfplumber` devolve vazio.
- **Cuidado com texto vazio + gramática:** o texto ajustado, forçado pela gramática a
  preencher `nome`, inventa o mesmo currículo falso ("Carlos Eduardo", `(11) 99999-9999`)
  para todos os 1.412 escaneados. Sem gramática ele responde corretamente tudo `null`. Em
  produção, texto extraído vazio deve desviar para a visão (ou OCR) antes do modelo.
- Ainda pendente: conjunto de teste real (`real_test`, fase 4, ver `CLAUDE.md`). O scan
  sintético é leve (mesma distribuição do aumento de treino); scans ruins de verdade
  ficam para o teste real.
