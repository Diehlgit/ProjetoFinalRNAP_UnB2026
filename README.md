# Classificação de Espécies Vegetais do Cerrado com Gating Orientado à Frequência

Projeto da disciplina *Redes Neurais e Aprendizado Profundo* (UnB, 2026/2).
Repositório: <https://github.com/Diehlgit/ProjetoFinalRNAP_UnB2026>

Este projeto investiga se os mecanismos de gating orientados à frequência propostos pelo
**GmNet** (WANG et al., 2026, ICLR 2026) melhoram o desempenho de redes neurais leves na
classificação de imagens de espécies vegetais nativas do Cerrado, a partir de observações do
iNaturalist no Distrito Federal.

> Status: em desenvolvimento. Nenhum resultado foi obtido até o momento.

## Estrutura

| Diretório | Conteúdo |
|-----------|----------|
| `configs/` | Configurações de caminhos, datasets e experimentos |
| `data/manifests/` | Manifestos CSV (imagens, espécies, licenças, atribuição). As imagens não são versionadas. |
| `notebooks/` | Notebooks de exploração, execução e análise |
| `src/cerrado_flora/` | Código reutilizável: `data`, `models`, `training`, `evaluation`, `utils` |
| `tests/` | Testes automatizados |
| `reports/` | Figuras e tabelas consolidadas |
| `docs/` | Decisões metodológicas e registro de uso de IA |
| `paper/` | Artigo em LaTeX (IEEE conferência, referências ABNT) |
| `outputs/` | Saídas locais temporárias (não versionadas) |

Imagens, datasets, checkpoints e resultados ficam no Google Drive.

## Configuração local

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

## Execução no Google Colab

Os experimentos rodam em kernels do Colab pela extensão oficial do VS Code (v0.2.1 ou superior).
Abra `notebooks/00_setup_colab.ipynb` e escolha o modo:

- **`upload`**, para desenvolvimento: envie `src/`, `configs/` e `pyproject.toml` com *Upload to Colab* (botão direito no Explorer).
- **`git`**, para experimentos reportados: faça commit e push; o notebook clona o repositório e registra o hash do commit.

## Referência

WANG, Y. et al. GmNet: Revisiting gating mechanisms from a frequency view. In: *International
Conference on Learning Representations (ICLR)*, 2026. Disponível em:
<https://openreview.net/pdf/f5c9e4310c6ef49e7dfecd3b09453af94ff17be5.pdf>.

## Autor

Arthur Diehl Barroso — Departamento de Ciência da Computação, Universidade de Brasília.
