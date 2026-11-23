# Projeto: Classificação de Espécies Vegetais do Cerrado com Gating Orientado à Frequência

## 1. Contexto

Este projeto investiga a aplicação de mecanismos de gating orientados à frequência, conforme apresentados no artigo:

GmNet: Revisiting Gating Mechanisms From A Frequency View (WANG et al., ICLR 2026). Código oficial: https://github.com/YFWang1999/GmNet. Preprint: arXiv:2503.22841.

O objetivo é avaliar se a incorporação dos mecanismos de gating orientados à frequência propostos pelo GmNet melhora o desempenho de redes neurais leves na classificação de imagens de espécies vegetais nativas do Cerrado. A investigação considera especialmente a capacidade desses mecanismos de enriquecer as representações visuais por meio de uma melhor exploração de componentes de diferentes frequências, incluindo detalhes como texturas e bordas.

## 2. Escopo científico

- Domínio: classificação de imagens de espécies vegetais nativas do Cerrado.
- Fonte inicial de observações: iNaturalist.
- Localização inicial da coleta: Distrito Federal, Brasil.
- Seleção das classes: espécies cuja ocorrência nativa no Cerrado seja validada.
- Modelo de referência: arquitetura baseline de classificação.
- Modelo experimental: baseline modificado com o mecanismo de gating do GmNet.
- Comparação: mesmas condições experimentais, conjunto de dados e divisões de treinamento, validação e teste.

A hipótese de pesquisa é se a incorporação de mecanismos de gating orientados à frequência, conforme propostos pelo GmNet, melhora o desempenho de redes neurais leves na classificação de imagens de espécies vegetais nativas do Cerrado.

Essa hipótese deve ser investigada experimentalmente, não tratada como resultado estabelecido.

## 3. Ambiente de desenvolvimento

- Editor: Visual Studio Code.
- Assistente de programação: Claude Code.
- Ambiente de execução dos experimentos: Google Colab.
- Linguagem principal: Python.
- Formato de exploração e execução: Jupyter Notebooks.
- Versionamento: Git. Repositório público: https://github.com/Diehlgit/ProjetoFinalRNAP_UnB2026 (nome exigido pelo enunciado).
- Idioma do artigo e da documentação: português do Brasil.

Os notebooks serão mantidos no repositório local e executados utilizando os kernels disponibilizados pela extensão oficial do Google Colab no VS Code (v0.2.1 ou superior, que suporta `drive.mount()`).

Limitações conhecidas da extensão: Colab Secrets (`userdata.get`) e `files.upload()` não são suportados. Não depender de tokens ou segredos nos notebooks.

Ambiente local: `.venv` com `pip install -e ".[dev]"`, usado para testes e lint. As dependências estão em `pyproject.toml`.

## 4. Organização do projeto

O projeto está organizado em três ambientes complementares:

- **Repositório local:** contém o código-fonte, notebooks, configurações, testes e documentação, com versionamento pelo Git.
- **Google Drive:** armazena as imagens, os datasets, os checkpoints e os resultados dos experimentos.
- **Google Colab:** fornece o ambiente remoto de execução, incluindo os recursos computacionais necessários ao treinamento.

### 4.1. Estrutura do repositório local

- `configs/`: configurações dos datasets e experimentos.
- `data/manifests/`: manifestos CSV com informações sobre as imagens, espécies, licenças e metadados.
- `notebooks/`: notebooks Jupyter para exploração, execução e análise dos experimentos.
- `src/cerrado_flora/data/`: coleta, validação e preparação dos dados.
- `src/cerrado_flora/models/`: implementação das arquiteturas dos modelos.
- `src/cerrado_flora/training/`: rotinas de treinamento e validação.
- `src/cerrado_flora/evaluation/`: métricas e procedimentos de avaliação.
- `src/cerrado_flora/utils/`: funções e componentes auxiliares.
- `tests/`: testes automatizados.
- `reports/`: gráficos, tabelas e resultados consolidados.
- `docs/`: documentação técnica. `docs/decisoes.md` registra decisões tomadas e **pendentes**; `docs/uso-de-ia.md` registra o uso de IA no projeto.
- `paper/`: documento científico final, desenvolvido em LaTeX (IEEEtran, citações ABNT autor-data via `abntex2cite`). Contém o arquivo principal, referências bibliográficas, seções do artigo, figuras, tabelas e materiais suplementares. Compilação em `paper/.build/` (não versionado).
- `outputs/`: saídas locais e temporárias, não versionadas. Resultados oficiais ficam no Google Drive.

### 4.2. Organização dos dados no Google Drive

Os arquivos de dados e resultados serão armazenados fora do repositório Git, em uma pasta dedicada no Google Drive:

- `raw/`: imagens originais obtidas das fontes de dados.
- `interim/`: dados intermediários durante o processamento.
- `processed/`: imagens e datasets preparados para os experimentos.
- `checkpoints/`: pesos e estados dos modelos durante e após o treinamento.
- `results/`: métricas, predições e resultados dos experimentos.

### 4.3. Execução no Google Colab

Os notebooks serão mantidos no repositório local e executados por meio da extensão do Google Colab no VS Code.

O kernel remoto não enxerga os arquivos locais. O notebook `notebooks/00_setup_colab.ipynb` prepara o ambiente em dois modos:

- **`upload`** (desenvolvimento): `src/`, `configs/` e `pyproject.toml` enviados com *Upload to Colab* no Explorer do VS Code. Não há hash de commit rastreável.
- **`git`** (experimentos reportados no artigo): commit e push locais, seguidos de clone/pull do repositório público no Colab, sem token.

Durante a execução:

- Resultados que entram no artigo devem vir de execuções no modo `git`, com metadados salvos por `cerrado_flora.utils.run_metadata` (commit, pacotes, GPU, seed).
- O Google Drive será montado para acessar os dados persistentes.
- Os arquivos necessários ao treinamento poderão ser copiados para o armazenamento temporário da máquina do Colab, em `/content/`, para melhorar o desempenho de leitura.
- Checkpoints e resultados importantes deverão ser salvos no Google Drive, pois o armazenamento local do Colab é temporário.

### 4.4. Organização e reutilização do código

A implementação reutilizável deverá permanecer em `src/`, evitando duplicação de código nos notebooks.

Os notebooks devem ser utilizados principalmente para exploração dos dados, configuração e execução dos experimentos e análise dos resultados.

As configurações específicas de cada experimento devem ser mantidas em `configs/`, permitindo reproduzir e comparar diferentes execuções sem modificar desnecessariamente o código-fonte.

## 5. Regras de desenvolvimento

1. Não alterar o escopo científico sem discutir previamente a mudança.
2. Não assumir que uma espécie observada no Distrito Federal é necessariamente nativa do Cerrado.
3. Verificar a licença de cada imagem individualmente, e não apenas a licença da fotografia padrão da espécie.
4. Preservar atribuição e informações de origem das imagens.
5. Não incluir imagens, credenciais ou grandes arquivos binários no Git.
6. Garantir que os conjuntos de treinamento, validação e teste não tenham vazamento de imagens ou observações.
7. Utilizar as mesmas divisões de dados e condições experimentais na comparação entre modelos.
8. Documentar decisões metodológicas e configurações dos experimentos.
9. Não inventar resultados, referências bibliográficas ou informações sobre os dados.
10. Antes de realizar alterações significativas, apresentar um plano e aguardar aprovação.
11. Não assumir decisões listadas como pendentes em `docs/decisoes.md`; perguntar antes de implementá-las.
12. Registrar em `docs/uso-de-ia.md` cada contribuição relevante de ferramentas de IA (ferramenta, modelo, tarefa, o que foi aceito/modificado/rejeitado e como foi verificado).

## 6. Processo de trabalho

O desenvolvimento será dividido nas seguintes etapas:

1. Auditoria e seleção das imagens.
2. Validação das espécies e das licenças.
3. Construção do conjunto de dados.
4. Implementação e treinamento do baseline.
5. Implementação e treinamento do modelo com gating do GmNet (desenho exato pendente: ver `docs/decisoes.md`).
6. Avaliação comparativa.
7. Análise dos erros e elaboração dos resultados.

Cada etapa deve produzir artefatos verificáveis antes de avançarmos para a seguinte.

## 7. Entregáveis (enunciado do Projeto Individual 2)

1. Artigo no modelo IEEE conferência, duas colunas, 6–8 páginas incluindo figuras e referências. Seções: título; introdução (motivação, propósito, pergunta de pesquisa); trabalhos relacionados (estado da arte, artigos da ICLR 2026); metodologia (hipótese, técnicas base, inovação proposta, link do GitHub com o diretório `/ProjetoFinalRNAP_UnB2026`); experimentos; conclusões; referências completas em ABNT (AUTOR, ano).
2. Relatório de uso de IA (2 páginas), específico e detalhado, construído a partir de `docs/uso-de-ia.md`.
3. De 1 a 4 artigos correlatos da ICLR 2026, com a inovação deste trabalho sobre eles explicitada e demonstrada nos resultados.
4. Apresentação oral de 5 minutos com 4 slides, na aula seguinte à entrega no aprender3.unb.br.
