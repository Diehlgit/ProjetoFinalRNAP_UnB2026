# Decisões metodológicas

Registro das decisões do projeto. Cada decisão tomada deve indicar data e justificativa.
Itens **pendentes** não devem ser assumidos na implementação até serem decididos.

## Decididas

| Data | Decisão | Justificativa |
|------|---------|---------------|
| 2026-11-23 | Artigo de referência: GmNet (WANG et al., 2026), ICLR 2026. Código oficial: github.com/YFWang1999/GmNet. | Publicação confirmada na ICLR 2026 (OpenReview / proceedings). |
| 2026-11-23 | Referências no formato ABNT (AUTOR, ano) via `abntex2cite` (`alf`) dentro do modelo IEEEtran. | Exigência do enunciado; compilação testada. |
| 2026-11-23 | Código no Colab: modo `upload` (Upload to Colab do VS Code) para desenvolvimento; modo `git` (clone do repositório público, sem token) para experimentos que entram no artigo. | Upload não deixa hash de commit rastreável; resultados reportados precisam de commit identificável. |
| 2026-11-23 | Google Drive montado via `drive.mount()` na extensão do Colab (suportado desde a v0.2.1). Colab Secrets (`userdata`) não são suportados pela extensão. | colab-vscode issue #256 e wiki "Known Issues". |
| 2026-11-23 | `outputs/` é apenas saída local/temporária, não versionada. Resultados oficiais ficam no Drive (`results/`); tabelas e figuras consolidadas em `reports/`. | Evitar ambiguidade entre os três locais. |
| 2026-11-23 | Repositório GitHub renomeado para `Diehlgit/ProjetoFinalRNAP_UnB2026` e tornado público. | Nome exigido pelo enunciado; acesso do professor; clone no Colab sem token. |
| 2026-11-23 | Idioma do artigo: português do Brasil. | Decisão do autor. |
| 2026-11-23 | A inovação sobre o GmNet será definida **após** a análise de desempenho no dataset montado. | Decidir com base em evidência do comportamento do modelo nos dados do Cerrado. |

## Pendentes

### Entrega
- [ ] **Prazos**: data de entrega dos relatórios e da apresentação (4 slides, 5 min).
- [ ] **Artigos correlatos adicionais da ICLR 2026** (até 4 no total).

### Modelos e comparação
- [ ] **Baseline**: qual arquitetura leve (p.ex. StarNet, MobileNetV3, EfficientNet-B0).
- [ ] **Desenho do experimento**: (a) baseline com o bloco de gating do GmNet substituído/inserido, isolando o efeito; (b) GmNet completo vs. baseline; ou ambos. (A seção 2 do CLAUDE.md descreve (a) e a etapa 5 da seção 6 descreve (b).)
- [ ] **Pré-treino**: ImageNet ou do zero — deve ser o mesmo para todos os modelos comparados.
  - Intenção do autor (2026-11-23): treinar do zero seguindo a arquitetura proposta, por um levantamento anterior indicar ausência de pesos públicos.
  - Verificação (2026-11-23): o README do repositório oficial aponta uma pasta no Google Drive (`GmNet_checkpoints`) que contém **apenas `gmnet_s3.npy` (30,1 MB)**; não há pesos de S1/S2. Formato `.npy` e conteúdo (ImageNet-1K?) ainda não verificados.
  - Consequências a considerar: (i) com treino do zero, o baseline também deve ser treinado do zero (regra 7); (ii) modelos treinados do zero em datasets pequenos tendem a ter desempenho bem inferior aos pré-treinados, o que pode mascarar o efeito do gating; (iii) usar o S3 pré-treinado exigiria um baseline pré-treinado de porte comparável.
- [ ] **Licença do código do GmNet**: o repositório `YFWang1999/GmNet` não declara licença (verificado em 2026-11-23). Definir se o modelo será reimplementado a partir do artigo ou se o código será reutilizado com citação explícita.
- [ ] **Critério de "leve"**: limite de parâmetros/FLOPs; reportar latência?
- [ ] **Inovação explícita** sobre o artigo (adiada para após a análise de desempenho; ideias: sensibilidade a filtros passa-baixa/passa-alta, espectro das ativações com e sem gating).

### Dados
- [ ] Número de espécies e mínimo de imagens por espécie.
- [ ] Filtro de qualidade do iNaturalist (apenas *research grade*?).
- [ ] Fonte e critério de validação de nativas do Cerrado (p.ex. Flora e Funga do Brasil/REFLORA; nativa com ocorrência no domínio vs. endêmica).
- [ ] Licenças aceitas (CC0, CC-BY, CC-BY-NC?).
- [ ] Unidade de divisão treino/validação/teste (observação; agrupar também por observador/local?) e proporções.
- [ ] Uso de múltiplas fotos por observação.

### Avaliação
- [ ] Métricas (top-1, macro-F1, acurácia balanceada).
- [ ] Número de seeds e teste estatístico.
- [ ] Registro de experimentos (CSV, TensorBoard ou W&B).
- [ ] Orçamento de GPU (Colab gratuito ou Pro).
