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
| 2026-11-23 | **Pré-treino: usar os pesos ImageNet-1K do GmNet-S3** (`gmnet_s3.npy`) com *fine-tuning* no dataset do Cerrado. O baseline também deverá partir de pesos ImageNet-1K, com porte comparável (~7–8 M parâmetros, ~1 GMAC). | Pesos públicos verificados (ver análise abaixo); em datasets pequenos o pré-treino tende a ser decisivo, e o treino do zero poderia mascarar o efeito do gating. Regra 7: mesmas condições para todos os modelos. |
| 2026-11-23 | Pesos pré-treinados de terceiros ficam no Google Drive (`<drive_root>/pretrained/`), fora do Git; procedência, checksum SHA-256 e configuração de arquitetura em `configs/pretrained.yaml`. | Regra 5 (sem binários grandes no Git); o checksum garante que a cópia usada é a verificada. |
| 2026-11-24 | **Fonte e recorte**: iNaturalist, Distrito Federal (`place_id` 12591), Plantae, apenas *research grade*, observações criadas até 2026-11-23 (`created_d2`). | Escopo da seção 2 mantido (só DF). *Research grade* exige concordância da comunidade na espécie e exclui plantas cultivadas. A data de corte limita a deriva da consulta. |
| 2026-11-24 | **Licenças aceitas**: CC0, CC-BY e CC-BY-NC, verificadas **foto a foto**; atribuição de cada foto no manifesto. | Uso acadêmico não comercial; imagens não são redistribuídas (Drive privado). Só CC0/CC-BY daria ~1,6 mil observações no DF, insuficiente. |
| 2026-11-24 | **Nativas do Cerrado**: Flora e Funga do Brasil (DwC-A do IPT/JBRJ, versão 393.433, SHA-256 no config), `establishmentMeans` = Nativa e Cerrado entre os domínios fitogeográficos (não exige endemismo). Nomes sem correspondência única vão para revisão manual e ficam fora até serem resolvidos em `native.manual`. | Fonte oficial e versionada; critério reproduzível. |
| 2026-11-24 | **Espécie sem distribuição na Flora e Funga** (sem domínio ou estabelecimento no registro da espécie): herda dos infraespecíficos aceitos — nativa só se **todos** forem "Nativa"; domínios e estados = união. Sem informação em nenhum nível → revisão manual (nunca exclusão automática). Caso que motivou: *Calliandra dysantha* (registro da espécie vazio; as 4 variedades são nativas do Cerrado). | Ausência de dado na fonte não é evidência de não ser nativa; a regra é geral e reproduzível, ao contrário de exceções manuais. |
| 2026-11-24 | ***Miconia fallax* mantida fora**: a Flora e Funga (v. 393.433) lista só "Amazônia" como domínio, embora registre ocorrência em DF, GO, MG, MS, MT e TO. Aplicado o critério como está; a inconsistência deve ser mencionada no artigo. | Evitar julgamentos *ad hoc* sobre a fonte de referência. |
| 2026-11-24 | **Classes**: espécies com ≥ 50 observações elegíveis; **limite de 150 observações** por classe, sorteadas com semente fixa. | No DF, ~35 espécies atingem 50 observações. O limite reduz o desbalanceamento (a maior tinha 334). |
| 2026-11-24 | **Fotos por observação**: todas as fotos elegíveis; a observação é a unidade de divisão. | Mais imagens sem vazamento entre fotos da mesma planta. |
| 2026-11-24 | **Divisão**: 70/15/15 **por classe** (cada espécie dividida separadamente, com ≥ 1 observação em validação e teste), semente 2026. Observações com fotos quase idênticas (dHash, distância ≤ 4) ficam no mesmo grupo e na mesma divisão; quase-duplicatas entre classes diferentes são descartadas. Sobreposição de observadores entre divisões é **relatada, não imposta**. | Garante a presença de todas as classes em todas as divisões. Impor a separação por observador poderia esvaziar classes dominadas por poucos observadores. |
| 2026-11-24 | **Baseline: StarNet-S4** (MA et al., 2024, CVPR 2024), pesos ImageNet-1K oficiais (release `checkpoints_v1`, Apache-2.0); código copiado em `src/cerrado_flora/models/starnet.py`. Checkpoint inspecionado estaticamente: 7.482.728 parâmetros, profundidades [3, 3, 12, 5], salvo na época 296 com top-1 de 78,84% no ImageNet; pickle só com globais de tensores e `argparse.Namespace`. Procedência e SHA-256 em `configs/pretrained.yaml`. | Antecessor direto do GmNet (o código do GmNet deriva do da StarNet), porte comparável (7,5 M vs. 7,82 M parâmetros; 1,08 vs. 1,24 GFLOPs), pesos públicos verificados. |
| 2026-11-24 | **Desenho do experimento: (b)** GmNet-S3 completo vs. StarNet-S4, com fine-tuning idêntico (configs diferem só no modelo, verificado por `tests/test_configs.py`). **Limitação**: além do gating (StarNet: `ReLU6(f1(x)) * f2(x)`; GmNet: `ReLU6(f1(x)) * f1(x)`), diferem larguras/profundidades, BN e *layer scale* no bloco e a receita de pré-treino no ImageNet (78,8% vs. 81,3%); a comparação não isola o efeito do gating, o que deve constar no artigo. | Só há pesos ImageNet públicos dos modelos completos; isolar o gating (desenho (a)) exigiria pré-treino no ImageNet. |
| 2026-11-24 | **Métricas**: top-1, macro-F1 e acurácia balanceada, por foto e por observação (média das probabilidades das fotos). Checkpoint escolhido pelo **macro-F1 por foto na validação**; teste avaliado uma única vez, com o checkpoint escolhido. | Classes de 50 a 150 observações: o macro-F1 pesa as classes pequenas igual às grandes. |
| 2026-11-24 | **Sementes**: 5 por modelo (2026–2030), iguais para GmNet-S3 e StarNet-S4. Antes havia só 2026. | Com 1 semente, a diferença observada (+1,8 pt de top-1 por foto e +0,5 pt por observação no teste) ficou dentro da oscilação entre épocas (~±1 pt); é preciso estimar a variação entre execuções. |
| 2026-11-24 | **Registro de experimentos**: CSV/JSON no Drive (`results/<exp>/seed_<n>/`: histórico por época, métricas, predições, matriz de confusão, metadados da execução); checkpoints em `checkpoints/<exp>/seed_<n>/`. | Sem contas nem tokens (Colab Secrets não funcionam na extensão do VS Code). |
| 2026-11-24 | **Testes que usam PyTorch rodam no Colab** (notebook 02, seção 2); o `.venv` local não tem PyTorch e roda só os demais testes. | Decisão do autor. |
| 2026-11-24 | **Reprodutibilidade do dataset**: o dataset reportado é o congelado em `data/manifests/images.csv` (IDs, URLs, divisão, SHA-256 de cada foto), versionado no Git. `python -m cerrado_flora.data fetch` recria as imagens a partir do manifesto e verifica os checksums; `build` reaplica o procedimento de `configs/dataset.yaml`. | O iNaturalist muda com o tempo; reexecutar a consulta não reproduz o mesmo conjunto. |
| 2026-11-23 | Código do GmNet **copiado do repositório oficial** (que não declara licença), com citação, em `src/cerrado_flora/models/gmnet.py`; alterações documentadas no cabeçalho do arquivo. | Uso exclusivamente acadêmico; reimplementar produziria o mesmo código e arriscaria divergências. |
| 2026-11-24 | **Enquadramento após as 5 sementes**: GmNet-S3 e StarNet-S4 são tratados como **comparáveis** (teste, macro-F1 por foto 0,851 ± 0,006 vs. 0,836 ± 0,012; por observação 0,927 ± 0,005 vs. 0,920 ± 0,016). O foco passa a ser **quanto o gating contribui** para o desempenho do GmNet. | Diferença pequena (~1 pt) e não estabelecida estatisticamente (sinal 5/5 por foto, p = 0,0625 bilateral; por observação o sinal varia); além disso a comparação (b) não isola o gating. Decisão do autor. |
| 2026-11-24 | **Ablação do gating (desenho (a))**: GmNet-S3 **treinado do zero** com e sem gating (`gmnet_s3_scratch.yaml` / `gmnet_s3_nogate_scratch.yaml`, diferem só em `model.gating`). Sem gating, o bloco calcula `ReLU6(x)` em vez de `ReLU6(x) * x`; o gating não tem parâmetros, então com a mesma semente os pesos iniciais são idênticos. Notebook 04. **Limitação**: mede a contribuição do gating sem pré-treino no ImageNet, com ~4,3 mil fotos de treino; não equivale à contribuição após o pré-treino. | Não há pesos ImageNet de um GmNet sem gating; partir dos pesos do GmNet e remover o gating mediria "retirar o gating de uma rede treinada com ele", não "gating vs. sem gating". |
| 2026-11-24 | **Receita do zero (provisória)**: 100 épocas, warmup 10, LR 1e-3; demais hiperparâmetros e aumentos iguais aos do fine-tuning. **Piloto** com 1 semente (2026) por variante antes de decidir o número de sementes. | Do zero são necessárias mais épocas e LR maior; o piloto verifica a convergência antes de gastar GPU. Ajustes valem igualmente para as duas variantes. |
| 2026-11-26 | **Piloto da ablação do zero posto de lado** (notebook 04, semente 2026): teste macro-F1 por foto 0,553 com gating vs. 0,549 sem; por observação 0,687 vs. 0,701. As duas variantes sobreajustam (perda de treino ~0,77, próxima do mínimo de ~0,64 com *label smoothing*; validação estável em ~2,0 desde a época ~20). | Com ~4,3 mil fotos de treino, os modelos do zero são fracos demais para servir de medida do gating. Decisão do autor: não usá-los na análise de frequência; retreinar com mais dados fica como caminho futuro. |
| 2026-11-26 | **Análise de sensibilidade a frequências** (notebook 05, `evaluation/frequency.py`) dos modelos de fine-tuning (GmNet-S3 e StarNet-S4, 5 sementes cada, `best.pt`), sem treino. Protocolo fixado antes dos resultados: teste com o pré-processamento de avaliação; filtro no espaço de pixels antes da normalização; FFT 2D com máscara circular ideal; passa-baixa (≤ *r*) e passa-alta (> *r*, mantendo a média de cada canal), *r* ∈ {4, 8, 16, 32, 64, 112}, sem recorte de valores; medida principal: macro-F1 por foto relativo à imagem original, média ± desvio entre sementes, resumido pela média entre os raios; por espécie na condição passa-baixa *r* = 16. Conferência: sem filtro, deve reproduzir o `metrics.json` do treino. **Limitações**: imagens filtradas fora da distribuição (comparar entre modelos, não valores absolutos); GmNet e StarNet diferem em mais do que o gating. | A hipótese trata do uso de frequências (texturas, bordas); acurácia igual não implica uso igual. Reaproveita os modelos treinados (custo de minutos de GPU). Os modelos do zero ficaram de fora (ver linha anterior); a inspeção interna dos mapas de ativação foi adiada pelo autor. |

## Pendentes

### Entrega
- [ ] **Prazos**: data de entrega dos relatórios e da apresentação (4 slides, 5 min).
- [ ] **Artigos correlatos adicionais da ICLR 2026** (até 4 no total).

### Modelos e comparação
- [ ] **Ablação do gating do zero com mais dados** (caminho futuro; p. ex. ampliar além do DF com menos classes, para mais imagens por classe). O piloto com o dataset atual sobreajustou.
- [ ] **Espectro dos mapas de ativação** antes/depois do gating dentro dos blocos (adiado pelo autor).
- [ ] **Critério de "leve"**: limite de parâmetros/FLOPs; reportar latência?
- [ ] **Inovação explícita** sobre o artigo. Candidata: a análise de sensibilidade a frequências (notebook 05) em plantas do Cerrado após fine-tuning; decisão final do autor depois dos resultados.

### Avaliação
- [ ] Teste estatístico (adiado pelo autor em favor da ablação do gating; proposta: pareado por observação no teste, por semente, e pareado entre sementes; vale para (b) e para a ablação).
- [ ] Orçamento de GPU (Colab gratuito ou Pro).
- [ ] Hiperparâmetros de fine-tuning: os de `configs/experiments/gmnet_s3_ft.yaml` são um ponto de partida, a revisar após a primeira execução.

## Análise do checkpoint `gmnet_s3.npy` (2026-11-23)

Fonte: pasta `GmNet_checkpoints` no Google Drive, apontada pelo README de `YFWang1999/GmNet`
(arquivo de 31.525.414 bytes; ID do Drive `1LgS6biHIOutFBCYMJDVO876PhQWgy9fa`).

- **Formato**: `.npy` com array de objeto (pickle protocolo 3) contendo um `dict` com 535 tensores (`state_dict` do PyTorch convertido para NumPy). A varredura estática do pickle (`pickletools`) mostra apenas os globais `numpy.core.multiarray._reconstruct`, `numpy.ndarray` e `numpy.dtype`, ou seja, nenhuma chamada executável. Carregar com um `Unpickler` restrito a esses três globais, nunca com `np.load(..., allow_pickle=True)` sem verificação.
- **Conteúdo**: 7,82 M de parâmetros; cabeça `head` de 1000 classes (ImageNet-1K); estatísticas de BN treinadas (`num_batches_tracked` = 132.500 em todas as camadas); sem NaN/Inf.
- **Divergência com o código público**: o checkpoint **não carrega** em `gmnet_s3()` do arquivo `gment.py` oficial, que usa `f12_bn=False` e `dwconv2_bn=False`. O checkpoint tem BN em `f1` e `dwconv2`. Carrega com `strict=True` usando:
  `Model(embed_dim=[48, 96, 192, 384], depths=[3, 3, 8, 3], mlp_ratio=[4, 4, 4, 4], kernel_size=7, f12_bn=True, g_bn=True, dwconv2_bn=True, act=nn.ReLU6, downsampler_act=nn.Identity, block=Block)`.
  Atenção: `gmnet_s3(pretrained=True)` também não serve, porque converte as convoluções 1×1 em `Linear` e muda os nomes das chaves.
- **Custo**: 1,24 GMACs com entrada 224×224 (medido com `torch.utils.flop_counter`).
- **Teste de sanidade** em 200 imagens do ImageNet (repositório `EliSchwartz/imagenet-sample-images`, 1 imagem por classe, a cada 5 classes; pré-processamento redimensionar 256 bicúbico + recorte central 224 + normalização ImageNet):

  | Configuração | Top-1 | Top-5 |
  |---|---|---|
  | ReLU6 (código oficial) + flags de BN acima | 86,5% | 98,5% |
  | ReLU | 4,5% | 6,0% |
  | Controle: pesos aleatórios | 0,0% | 1,0% |

  O resultado é compatível com os 81,3% reportados para o GmNet-S3 (amostra pequena e com imagens "típicas"; não substitui a avaliação no conjunto de validação). Confirma também que a ativação do gating é ReLU6.
- **Gating do bloco** (código oficial): `x = ReLU6(f1(dwconv(x))) * f1(dwconv(x))`, seguido de `g`, `dwconv2`, *layer scale* `gamma` e conexão residual.
