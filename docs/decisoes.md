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
| 2026-11-23 | Código do GmNet **copiado do repositório oficial** (que não declara licença), com citação, em `src/cerrado_flora/models/gmnet.py`; alterações documentadas no cabeçalho do arquivo. | Uso exclusivamente acadêmico; reimplementar produziria o mesmo código e arriscaria divergências. |

## Pendentes

### Entrega
- [ ] **Prazos**: data de entrega dos relatórios e da apresentação (4 slides, 5 min).
- [ ] **Artigos correlatos adicionais da ICLR 2026** (até 4 no total).

### Modelos e comparação
- [ ] **Baseline**: arquitetura leve **com pesos ImageNet-1K disponíveis** e porte comparável ao GmNet-S3 (7,82 M parâmetros, 1,24 GMACs). Candidato natural: StarNet-S4, antecessor direto do GmNet (disponibilidade dos pesos a verificar). Outros: MobileNetV3, EfficientNet-B0 (menores).
- [ ] **Desenho do experimento**: (a) baseline com o bloco de gating do GmNet substituído/inserido, isolando o efeito; (b) GmNet completo vs. baseline; ou ambos. (A seção 2 do CLAUDE.md descreve (a) e a etapa 5 da seção 6 descreve (b).)
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
