# Registro de uso de IA

Base para o relatório obrigatório (2 páginas) sobre o uso de modelos de IA no projeto.
Registrar **durante** o desenvolvimento: data, ferramenta e modelo, tarefa, o que foi
aceito, modificado ou rejeitado, e como o resultado foi verificado.

| Data | Ferramenta / modelo | Tarefa | Resultado (aceito / modificado / rejeitado) | Verificação |
|------|---------------------|--------|---------------------------------------------|-------------|
| 2026-11-23 | Claude Code (VS Code), Claude Opus 5.5 | Revisão da configuração do repositório frente ao enunciado e ao CLAUDE.md; levantamento de lacunas e decisões pendentes. | Aceito. | Leitura dos itens apontados. |
| 2026-11-23 | Claude Code (VS Code), Claude Opus 5.5 | Correções de configuração: branch `main`, `.gitkeep`/`__init__.py`, `pyproject.toml`, `.gitignore`, utilitário de metadados de execução, teste mínimo, notebook de setup do Colab, esqueleto do artigo com ABNT, documentação. | Modificado: proposta inicial de token do GitHub descartada pelo autor em favor de *Upload to Colab* / repositório público. | Teste `pytest`; compilação LaTeX; verificação da publicação do GmNet na ICLR 2026 e do suporte a `drive.mount` na extensão do Colab. |
| 2026-11-23 | Claude Code (VS Code), Claude Opus 5.5 | Renomeação do repositório GitHub para `ProjetoFinalRNAP_UnB2026` e mudança para público (via `gh`); atualização de URLs; verificação da disponibilidade de pesos e da licença do repositório oficial do GmNet. | Aceito. A verificação de pesos contradisse parcialmente um levantamento anterior do autor (há apenas `gmnet_s3.npy`); decisão de pré-treino mantida como pendente. | `gh repo view`; leitura do README e da pasta de checkpoints do GmNet. |
