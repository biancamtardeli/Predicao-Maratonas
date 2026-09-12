# Predição de Desempenho de Equipes em Maratonas de Programação
### IFSULDEMINAS – Campus Muzambinho

Código de apoio ao TCC de Bianca Martins Tardeli (orientação: Prof. Ricardo
José Martins). Implementa o pipeline descrito no projeto de pesquisa:
limpeza dos dados → engenharia de atributos históricos → agregação por
equipe → treinamento e avaliação de modelos de classificação supervisionada.

---

## 1. Como rodar

```bash
pip install -r requirements.txt
python main.py
```

Isso executa o pipeline inteiro e salva os artefatos em `outputs/`:

| Arquivo | Conteúdo |
|---|---|
| `relatorio_nomes_reconciliados.csv` | Pares de nomes com pequenas diferenças de digitação que foram unificados (**revisar manualmente**) |
| `eventos_equipe_rotulados.csv` | Uma linha por equipe×evento, com o rótulo `alto_desempenho` |
| `participacoes_com_atributos.csv` | Uma linha por aluno×evento, com as 3 features históricas |
| `equipes_features.csv` | Tabela final usada no treino (uma linha por equipe×evento) |
| `metricas_validacao_cruzada.csv` | Comparação dos modelos via validação cruzada estratificada |
| `metricas_holdout_temporal.csv` | Avaliação "no mundo real" (treino no passado, teste no ano-holdout) |
| `importancia_atributos.csv` | Quais atributos mais pesam na decisão do modelo |

Para simular uma equipe hipotética antes da inscrição:

```bash
python exemplo_predicao.py
```

(edite os nomes e a olimpíada-alvo dentro do arquivo, ou use
`src/predict.py::carregar_preditor()` a partir de outro script/notebook).

## 2. Estrutura do código

```
src/
  config.py               -> todos os parâmetros ajustáveis (limiares, regras)
  data_processing.py      -> Fase 1: limpeza do CSV bruto
  name_reconciliation.py  -> corrige erros de digitação em nomes
  team_labels.py          -> Fase A: rótulo "alto desempenho" por equipe/evento
  individual_features.py  -> Fase B: as 3 features históricas por aluno
  team_features.py        -> Fase C: agregação em nível de equipe
  train.py                -> Fase D: treino, CV estratificada, holdout temporal
  predict.py               -> Fase E: predição de equipes hipotéticas
main.py                    -> roda o pipeline inteiro e salva outputs/
exemplo_predicao.py        -> exemplo de uso do preditor treinado
```

Cada fase é uma função pura (recebe DataFrame(s), devolve DataFrame(s)),
o que facilita tanto testar isoladamente quanto explicar cada etapa no
texto do TCC.

---

## 3. Respostas aos pontos levantados pela banca

### (1) Incorporar a OLIP 2026 como teste "no mundo real"

A base atual (532 registros) **ainda não contém a OLIP 2026** — ela é
justamente o próximo evento do cronograma do projeto (a coleta de dados
vai até dezembro/2026). O pipeline já está preparado para o que a banca
sugeriu: em `config.py` existe o parâmetro

```python
ANO_TESTE_HOLDOUT = 2025   # trocar para 2026 assim que a OLIP 2026 acontecer
```

A função `train.separar_treino_teste_temporal()` treina o modelo usando
**apenas eventos anteriores** a esse ano e testa **exatamente** nos
eventos daquele ano — ou seja, o modelo nunca "vê" o futuro durante o
treino. Assim que os resultados da OLIP 2026 forem cadastrados na
planilha, basta atualizar esse parâmetro para `2026` e rodar `python
main.py` novamente: o relatório `metricas_holdout_temporal.csv` passará a
refletir exatamente a validação sugerida pela banca. Enquanto isso, o
holdout está configurado em 2025 (último ano completo disponível) como
"ensaio" da metodologia.

### (2) Como o limiar de "alto desempenho" é definido

Esse foi o ponto que exigiu mais cuidado durante a implementação. Uma
análise exploratória da coluna `Classificação` mostrou que ela tem **duas
naturezas diferentes** dependendo da competição:

- **Competições internas ao campus** (OLIP, Escola de Inverno): a
  classificação é a posição relativa **só entre as equipes do
  IFSULDEMINAS-Muzambinho** que disputaram aquele evento. Nessas provas
  há muitos empates de posição (várias equipes registradas como "1º
  lugar"), então um corte fixo "top 3" chegava a rotular praticamente
  toda a base como "alto desempenho" em alguns anos — um rótulo
  inútil para o modelo aprender qualquer coisa. Por isso, para essas
  competições o critério é **posição 1** (a equipe vencedora daquela
  edição).
- **Competições externas/multi-institucionais** (Regional, Mineira,
  Nacional, Minas de Minas, InterIF): a classificação já é um **rank
  absoluto**, disputado com dezenas de equipes de outras instituições
  que não estão nesta base. Nesses casos usamos o critério clássico de
  medalha do ICPC/SBC: **posição ≤ 3 (ouro/prata/bronze)**.

Além disso, para as competições internas, um evento só é considerado
válido para treino se tiver um número mínimo de equipes do campus
competindo (`N_MIN_EQUIPES_PARA_PODIO = 4`, ajustável em `config.py`) —
com menos que isso, qualquer corte é praticamente arbitrário. Eventos que
não atendem a esse critério (ou que não têm classificação registrada)
recebem `evento_valido = False` e são automaticamente excluídos do
treino, mas continuam na planilha para fins de auditoria.

Toda essa lógica está centralizada e documentada em `config.py`
(`LIMIAR_POR_OLIMPIADA`) e implementada em `team_labels.py`, para que
qualquer recalibração futura (por exemplo, se a Regional passar a ter
poucas equipes concorrentes e o corte top-3 deixar de fazer sentido)
seja só uma mudança de parâmetro, não de código.

Com essas regras, a base resultante tem **130 equipes-evento válidas**,
das quais **19,2% são "alto desempenho"** — uma taxa desbalanceada, mas
plausível (nem toda equipe medalha, o que é justamente o comportamento
esperado de uma competição real).

### (3) Classes desbalanceadas

O pipeline trata o desbalanceamento em duas frentes complementares:

1. **`StratifiedKFold`** na validação cruzada (`train.validacao_cruzada`):
   garante que toda dobra de treino/validação contenha exemplos das
   duas classes, evitando o cenário em que uma dobra fica sem nenhuma
   equipe de alto desempenho. O número de dobras é automaticamente
   limitado ao tamanho da classe minoritária, para nunca gerar uma
   dobra vazia.
2. **`class_weight="balanced"`** nos modelos baseados em árvore
   (RandomForest, XGBoost): penaliza mais os erros na classe
   minoritária, sem precisar inventar dados.
3. **SMOTE** (oversampling sintético, biblioteca `imbalanced-learn`)
   está disponível como alternativa opcional (`usar_smote=True` nas
   funções de `train.py`) e é aplicado **somente dentro dos folds de
   treino** (nunca no teste), para não vazar informação sintética para
   a avaliação.

Recomenda-se comparar, no texto do TCC, os resultados com e sem SMOTE —
com uma base de ~130 equipes e ~25 positivos, o ganho do SMOTE tende a
ser modesto e vale a pena reportar isso como uma limitação honesta do
tamanho atual da base (que deve crescer com a OLIP 2026 e edições
futuras).

### (4) Predição por equipe, por nomes, e ponderação por competição

Sim — o modelo final (`predict.PreditorDeEquipe.prever`) recebe **os
nomes dos integrantes** e a **olimpíada-alvo**, exatamente como descrito
no objetivo do projeto ("testar combinações de perfis antes da
inscrição"). Internamente:

- Cada aluno é localizado no histórico (com tolerância a pequenas
  diferenças de digitação, via normalização de acentos + fusão de nomes
  muito parecidos em `name_reconciliation.py`); alunos nunca vistos
  entram como "estreantes" (todas as features de experiência zeradas).
- A **olimpíada é uma variável categórica de entrada do modelo**
  (one-hot encoding), então a mesma equipe recebe probabilidades
  **diferentes** dependendo da competição-alvo — o exemplo em
  `exemplo_predicao.py` mostra isso na prática: a mesma equipe fictícia
  tem ~54% de chance estimada de bom desempenho na OLIP (competição
  interna, mais fácil) contra ~21% no Regional (competição externa,
  mais difícil), o que é coerente com a dificuldade real de cada etapa.

Já as competições **individuais** (Algar e OBI) **não formam equipe**
— não faz sentido pedir ao modelo "a equipe X e Y vai bem no Algar",
porque cada aluno compete sozinho. A participação nessas provas **não é
descartada**: ela entra como parte da "bagagem de nível superior" de
cada aluno (`bagagem_nivel_superior`), contribuindo para as features de
qualquer equipe da qual esse aluno venha a fazer parte depois. Se no
futuro for do interesse do projeto também prever o desempenho
*individual* no Algar/OBI, o código já separa esses eventos
(`eh_individual=True` em `data_processing.py`) e um segundo modelo
(a nível de aluno, não de equipe) pode ser construído reaproveitando as
mesmas 3 features individuais já calculadas — é uma extensão natural,
mas fora do escopo do modelo principal de formação de equipes.

---

## 4. Limitações conhecidas (documentar no texto do TCC)

- **Base ainda pequena** para o padrão de Machine Learning (130 eventos
  válidos, 25 positivos): os resultados de validação cruzada devem ser
  interpretados com cautela e tendem a melhorar à medida que novas
  edições (a começar pela OLIP 2026) forem incorporadas.
- **Ordem cronológica dentro do mesmo ano é aproximada** (não há mês
  exato de cada evento na planilha); usamos uma ordem "típica" por tipo
  de olimpíada (`config.ORDEM_APROXIMADA_NO_ANO`). Se houver como
  recuperar a data exata de cada edição, isso pode ser refinado.
- **Reconciliação de nomes é heurística** (similaridade de string com
  limiar alto): o relatório em `outputs/relatorio_nomes_reconciliados.csv`
  deve ser revisado manualmente antes da versão final do trabalho.
- **XGBoost é opcional**: se não estiver instalado, o pipeline roda
  normalmente só com RandomForest e GradientBoosting (avisando no
  console). Para reproduzir a metodologia completa do projeto de
  pesquisa, instale com `pip install xgboost`.
