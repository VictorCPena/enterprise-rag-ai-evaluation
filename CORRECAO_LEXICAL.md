# Correção da abstenção lexical — 06/11/2026

## Comportamento corrigido

Antes, palavras coincidentes em um bloco inteiro podiam autorizar a resposta, mesmo quando a citação escolhida não sustentava a consulta. A tokenização também separava `won't` em partes e podia produzir uma correspondência indevida com `won`.

Agora cada trecho citado precisa conter pelo menos 40% dos termos de conteúdo da consulta, com pelo menos dois termos correspondentes (um para consultas com um único termo). Todos os números explícitos pedidos precisam aparecer nesse mesmo trecho. Contrações são preservadas. A busca bruta continua inspecionável; a resposta se abstém quando nenhum trecho atende à regra. As posições das citações continuam referenciando o documento original. O caminho de geração recebe somente trechos aprovados e não chama o gerador quando não há evidência suficiente.

Essa política se aplica às três estratégias. É uma heurística lexical conservadora, não uma prova de pertinência semântica. Pode recusar paráfrases corretas, especialmente quando os termos aparecem em frases diferentes. O limiar foi examinado em dados de desenvolvimento e não possui garantia estatística de generalização.

## Evidências executadas em Linux/Python 3.12

- Linha de base: 10 testes antigos aprovados; os 12 novos testes de regressão falhavam antes da alteração.
- Depois: 22 testes aprovados, Ruff sem erros e pip check sem dependências quebradas.
- Pipeline completo: indexação dos 12 documentos públicos e avaliação com hashing.
- Conjunto original de desenvolvimento: Recall@5 de 1,0 nas três estratégias. A lexical passou de 3/4 para 4/4 abstenções nas negativas, mantendo respostas nas 12/12 positivas. Vetorial e híbrida se abstiveram nas 4/4 negativas e responderam a 11/12 positivas cada. Essas duas abstenções em positivas são registradas como limitações.
- Sondagens de regressão: 8/8 negativas recusadas nas três estratégias. Nas oito positivas com outras formulações, lexical respondeu a 6/8, vetorial a 5/8 e híbrida a 6/8. Não apresentamos esse conjunto escrito após o diagnóstico como teste cego.
- HTTP real: 72 requisições retornaram HTTP 200, sem erros, com recuperação após reiniciar. Esse ensaio curto não demonstra capacidade de produção.
- Um aviso de depreciação do cliente de testes FastAPI/Starlette permanece; não houve falha nas asserções.

O modo neural e o LLM real não foram reexecutados nesta correção; o controle anterior à geração foi exercitado com um gerador de teste. A execução posterior no Mac Intel aprovou 23 testes com hashing e o índice de 12 documentos. Veja o [registro do Mac](evidence/citation-dedup-20261109/mac-run/result.json).

## Comandos utilizados

```bash
python -m pytest tests/test_pro_rag.py -q
python -m pytest tests/test_evidence_abstention.py -q  # falhava antes da correção
python -m pytest -q
python -m ruff check rag_platform tests/test_pro_rag.py tests/test_evidence_abstention.py
python -m rag_platform.demo --model hashing
python -m rag_platform.abstention_regression
EMBED_MODEL=hashing python scripts/load_api.py --module rag_platform.api:app
python -m pip check
```

Os JSONs datados `20261106` em `evidence/` registram as execuções daquela correção. `abstention_regression.json` contém todas as sondagens, inclusive as positivas recusadas. O ambiente de QA está em `evidence/lexical_fix_environment_20261106.txt`. A correção não altera dependências; não foi feita uma nova auditoria de vulnerabilidades.

## Reproduzir localmente no Mac Intel

A correção já integra esta cópia. Depois do [setup](README.md#intel-mac-local-profile), execute na raiz do projeto:

```bash
EMBED_MODEL=hashing python -m pytest -q
python -m rag_platform.demo --model hashing
python -m rag_platform.abstention_regression
EMBED_MODEL=hashing python -m uvicorn rag_platform.api:app --host 127.0.0.1 --port 8001
```

Abra [a interface](http://127.0.0.1:8001/). Em lexical, teste `Who won the 2099 World Cup?`
para verificar a abstenção. Para um caso positivo com citações, teste
`How do I connect to a SQLite database and execute SQL statements?`.
Para repetir o ensaio HTTP automatizado, use o comando de carga acima.
