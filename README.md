# Desafio Prompt Engineer - Bug to User Story

Otimização de prompts usando LangChain e LangSmith para converter relatórios de bugs em User Stories de alta qualidade.

## Objetivo

Criar um software capaz de:

- Fazer pull de prompts do LangSmith Prompt Hub
- Refatorar e otimizar usando técnicas avançadas de Prompt Engineering
- Fazer push dos prompts otimizados de volta ao LangSmith
- Avaliar a qualidade através de métricas customizadas
- Atingir pontuação mínima de 0.9 (90%) em todas as métricas

## Pré-requisitos

- Python 3.9+
- Conta no LangSmith
- API Key da OpenAI ou Google (Gemini)

## Instalação

```bash
# Criar ambiente virtual
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Instalar dependências
pip install -r requirements.txt
```

## Configuração

Crie um arquivo `.env` com suas credenciais:

```env
LANGCHAIN_API_KEY=your_langsmith_api_key
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=desafio-prompt-engineer

# Escolha um provider
OPENAI_API_KEY=your_openai_api_key
# ou
GOOGLE_API_KEY=your_google_api_key
```

## Como Executar

### 1. Pull dos prompts iniciais

```bash
python src/pull_prompts.py
```

### 2. Refatorar prompts

Edite o arquivo `prompts/bug_to_user_story_v2.yml` aplicando as técnicas de prompt engineering.

### 3. Push dos prompts otimizados

```bash
python src/push_prompts.py
```

### 4. Avaliação

```bash
python src/evaluate.py
```

### 5. Testes

```bash
pytest tests/test_prompts.py
```

## Estrutura do Projeto

```
desafio-prompt-engineer/
├── .env.example              # Template das variáveis de ambiente
├── requirements.txt          # Dependências Python
├── README.md                 # Documentação
│
├── prompts/
│   ├── bug_to_user_story_v1.yml       # Prompt inicial (após pull)
│   └── bug_to_user_story_v2.yml       # Prompt otimizado
│
├── src/
│   ├── pull_prompts.py       # Pull do LangSmith
│   ├── push_prompts.py       # Push ao LangSmith
│   ├── evaluate.py           # Avaliação automática
│   ├── metrics.py            # 4 métricas implementadas
│   ├── dataset.py            # 15 exemplos de bugs
│   └── utils.py              # Funções auxiliares
│
└── tests/
    └── test_prompts.py       # Testes de validação
```

## Métricas de Avaliação

Todas as métricas devem atingir >= 0.9:

- **Tone Score**: Avalia o tom e linguagem
- **Acceptance Criteria Score**: Avalia critérios de aceitação
- **User Story Format Score**: Avalia o formato da User Story
- **Completeness Score**: Avalia a completude

## Técnicas Aplicadas

Seis técnicas avançadas de Prompt Engineering foram aplicadas para otimizar a conversão de bug reports em User Stories:

### Técnicas Principais:

1. **Role Prompting**: Atribuição de persona específica ao LLM.
   - _Implementação_: "Você é um Product Manager Sênior empático que transforma bugs em User Stories de alta qualidade."
   - _Justificativa_: Um PM experiente entende tanto o lado técnico quanto as necessidades do usuário, sendo ideal para essa transformação.

2. **Few-Shot Learning**: Fornecimento de exemplos input/output detalhados.
   - _Implementação_: Dois exemplos completos (bug de login mobile e bug de pagamento Amex) com User Stories empáticas e critérios Given-When-Then.
   - _Justificativa_: Exemplos concretos demonstram o tom, formato e nível de detalhe esperados.

3. **Chain of Thought**: Instruções passo a passo para raciocínio estruturado.
   - _Implementação_: Seis passos: (1) Identificar usuário, (2) Sentir a frustração, (3) Transformar em desejo positivo, (4) Articular valor, (5) Extrair detalhes técnicos, (6) Definir verificação.
   - _Justificativa_: Garante análise completa do bug antes da escrita.

4. **Negative Examples (Contrastive Learning)**: Mostrar o que evitar.
   - _Implementação_: Exemplos de tom frio/técnico vs tom empático/orientado a valor.
   - _Justificativa_: Ajuda o modelo a entender a diferença entre uma User Story de baixa e alta qualidade.

5. **Rubric-based Prompting**: Incluir a rubrica de avaliação no prompt.
   - _Implementação_: Seção "COMO VOCÊ SERÁ AVALIADO" listando os 4 critérios e seus subcritérios (0-10).
   - _Justificativa_: O modelo sabe exatamente como será julgado e otimiza para esses critérios.

6. **Emotional Priming**: Frases que ativam empatia antes da escrita.
   - _Implementação_: "Imagine a frustração do usuário", "Coloque-se no lugar dele", "Você é a voz dele".
   - _Justificativa_: Melhora significativamente o Tone Score ao ativar linguagem mais empática.

### Melhorias Estruturais:

- **Formato de Saída Padronizado**: Template Markdown com 4 seções obrigatórias (User Story, Critérios de Aceitação, Contexto Técnico, Impacto e Prioridade)
- **Critérios de Aceitação**: Formato Dado-Quando-Então com 6-8 critérios testáveis
- **Preservação de Dados**: Instruções explícitas para manter IDs, valores, erros do bug original
- **Linguagem Positiva**: Foco no que o usuário QUER fazer, não no que está quebrado

## Resultados Finais

### Status: ✅ APROVADO

Todas as 4 métricas atingiram o critério de aprovação (>= 0.9).

### Link do LangSmith Dashboard

O prompt otimizado está disponível publicamente no LangSmith Hub:

- URL: `https://smith.langchain.com/prompts/bug_to_user_story_v2`
- Projeto: `https://smith.langchain.com/projects/desafio-prompt-engineer`

### Scores Finais (Iteração #13)

| Métrica                   | Score    | Status |
| ------------------------- | -------- | ------ |
| Tone Score                | **0.91** | ✅     |
| Acceptance Criteria Score | **0.91** | ✅     |
| User Story Format Score   | **0.95** | ✅     |
| Completeness Score        | **0.99** | ✅     |
| **Média**                 | **0.94** | ✅     |

### Evolução das Métricas

| Métrica             | Iteração 1 | Iteração 7 | Iteração 13 (Final) |
| ------------------- | ---------- | ---------- | ------------------- |
| Tone Score          | 0.835      | 0.848      | **0.91**            |
| Acceptance Criteria | 0.830      | 0.895      | **0.91**            |
| User Story Format   | 0.912      | 0.912      | **0.95**            |
| Completeness        | 0.868      | 0.903      | **0.99**            |
| **Média**           | 0.861      | 0.889      | **0.94**            |

### Configuração Final

- **Modelo de Geração**: gpt-5-mini
- **Modelo de Avaliação**: gpt-5
- **Total de Iterações**: 13

### Nota sobre Modelos e Custos

> **Importante**: O modelo `gpt-4o-mini` (sugerido inicialmente) **não conseguiu atingir os requisitos mínimos** de 0.9 em todas as métricas, especialmente no **Tone Score**, que ficou travado em ~0.85 mesmo após múltiplas iterações e técnicas avançadas de prompt engineering.
>
> Foi necessário migrar para modelos mais capazes:
>
> - **gpt-4o**: Tone Score chegou a 0.898 (ainda abaixo de 0.9)
> - **gpt-5-mini**: Tone Score atingiu 0.87, outras métricas passaram
> - **gpt-5**: Tone Score atingiu 0.90, todas as métricas passaram
>
> **Impacto no custo**: O budget máximo estimado de **$5** foi ultrapassado, resultando em um custo total de aproximadamente **$7** para completar este desafio com sucesso.

### Melhorias Alcançadas

1. **Tom Empático**: Uso de emotional priming e linguagem positiva aumentou Tone Score de 0.83 para 0.91
2. **Critérios Testáveis**: Formato Dado-Quando-Então com 6-8 critérios específicos
3. **Completude**: Preservação de todos os detalhes técnicos do bug original
4. **Estrutura**: Template Markdown com seções claramente separadas
5. **Valor de Negócio**: "Para que" sempre articula benefício real para o usuário

## Como Executar

Workflow completo para reproduzir os resultados:

```bash
# 1. Configurar ambiente
conda activate desafio_02  # ou seu ambiente Python
pip install -r requirements.txt

# 2. Configurar credenciais
cp .env.example .env
# Edite .env com suas API keys

# 3. Pull do prompt inicial
python src/pull_prompts.py

# 4. Validar estrutura do prompt otimizado
pytest tests/test_prompts.py

# 5. Push do prompt otimizado para LangSmith
python src/push_prompts.py

# 6. Avaliar qualidade do prompt
python src/evaluate.py

# 7. (Se necessário) Iterar até atingir >= 0.9 em todas as métricas
# - Edite prompts/bug_to_user_story_v2.yml
# - Repita passos 4-6
```

### Comandos Rápidos

| Ação                     | Comando                           |
| ------------------------ | --------------------------------- |
| Instalar dependências    | `pip install -r requirements.txt` |
| Pull prompt inicial      | `python src/pull_prompts.py`      |
| Validar estrutura        | `pytest tests/test_prompts.py`    |
| Push prompt otimizado    | `python src/push_prompts.py`      |
| Avaliar qualidade        | `python src/evaluate.py`          |
| Executar todos os testes | `pytest tests/`                   |

## Licença

MIT
