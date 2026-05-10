"""
Testes pytest para validação da estrutura do prompt.

Este módulo implementa 6 testes obrigatórios (FR-012) para garantir que o
prompt otimizado atende a todos os requisitos estruturais antes da avaliação.

Testes:
1. test_system_prompt_existe_e_nao_e_vazio     - Verifica se system_prompt existe e tem conteúdo
2. test_system_prompt_define_papel_do_agente   - Verifica padrão "You are" ou "Você é"
3. test_system_prompt_menciona_formato         - Verifica menção a "User Story" ou "Markdown"
4. test_exemplos_few_shot_presentes            - Verifica se a lista de exemplos tem >= 1 item
5. test_sem_marcadores_todo                    - Verifica ausência de marcadores "[TODO]"
6. test_minimo_de_tecnicas_documentadas        - Verifica se metadata.techniques tem >= 2 itens

Uso:
    pytest tests/test_prompts.py
"""

import sys
import pytest
from pathlib import Path

# Adiciona src ao path para imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils import load_yaml

# Caminho para o arquivo de prompt otimizado
CAMINHO_PROMPT = Path(__file__).parent.parent / "prompts" / "bug_to_user_story_v2.yml"


@pytest.fixture
def dados_do_prompt():
    """
    Carrega os dados do prompt otimizado a partir do arquivo YAML.

    Retorna:
        Dicionário com os dados do prompt.

    Lança:
        FileNotFoundError: Se o arquivo de prompt não existir.
    """
    if not CAMINHO_PROMPT.exists():
        pytest.fail(
            f"Arquivo de prompt não encontrado em {CAMINHO_PROMPT}. "
            "Crie o prompt otimizado antes de rodar os testes."
        )
    return load_yaml(str(CAMINHO_PROMPT))


def test_system_prompt_existe_e_nao_e_vazio(dados_do_prompt):
    """
    Verifica se o prompt contém um system_prompt não vazio.

    O system_prompt é o conjunto central de instruções que guia o comportamento
    do LLM. Deve existir e conter conteúdo real.

    Dica: Adicione um campo 'system_prompt' com definição de papel e instruções.
    """
    assert "system_prompt" in dados_do_prompt, (
        "Campo obrigatório ausente: system_prompt. "
        "Dica: Adicione um campo 'system_prompt' com definição de papel e instruções."
    )

    system_prompt = dados_do_prompt["system_prompt"]

    assert system_prompt is not None, (
        "system_prompt não pode ser None. "
        "Dica: Forneça conteúdo real para o system prompt."
    )

    assert isinstance(system_prompt, str), (
        f"system_prompt deve ser uma string, mas é {type(system_prompt).__name__}. "
        "Dica: Garanta que system_prompt seja um valor de texto, não uma lista ou dicionário."
    )

    assert len(system_prompt.strip()) > 0, (
        "system_prompt não pode estar vazio ou conter apenas espaços. "
        "Dica: Adicione instruções significativas que definam o papel e o comportamento do LLM."
    )


def test_system_prompt_define_papel_do_agente(dados_do_prompt):
    """
    Verifica se o system_prompt contém um padrão de definição de papel.

    Role Prompting é uma técnica de engenharia de prompt onde atribuímos
    uma persona específica ao LLM (ex: "You are an experienced Product Manager").

    Padrões aceitos: "You are" ou "Você é" (português)

    Dica: Adicione uma definição de papel clara como
    "You are an experienced Product Manager".
    """
    assert (
        "system_prompt" in dados_do_prompt
    ), "Não é possível verificar a definição de papel: system_prompt está ausente."

    conteudo = dados_do_prompt["system_prompt"].lower()

    tem_papel_ingles = "you are" in conteudo
    tem_papel_portugues = "você é" in conteudo

    assert tem_papel_ingles or tem_papel_portugues, (
        "O system prompt deve conter uma definição de papel. "
        "Padrão esperado: 'You are' ou 'Você é'. "
        "Dica: Adicione algo como 'You are an experienced Product Manager "
        "with expertise in Agile methodologies.'"
    )


def test_system_prompt_menciona_formato(dados_do_prompt):
    """
    Verifica se o system_prompt menciona o formato esperado de saída.

    O prompt deve especificar claramente que a saída deve estar no formato
    de User Story e/ou estrutura Markdown.

    Menções aceitas: "User Story" ou "Markdown"

    Dica: Adicione instruções sobre o formato de saída, ex:
    "Format your response as a User Story in Markdown".
    """
    assert (
        "system_prompt" in dados_do_prompt
    ), "Não é possível verificar a menção ao formato: system_prompt está ausente."

    conteudo = dados_do_prompt["system_prompt"].lower()

    tem_user_story = "user story" in conteudo
    tem_markdown = "markdown" in conteudo

    assert tem_user_story or tem_markdown, (
        "O system prompt deve mencionar o formato de saída. "
        "Menção esperada: 'User Story' ou 'Markdown'. "
        "Dica: Adicione instruções como 'Structure your response in Markdown with "
        "the following sections: User Story title, As a/I want/So that format, "
        "and Acceptance Criteria.'"
    )


def test_exemplos_few_shot_presentes(dados_do_prompt):
    """
    Verifica se o prompt contém ao menos um exemplo de few-shot.

    Few-shot learning é uma técnica de engenharia de prompt onde fornecemos
    exemplos de entrada/saída para guiar o comportamento do LLM.

    Cada exemplo deve ter 'input' (bug report) e 'output' (User Story).

    Dica: Adicione uma lista 'examples' com pelo menos um item
    no formato {input: ..., output: ...}.
    """
    assert "examples" in dados_do_prompt, (
        "Campo obrigatório ausente: examples. "
        "Dica: Adicione uma lista 'examples' com pares de entrada/saída para few-shot."
    )

    exemplos = dados_do_prompt["examples"]

    assert isinstance(exemplos, list), (
        f"'examples' deve ser uma lista, mas é {type(exemplos).__name__}. "
        "Dica: Formate os exemplos como lista YAML com itens '- input:' e 'output:'."
    )

    assert len(exemplos) >= 1, (
        f"'examples' deve ter ao menos 1 item, mas encontrou {len(exemplos)}. "
        "Dica: Adicione pelo menos um exemplo mostrando como converter um bug report em User Story."
    )

    for indice, exemplo in enumerate(exemplos):
        assert "input" in exemplo, (
            f"Exemplo {indice} está sem o campo obrigatório: input. "
            "Dica: Cada exemplo precisa de um campo 'input' com um bug report de amostra."
        )
        assert "output" in exemplo, (
            f"Exemplo {indice} está sem o campo obrigatório: output. "
            "Dica: Cada exemplo precisa de um campo 'output' com a User Story esperada."
        )
        assert exemplo["input"], (
            f"Exemplo {indice} tem 'input' vazio. "
            "Dica: Forneça um bug report realista como entrada."
        )
        assert exemplo["output"], (
            f"Exemplo {indice} tem 'output' vazio. "
            "Dica: Forneça a transformação esperada em User Story como saída."
        )


def test_sem_marcadores_todo(dados_do_prompt):
    """
    Verifica se o prompt não contém marcadores [TODO].

    Marcadores [TODO] indicam seções incompletas que precisam ser preenchidas.
    Todos os placeholders devem ser resolvidos antes de o prompt estar pronto para uso.

    Dica: Busque por '[TODO]' no seu prompt e substitua pelo conteúdo real.
    """
    conteudo_completo = str(dados_do_prompt)

    assert "[TODO]" not in conteudo_completo, (
        "O prompt contém marcadores [TODO] que precisam ser resolvidos. "
        "Dica: Busque por '[TODO]' em prompts/bug_to_user_story_v2.yml e "
        "substitua todos os placeholders pelo conteúdo real."
    )

    conteudo_lower = conteudo_completo.lower()
    padroes_todo = ["[todo]", "# todo", "// todo", "/* todo"]

    for padrao in padroes_todo:
        assert padrao not in conteudo_lower, (
            f"O prompt contém marcador TODO: '{padrao}'. "
            "Dica: Todos os placeholders TODO devem ser substituídos por conteúdo real."
        )


def test_minimo_de_tecnicas_documentadas(dados_do_prompt):
    """
    Verifica se os metadados do prompt listam ao menos 2 técnicas de engenharia de prompt.

    O campo 'techniques' nos metadados documenta quais métodos de engenharia de prompt
    foram aplicados (ex: role_prompting, few_shot_learning, chain_of_thought).

    Esperado: metadata.techniques com >= 2 itens.

    Dica: Adicione técnicas como 'role_prompting', 'few_shot_learning', 'chain_of_thought'.
    """
    assert "metadata" in dados_do_prompt, (
        "Campo obrigatório ausente: metadata. "
        "Dica: Adicione uma seção 'metadata' com name, description e techniques."
    )

    metadata = dados_do_prompt["metadata"]

    assert "techniques" in metadata, (
        "metadata está sem o campo obrigatório: techniques. "
        "Dica: Adicione uma lista 'techniques' documentando os métodos de engenharia de prompt usados."
    )

    tecnicas = metadata["techniques"]

    assert isinstance(tecnicas, list), (
        f"metadata.techniques deve ser uma lista, mas é {type(tecnicas).__name__}. "
        "Dica: Formate as técnicas como lista YAML, ex: '- role_prompting'."
    )

    assert len(tecnicas) >= 2, (
        f"metadata.techniques deve ter ao menos 2 itens, mas encontrou {len(tecnicas)}. "
        "Dica: Aplique e documente ao menos 2 técnicas. Sugestões: "
        "role_prompting, few_shot_learning, chain_of_thought."
    )

    for indice, tecnica in enumerate(tecnicas):
        assert isinstance(tecnica, str) and tecnica.strip(), (
            f"metadata.techniques[{indice}] deve ser uma string não vazia. "
            "Dica: Use nomes descritivos como 'role_prompting'."
        )


# Teste bônus para validação completa
def test_estrutura_completa_do_prompt(dados_do_prompt):
    """
    Bônus: Verifica se a estrutura geral do prompt está completa e válida.

    Realiza uma checagem abrangente de todos os campos obrigatórios
    para garantir que o prompt está pronto para avaliação.
    """
    assert "metadata" in dados_do_prompt, "Seção 'metadata' ausente"

    metadata = dados_do_prompt["metadata"]
    assert "name" in metadata and metadata["name"], "metadata.name ausente ou vazio"
    assert (
        "description" in metadata and metadata["description"]
    ), "metadata.description ausente ou vazio"

    assert "system_prompt" in dados_do_prompt, "Campo 'system_prompt' ausente"
    assert (
        len(dados_do_prompt["system_prompt"].strip()) > 100
    ), "system_prompt parece curto demais. Um bom prompt deve ter instruções detalhadas."

    assert (
        "user_prompt_template" in dados_do_prompt
    ), "Campo 'user_prompt_template' ausente"
    assert (
        "{bug_report}" in dados_do_prompt["user_prompt_template"]
    ), "user_prompt_template deve conter a variável {bug_report}"

    assert "examples" in dados_do_prompt, "Campo 'examples' ausente"
    assert len(dados_do_prompt["examples"]) >= 1, "É necessário ao menos 1 exemplo"
