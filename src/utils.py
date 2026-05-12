"""
Funções auxiliares para o projeto de otimização de prompts.
"""

import os
import yaml
import json
from typing import Dict, Any, Optional
from pathlib import Path
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()


def load_env() -> bool:
    """
    Carrega variáveis de ambiente do arquivo .env.

    Returns:
        True se arquivo .env foi carregado, False caso contrário
    """
    env_path = Path(".env")
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
        return True
    return False


def validate_env_vars(required_vars: list) -> bool:
    """
    Valida se variáveis de ambiente obrigatórias estão configuradas.

    Args:
        required_vars: Lista de variáveis obrigatórias

    Returns:
        True se todas configuradas, False caso contrário
    """
    missing_vars = []

    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)

    if missing_vars:
        print_error("Variáveis de ambiente faltando:")
        for var in missing_vars:
            print(f"   - {var}")
        print("\nConfigure-as no arquivo .env antes de continuar.")
        return False

    return True


def print_error(message: str) -> None:
    """
    Imprime mensagem de erro formatada.

    Args:
        message: Mensagem de erro
    """
    print(f"❌ {message}")


def print_success(message: str) -> None:
    """
    Imprime mensagem de sucesso formatada.

    Args:
        message: Mensagem de sucesso
    """
    print(f"✓ {message}")


def load_yaml(file_path: str) -> Optional[Dict[str, Any]]:
    """
    Carrega arquivo YAML.

    Args:
        file_path: Caminho do arquivo YAML

    Returns:
        Dicionário com conteúdo do YAML ou None se erro
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return data
    except FileNotFoundError:
        print(f"❌ Arquivo não encontrado: {file_path}")
        return None
    except yaml.YAMLError as e:
        print(f"❌ Erro ao parsear YAML: {e}")
        return None
    except Exception as e:
        print(f"❌ Erro ao carregar arquivo: {e}")
        return None


def save_yaml(data: Dict[str, Any], file_path: str) -> bool:
    """
    Salva dados em arquivo YAML.

    Args:
        data: Dados para salvar
        file_path: Caminho do arquivo de saída

    Returns:
        True se sucesso, False caso contrário
    """
    try:
        output_file = Path(file_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w", encoding="utf-8") as f:
            yaml.dump(data, f, allow_unicode=True, sort_keys=False, indent=2)

        return True
    except Exception as e:
        print(f"❌ Erro ao salvar arquivo: {e}")
        return False


def check_env_vars(required_vars: list) -> bool:
    """
    Verifica se variáveis de ambiente obrigatórias estão configuradas.

    Args:
        required_vars: Lista de variáveis obrigatórias

    Returns:
        True se todas configuradas, False caso contrário
    """
    missing_vars = []

    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)

    if missing_vars:
        print("❌ Variáveis de ambiente faltando:")
        for var in missing_vars:
            print(f"   - {var}")
        print("\nConfigure-as no arquivo .env antes de continuar.")
        return False

    return True


def format_score(score: float, threshold: float = 0.9) -> str:
    """
    Formata score com indicador visual de aprovação.

    Args:
        score: Score entre 0.0 e 1.0
        threshold: Limite mínimo para aprovação

    Returns:
        String formatada com score e símbolo
    """
    symbol = "✓" if score >= threshold else "✗"
    return f"{score:.2f} {symbol}"


def print_section_header(title: str, char: str = "=", width: int = 50):
    """
    Imprime cabeçalho de seção formatado.

    Args:
        title: Título da seção
        char: Caractere para a linha
        width: Largura da linha
    """
    print("\n" + char * width)
    print(title)
    print(char * width + "\n")


def validate_prompt_structure(prompt_data: Dict[str, Any]) -> tuple[bool, list]:
    """
    Valida estrutura básica de um prompt.

    Args:
        prompt_data: Dados do prompt

    Returns:
        (is_valid, errors) - Tupla com status e lista de erros
    """
    errors = []

    required_fields = ["description", "system_prompt", "version"]
    for field in required_fields:
        if field not in prompt_data:
            errors.append(f"Campo obrigatório faltando: {field}")

    system_prompt = prompt_data.get("system_prompt", "").strip()
    if not system_prompt:
        errors.append("system_prompt está vazio")

    if "TODO" in system_prompt:
        errors.append("system_prompt ainda contém TODOs")

    techniques = prompt_data.get("techniques_applied", [])
    if len(techniques) < 2:
        errors.append(
            f"Mínimo de 2 técnicas requeridas, encontradas: {len(techniques)}"
        )

    return (len(errors) == 0, errors)


def extract_json_from_response(response_text: str) -> Optional[Dict[str, Any]]:
    """
    Extrai JSON de uma resposta de LLM que pode conter texto adicional.

    Args:
        response_text: Texto da resposta do LLM

    Returns:
        Dicionário extraído ou None se não encontrar JSON válido
    """
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        start = response_text.find("{")
        end = response_text.rfind("}") + 1

        if start != -1 and end > start:
            try:
                json_str = response_text[start:end]
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass

    return None


def get_llm(model: Optional[str] = None, temperature: float = 0.0):
    """
    Retorna uma instância de LLM configurada baseada no provider.

    Args:
        model: Nome do modelo (opcional, usa LLM_MODEL do .env por padrão)
        temperature: Temperatura para geração (padrão: 0.0 para determinístico)

    Returns:
        Instância de ChatOpenAI ou ChatGoogleGenerativeAI

    Raises:
        ValueError: Se provider não for suportado ou API key não configurada
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite", temperature=temperature, google_api_key=api_key
    )


def get_eval_llm(temperature: float = 0.0):
    """
    Retorna LLM configurado especificamente para avaliação (usa EVAL_MODEL).

    Args:
        temperature: Temperatura para geração

    Returns:
        Instância de LLM configurada para avaliação
    """
    eval_model = os.getenv("EVAL_MODEL", "gpt-4o")
    return get_llm(model=eval_model, temperature=temperature)


# ============================================================
# Funções de Histórico de Avaliações
# ============================================================

EVALUATION_HISTORY_PATH = Path(__file__).parent.parent / "evaluations" / "history.json"


def load_evaluation_history() -> list:
    """
    Carrega histórico de avaliações do arquivo JSON.

    Returns:
        Lista de avaliações anteriores (mais recente por último)
    """
    if not EVALUATION_HISTORY_PATH.exists():
        return []

    try:
        with open(EVALUATION_HISTORY_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, Exception) as e:
        print(f"⚠️  Aviso: Erro ao carregar histórico: {e}")
        return []


def save_evaluation_result(result: Dict[str, Any]) -> bool:
    """
    Salva resultado de avaliação no histórico.

    Args:
        result: Dicionário com resultado da avaliação contendo:
            - timestamp: Data/hora da avaliação
            - prompt_name: Nome do prompt avaliado
            - scores: Dicionário com as 4 métricas
            - passed: Se passou em todas as métricas
            - iteration: Número da iteração

    Returns:
        True se salvou com sucesso, False caso contrário
    """
    from datetime import datetime

    history = load_evaluation_history()

    # Adicionar timestamp se não existir
    if "timestamp" not in result:
        result["timestamp"] = datetime.now().isoformat()

    # Calcular número da iteração
    result["iteration"] = len(history) + 1

    history.append(result)

    try:
        EVALUATION_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)

        with open(EVALUATION_HISTORY_PATH, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)

        return True
    except Exception as e:
        print(f"❌ Erro ao salvar histórico: {e}")
        return False


def get_last_evaluation() -> Optional[Dict[str, Any]]:
    """
    Retorna a última avaliação do histórico.

    Returns:
        Dicionário com última avaliação ou None se não houver histórico
    """
    history = load_evaluation_history()
    return history[-1] if history else None


def format_comparison(current: float, previous: float) -> str:
    """
    Formata comparação entre score atual e anterior.

    Args:
        current: Score atual
        previous: Score anterior

    Returns:
        String formatada com indicador de melhoria/piora
    """
    diff = current - previous

    if diff > 0.001:
        return f"↑ +{diff:.4f}"
    elif diff < -0.001:
        return f"↓ {diff:.4f}"
    else:
        return "= 0.0000"


def print_evaluation_comparison(
    current_scores: Dict[str, float], previous: Optional[Dict[str, Any]]
) -> None:
    """
    Imprime comparação entre avaliação atual e anterior.

    Args:
        current_scores: Scores da avaliação atual
        previous: Resultado da avaliação anterior (ou None)
    """
    if not previous:
        print("\n📊 Primeira avaliação - sem histórico para comparar")
        return

    print("\n" + "=" * 60)
    print("📊 COMPARAÇÃO COM AVALIAÇÃO ANTERIOR")
    print("=" * 60)

    prev_scores = previous.get("scores", {})
    prev_iteration = previous.get("iteration", "?")
    prev_timestamp = previous.get("timestamp", "N/A")

    print(f"\nIteração anterior: #{prev_iteration} ({prev_timestamp[:19]})")
    print(f"Iteração atual:    #{len(load_evaluation_history()) + 1}")
    print()

    print(f"{'Métrica':<30} {'Anterior':>10} {'Atual':>10} {'Variação':>15}")
    print("-" * 65)

    metrics = [
        ("tone_score", "Tone Score"),
        ("acceptance_criteria_score", "Acceptance Criteria"),
        ("user_story_format_score", "User Story Format"),
        ("completeness_score", "Completeness"),
    ]

    improved_count = 0
    worsened_count = 0

    for key, label in metrics:
        prev_val = prev_scores.get(key, 0.0)
        curr_val = current_scores.get(key, 0.0)
        comparison = format_comparison(curr_val, prev_val)

        if curr_val > prev_val + 0.001:
            improved_count += 1
        elif curr_val < prev_val - 0.001:
            worsened_count += 1

        print(f"{label:<30} {prev_val:>10.4f} {curr_val:>10.4f} {comparison:>15}")

    print("-" * 65)

    # Média
    prev_avg = sum(prev_scores.values()) / len(prev_scores) if prev_scores else 0
    curr_avg = (
        sum(current_scores.values()) / len(current_scores) if current_scores else 0
    )
    avg_comparison = format_comparison(curr_avg, prev_avg)

    print(f"{'MÉDIA':<30} {prev_avg:>10.4f} {curr_avg:>10.4f} {avg_comparison:>15}")
    print()

    # Resumo
    if improved_count > worsened_count:
        print(
            f"✅ Progresso: {improved_count} métricas melhoraram, {worsened_count} pioraram"
        )
    elif worsened_count > improved_count:
        print(
            f"⚠️  Regressão: {worsened_count} métricas pioraram, {improved_count} melhoraram"
        )
    else:
        print(f"➡️  Estável: {improved_count} melhoraram, {worsened_count} pioraram")
