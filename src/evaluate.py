"""
Script COMPLETO para avaliar prompts otimizados.

Este script:
1. Carrega dataset de avaliação de arquivo .jsonl (datasets/bug_to_user_story.jsonl)
2. Cria/atualiza dataset no LangSmith
3. Puxa prompts otimizados do LangSmith Hub (fonte única de verdade)
4. Executa prompts contra o dataset
5. Calcula 4 métricas específicas para Bug to User Story:
   - Tone Score: Tom profissional e empático
   - Acceptance Criteria Score: Qualidade dos critérios de aceitação
   - User Story Format Score: Formato correto (Como... Eu quero... Para que...)
   - Completeness Score: Completude e contexto técnico
6. Publica resultados no dashboard do LangSmith
7. Exibe resumo no terminal

Critério de Aprovação: TODAS as 4 métricas devem ser >= 0.9

Suporta providers de LLM:
- Google Gemini (gemini-1.5-flash, gemini-1.5-pro)

Configure o provider no arquivo .env através da variável LLM_PROVIDER.
"""

import os
import sys
import json
from typing import List, Dict, Any
from pathlib import Path
from dotenv import load_dotenv
from langsmith import Client
from langchain_core.prompts import ChatPromptTemplate
from openai import api_key
from openai import api_key
from utils import (
    check_env_vars,
    format_score,
    print_section_header,
    get_llm as get_configured_llm,
    get_last_evaluation,
    save_evaluation_result,
    print_evaluation_comparison,
    load_evaluation_history,
)
from metrics import (
    evaluate_tone_score,
    evaluate_acceptance_criteria_score,
    evaluate_user_story_format_score,
    evaluate_completeness_score,
)
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()


def get_llm():

    api_key = os.getenv("GOOGLE_API_KEY")
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", temperature=0, google_api_key=api_key
    )


def load_dataset_from_jsonl(jsonl_path: str) -> List[Dict[str, Any]]:
    examples = []

    try:
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:  # Ignorar linhas vazias
                    example = json.loads(line)
                    examples.append(example)

        return examples

    except FileNotFoundError:
        print(f"❌ Arquivo não encontrado: {jsonl_path}")
        print(
            "\nCertifique-se de que o arquivo datasets/bug_to_user_story.jsonl existe."
        )
        return []
    except json.JSONDecodeError as e:
        print(f"❌ Erro ao parsear JSONL: {e}")
        return []
    except Exception as e:
        print(f"❌ Erro ao carregar dataset: {e}")
        return []


def create_evaluation_dataset(
    client: Client, dataset_name: str, jsonl_path: str
) -> str:
    print(f"Criando dataset de avaliação: {dataset_name}...")

    examples = load_dataset_from_jsonl(jsonl_path)

    if not examples:
        print("❌ Nenhum exemplo carregado do arquivo .jsonl")
        return dataset_name

    print(f"   ✓ Carregados {len(examples)} exemplos do arquivo {jsonl_path}")

    try:
        datasets = client.list_datasets(dataset_name=dataset_name)
        existing_dataset = None

        for ds in datasets:
            if ds.name == dataset_name:
                existing_dataset = ds
                break

        if existing_dataset:
            print(f"   ✓ Dataset '{dataset_name}' já existe, usando existente")
            return dataset_name
        else:
            dataset = client.create_dataset(dataset_name=dataset_name)

            for example in examples:
                client.create_example(
                    dataset_id=dataset.id,
                    inputs=example["inputs"],
                    outputs=example["outputs"],
                )

            print(f"   ✓ Dataset criado com {len(examples)} exemplos")
            return dataset_name

    except Exception as e:
        print(f"   ⚠️  Erro ao criar dataset: {e}")
        return dataset_name


def pull_prompt_from_langsmith(prompt_name: str) -> ChatPromptTemplate:
    try:
        print(f"   Puxando prompt do LangSmith Hub: {prompt_name}")
        client = Client()
        prompt = client.pull_prompt(prompt_name)
        print(f"   ✓ Prompt carregado com sucesso")
        return prompt

    except Exception as e:
        error_msg = str(e).lower()

        print(f"\n{'=' * 70}")
        print(f"❌ ERRO: Não foi possível carregar o prompt '{prompt_name}'")
        print(f"{'=' * 70}\n")

        if "not found" in error_msg or "404" in error_msg:
            print("⚠️  O prompt não foi encontrado no LangSmith Hub.\n")
            print("AÇÕES NECESSÁRIAS:")
            print("1. Verifique se você já fez push do prompt otimizado:")
            print(f"   python src/push_prompts.py")
            print()
            print("2. Confirme se o prompt foi publicado com sucesso em:")
            print(f"   https://smith.langchain.com/prompts")
            print()
            print(
                f"3. Certifique-se de que o nome do prompt está correto: '{prompt_name}'"
            )
            print()
            print("4. Se você alterou o prompt no YAML, refaça o push:")
            print(f"   python src/push_prompts.py")
        else:
            print(f"Erro técnico: {e}\n")
            print("Verifique:")
            print("- LANGCHAIN_API_KEY está configurada corretamente no .env")
            print("- Você tem acesso ao workspace do LangSmith")
            print("- Sua conexão com a internet está funcionando")

        print(f"\n{'=' * 70}\n")
        raise


def evaluate_prompt_on_example(
    prompt_template: ChatPromptTemplate, example: Any, llm: Any
) -> Dict[str, Any]:
    try:
        inputs = example.inputs if hasattr(example, "inputs") else {}
        outputs = example.outputs if hasattr(example, "outputs") else {}

        chain = prompt_template | llm

        response = chain.invoke(inputs)
        answer = response.content

        reference = outputs.get("reference", "") if isinstance(outputs, dict) else ""

        if isinstance(inputs, dict):
            question = inputs.get(
                "question", inputs.get("bug_report", inputs.get("pr_title", "N/A"))
            )
        else:
            question = "N/A"

        return {"answer": answer, "reference": reference, "question": question}

    except Exception as e:
        print(f"      ⚠️  Erro ao avaliar exemplo: {e}")
        import traceback

        print(f"      Traceback: {traceback.format_exc()}")
        return {"answer": "", "reference": "", "question": ""}


def evaluate_prompt(
    prompt_name: str, dataset_name: str, client: Client, sample_size: int = None
) -> Dict[str, float]:
    print(f"\n🔍 Avaliando: {prompt_name}")

    try:
        prompt_template = pull_prompt_from_langsmith(prompt_name)

        all_examples = list(client.list_examples(dataset_name=dataset_name))
        total_in_dataset = len(all_examples)

        # Usar amostra se especificado, senão usar todos
        if sample_size and sample_size < total_in_dataset:
            import random

            # Sem seed fixo = amostragem verdadeiramente aleatória a cada execução
            examples = random.sample(all_examples, sample_size)
            print(
                f"   Dataset: {total_in_dataset} exemplos (amostra aleatória de {sample_size})"
            )
        else:
            examples = all_examples
            print(f"   Dataset: {total_in_dataset} exemplos (avaliando todos)")

        llm = get_llm()

        # 4 métricas específicas para Bug to User Story (critério de aprovação)
        tone_scores = []
        acceptance_criteria_scores = []
        user_story_format_scores = []
        completeness_scores = []

        print("   Avaliando exemplos...")

        total_examples = len(examples)
        for i, example in enumerate(examples, 1):
            result = evaluate_prompt_on_example(prompt_template, example, llm)

            if result["answer"]:
                # Usar as 4 métricas específicas conforme Intro.md
                tone = evaluate_tone_score(
                    result["question"], result["answer"], result["reference"]
                )
                acceptance = evaluate_acceptance_criteria_score(
                    result["question"], result["answer"], result["reference"]
                )
                format_score = evaluate_user_story_format_score(
                    result["question"], result["answer"], result["reference"]
                )
                completeness = evaluate_completeness_score(
                    result["question"], result["answer"], result["reference"]
                )

                tone_scores.append(tone["score"])
                acceptance_criteria_scores.append(acceptance["score"])
                user_story_format_scores.append(format_score["score"])
                completeness_scores.append(completeness["score"])

                print(
                    f"      [{i}/{total_examples}] Tone:{tone['score']:.2f} AC:{acceptance['score']:.2f} Format:{format_score['score']:.2f} Complete:{completeness['score']:.2f}"
                )

        # Calcular médias das 4 métricas específicas
        avg_tone = sum(tone_scores) / len(tone_scores) if tone_scores else 0.0
        avg_acceptance = (
            sum(acceptance_criteria_scores) / len(acceptance_criteria_scores)
            if acceptance_criteria_scores
            else 0.0
        )
        avg_format = (
            sum(user_story_format_scores) / len(user_story_format_scores)
            if user_story_format_scores
            else 0.0
        )
        avg_completeness = (
            sum(completeness_scores) / len(completeness_scores)
            if completeness_scores
            else 0.0
        )

        return {
            "tone_score": round(avg_tone, 4),
            "acceptance_criteria_score": round(avg_acceptance, 4),
            "user_story_format_score": round(avg_format, 4),
            "completeness_score": round(avg_completeness, 4),
        }

    except Exception as e:
        print(f"   ❌ Erro na avaliação: {e}")
        return {
            "tone_score": 0.0,
            "acceptance_criteria_score": 0.0,
            "user_story_format_score": 0.0,
            "completeness_score": 0.0,
        }


def display_results(
    prompt_name: str,
    scores: Dict[str, float],
    previous_evaluation: Dict[str, Any] = None,
) -> bool:
    print("\n" + "=" * 50)
    print(f"Prompt: {prompt_name}")
    print("=" * 50)

    # Mostrar número da iteração
    history = load_evaluation_history()
    current_iteration = len(history) + 1
    print(f"\n🔄 Iteração: #{current_iteration}")

    print("\n📊 Métricas Bug to User Story (Critério de Aprovação):")
    print(f"  - Tone Score: {format_score(scores['tone_score'], threshold=0.9)}")
    print(
        f"  - Acceptance Criteria Score: {format_score(scores['acceptance_criteria_score'], threshold=0.9)}"
    )
    print(
        f"  - User Story Format Score: {format_score(scores['user_story_format_score'], threshold=0.9)}"
    )
    print(
        f"  - Completeness Score: {format_score(scores['completeness_score'], threshold=0.9)}"
    )

    average_score = sum(scores.values()) / len(scores)

    # Verificar se TODAS as métricas estão >= 0.9 (não apenas a média)
    all_metrics_pass = all(score >= 0.9 for score in scores.values())

    print("\n" + "-" * 50)
    print(f"📊 MÉDIA DAS 4 MÉTRICAS: {average_score:.4f}")
    print("-" * 50)

    # IMPORTANTE: Todas as 4 métricas devem estar >= 0.9
    if all_metrics_pass and average_score >= 0.9:
        print(f"\n✅ STATUS: APROVADO")
        print(f"   ✓ Todas as 4 métricas >= 0.9")
        print(f"   ✓ Média >= 0.9")
    else:
        print(f"\n❌ STATUS: REPROVADO")
        if not all_metrics_pass:
            print(f"   ⚠️  Nem todas as métricas estão >= 0.9:")
            for name, score in scores.items():
                if score < 0.9:
                    print(f"      - {name}: {score:.4f} (precisa >= 0.9)")
        if average_score < 0.9:
            print(f"   ⚠️  Média atual: {average_score:.4f} | Necessário: >= 0.9")

    # Mostrar comparação com avaliação anterior
    print_evaluation_comparison(scores, previous_evaluation)

    return all_metrics_pass and average_score >= 0.9


def main():
    print_section_header("AVALIAÇÃO DE PROMPTS OTIMIZADOS")

    # Carregar avaliação anterior para comparação
    previous_evaluation = get_last_evaluation()
    history = load_evaluation_history()

    if previous_evaluation:
        print(f"📜 Histórico: {len(history)} avaliações anteriores")
        print(f"   Última iteração: #{previous_evaluation.get('iteration', '?')}")
    else:
        print("📜 Histórico: Nenhuma avaliação anterior encontrada")

    print()

    # Sample size para avaliação (None = usar todos, número = usar amostra)
    sample_size_env = os.getenv("EVAL_SAMPLE_SIZE", "5")
    sample_size = int(sample_size_env) if sample_size_env.lower() != "all" else None

    print(f"Sample Size: {sample_size if sample_size else 'ALL (todos os exemplos)'}\n")

    client = Client()
    project_name = os.getenv(
        "LANGCHAIN_PROJECT", "prompt-optimization-challenge-resolved"
    )

    jsonl_path = "datasets/bug_to_user_story.jsonl"

    if not Path(jsonl_path).exists():
        print(f"❌ Arquivo de dataset não encontrado: {jsonl_path}")
        print("\nCertifique-se de que o arquivo existe antes de continuar.")
        return 1

    dataset_name = f"{project_name}-eval"
    create_evaluation_dataset(client, dataset_name, jsonl_path)

    print("\n" + "=" * 70)
    print("PROMPTS PARA AVALIAR")
    print("=" * 70)
    print("\nEste script irá puxar prompts do LangSmith Hub.")
    print("Certifique-se de ter feito push dos prompts antes de avaliar:")
    print("  python src/push_prompts.py\n")

    prompts_to_evaluate = [
        "bug_to_user_story_v2",
    ]

    all_passed = True
    evaluated_count = 0
    results_summary = []

    for prompt_name in prompts_to_evaluate:
        evaluated_count += 1

        try:
            scores = evaluate_prompt(
                prompt_name, dataset_name, client, sample_size=sample_size
            )

            # Passar avaliação anterior para comparação
            passed = display_results(prompt_name, scores, previous_evaluation)
            all_passed = all_passed and passed

            result_data = {"prompt": prompt_name, "scores": scores, "passed": passed}

            results_summary.append(result_data)

            # Salvar resultado no histórico
            save_evaluation_result(result_data)
            print(f"\n💾 Resultado salvo no histórico (evaluations/history.json)")

        except Exception as e:
            print(f"\n❌ Falha ao avaliar '{prompt_name}': {e}")
            all_passed = False

            error_result = {
                "prompt": prompt_name,
                "scores": {
                    "tone_score": 0.0,
                    "acceptance_criteria_score": 0.0,
                    "user_story_format_score": 0.0,
                    "completeness_score": 0.0,
                },
                "passed": False,
                "error": str(e),
            }

            results_summary.append(error_result)

            # Salvar mesmo resultados com erro
            save_evaluation_result(error_result)
            print(f"\n💾 Resultado (com erro) salvo no histórico")

    print("\n" + "=" * 50)
    print("RESUMO FINAL")
    print("=" * 50 + "\n")

    if evaluated_count == 0:
        print("⚠️  Nenhum prompt foi avaliado")
        return 1

    # Mostrar total de iterações
    total_iterations = len(load_evaluation_history())
    print(f"📊 Total de iterações realizadas: {total_iterations}")
    print(f"   (Esperado: 3-5 iterações para atingir >= 0.9)\n")

    print(f"Prompts avaliados: {evaluated_count}")
    print(f"Aprovados: {sum(1 for r in results_summary if r['passed'])}")
    print(f"Reprovados: {sum(1 for r in results_summary if not r['passed'])}\n")

    if all_passed:
        print("✅ Todos os prompts atingiram as 4 métricas >= 0.9!")
        print("\n   Critérios de Aprovação Atingidos:")
        print("   ✓ Tone Score >= 0.9")
        print("   ✓ Acceptance Criteria Score >= 0.9")
        print("   ✓ User Story Format Score >= 0.9")
        print("   ✓ Completeness Score >= 0.9")
        print(f"\n✓ Confira os resultados em:")
        print(f"  https://smith.langchain.com/projects/{project_name}")
        print("\nPróximos passos:")
        print("1. Documente o processo no README.md")
        print("2. Capture screenshots das avaliações")
        print("3. Faça commit e push para o GitHub")
        return 0
    else:
        print("⚠️  Alguns prompts não atingiram todas as 4 métricas >= 0.9")
        print("\n   Critérios de Aprovação (TODAS devem ser >= 0.9):")
        print("   - Tone Score")
        print("   - Acceptance Criteria Score")
        print("   - User Story Format Score")
        print("   - Completeness Score")
        print("\nPróximos passos:")
        print(
            "1. Refatore os prompts com score baixo em prompts/bug_to_user_story_v2.yml"
        )
        print(
            "2. Faça commit das alterações: git add prompts/ && git commit -m 'Iteração N: melhorias no prompt'"
        )
        print("3. Faça push para o LangSmith: python src/push_prompts.py")
        print("4. Execute novamente: python src/evaluate.py")
        print("5. Repita até TODAS as métricas >= 0.9 (esperado: 3-5 iterações)")
        return 1


if __name__ == "__main__":
    sys.exit(main())
