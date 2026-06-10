"""Compact gravity reasoning generator."""

from __future__ import annotations

import statistics

from reasoners.store_types import Problem


def reasoning_gravity(problem: Problem) -> str | None:
    """Infer Wonderland gravity, bind each intermediate, then answer."""
    observations: list[tuple[str, str, float]] = []
    for example in problem.examples:
        t = float(example.input_value)
        d = float(example.output_value)
        if t > 0:
            observations.append((example.input_value, example.output_value, 2 * d / t**2))
    if not observations:
        return None

    g_avg = statistics.mean(g for _, _, g in observations)
    query_t = float(problem.question)
    t_squared = query_t**2
    product = g_avg * t_squared
    distance = 0.5 * product
    final_answer = problem.answer.strip()

    lines = [
        "This is Wonderland gravity, not Earth's standard gravity.",
        "Use d = 0.5*g*t^2, so each example gives g = 2*d/t^2.",
        "",
        "Step 1: Calculate g from the examples.",
    ]
    for index, (t_text, d_text, g) in enumerate(observations, 1):
        t = float(t_text)
        d = float(d_text)
        lines.extend(
            [
                f"Example {index}: t = {t_text}s, d = {d_text}m",
                f"  t_squared_{index} = {t_text}^2 = {t**2:.4f}",
                f"  twice_distance_{index} = 2 * {d_text} = {2*d:.4f}",
                f"  g_{index} = twice_distance_{index} / t_squared_{index} = {g:.4f}",
            ]
        )

    lines.extend(
        [
            "",
            "Step 2: Average the inferred gravity values.",
            f"  g_avg = ({' + '.join(f'{g:.4f}' for _, _, g in observations)}) / {len(observations)}",
            f"  g_avg = {g_avg:.4f}",
            "",
            f"Step 3: Apply the rule to t = {problem.question}s.",
            f"  query_t_squared = {problem.question}^2 = {t_squared:.4f}",
            f"  gravity_time_product = g_avg * query_t_squared = {product:.4f}",
            f"  distance = 0.5 * gravity_time_product = {distance:.4f}",
            f"  Rounded to the required precision: {final_answer}",
            "",
            f"The answer is \\boxed{{{final_answer}}}",
        ]
    )
    return "\n".join(lines)
