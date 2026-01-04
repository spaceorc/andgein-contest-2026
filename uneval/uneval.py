import os
import time
import requests
from dotenv import load_dotenv
from itertools import product

load_dotenv()

URL = "https://2026.andgein.ru/api/tasks/uneval"
HEADERS = {"Key": os.environ["ANDGEIN_API_KEY"]}


def check_expression(expr: str, n: int, prohibited_characters: list[str], max_answer_length: int) -> bool:
    if len(expr) > max_answer_length:
        return False
    if any(c in expr for c in prohibited_characters):
        return False
    try:
        result = eval(expr)
        if result == n:
            return True
    except:
        return False
    return False


def generate_expressions_with_ones(max_len: int) -> list[str]:
    """Generate expressions using only 1s, operators, and parentheses."""
    # Building blocks with 1s
    ones = ["1", "11", "111", "1111", "11111"]
    ops = ["+", "-", "*", "**", "<<", ">>", "^", "|", "&", "//", "%"]

    expressions = set()

    # Single numbers
    for o in ones:
        if len(o) <= max_len:
            expressions.add(o)

    # Two operands: a op b
    for a in ones:
        for b in ones:
            for op in ops:
                expr = f"{a}{op}{b}"
                if len(expr) <= max_len:
                    expressions.add(expr)
                # With parentheses
                expr = f"({a}{op}{b})"
                if len(expr) <= max_len:
                    expressions.add(expr)

    # Three operands: a op1 b op2 c
    for a in ones:
        for b in ones:
            for c in ones:
                for op1 in ops:
                    for op2 in ops:
                        expr = f"{a}{op1}{b}{op2}{c}"
                        if len(expr) <= max_len:
                            expressions.add(expr)
                        # Various parenthesizations
                        expr = f"({a}{op1}{b}){op2}{c}"
                        if len(expr) <= max_len:
                            expressions.add(expr)
                        expr = f"{a}{op1}({b}{op2}{c})"
                        if len(expr) <= max_len:
                            expressions.add(expr)

    return list(expressions)


def solve(n: int, prohibited_characters: list[str], max_answer_length: int) -> str | None:
    """Find an expression that evaluates to n without using prohibited characters."""

    # Generate candidate expressions
    candidates = generate_expressions_with_ones(max_answer_length)

    # Check each candidate
    for expr in candidates:
        if check_expression(expr, n, prohibited_characters, max_answer_length):
            return expr

    # Try more complex expressions with allowed digits
    allowed_digits = [str(d) for d in range(10) if str(d) not in prohibited_characters]

    # Build numbers from allowed digits
    numbers = []
    for length in range(1, 6):
        for combo in product(allowed_digits, repeat=length):
            num_str = ''.join(combo)
            if num_str and num_str[0] != '0':  # No leading zeros
                numbers.append(num_str)

    ops = ["+", "-", "*", "**", "<<", ">>", "^", "|", "&", "//", "%"]

    # Two operands
    for a in numbers:
        for b in numbers:
            for op in ops:
                expr = f"{a}{op}{b}"
                if check_expression(expr, n, prohibited_characters, max_answer_length):
                    return expr

    # Three operands
    for a in numbers[:50]:  # Limit search space
        for b in numbers[:50]:
            for c in numbers[:50]:
                for op1 in ops:
                    for op2 in ops:
                        expr = f"{a}{op1}{b}{op2}{c}"
                        if check_expression(expr, n, prohibited_characters, max_answer_length):
                            return expr

    return None


if __name__ == "__main__":
    # Solve as many levels as we can!
    while True:
        response = requests.get(URL, headers=HEADERS)
        if response.status_code != 200:
            print("Task completed or error:", response.text)
            break

        task = response.json()
        n = int(task["parameters"]["number"])
        prohibited_characters = task["parameters"]["prohibited_characters"]
        max_answer_length = task["parameters"]["max_answer_length"]
        print(f"Level {task['current_level']}/{task['levels_count']}: n={n}, prohibited={prohibited_characters}, max_len={max_answer_length}")

        answer = solve(n, prohibited_characters, max_answer_length)

        if answer is None:
            print("Could not find answer!")
            break

        print(f"Answer: {answer} = {eval(answer)}")

        r = requests.post(URL, headers=HEADERS, json={"level": task["current_level"], "answer": str(answer)}).json()
        if not r["is_correct"]:
            print("Wrong:", r["checker_output"])
            break

        print(f"Level {task['current_level']} complete!")
        time.sleep(0.1)
