import os
import time
import requests
from dotenv import load_dotenv

load_dotenv()

URL = "https://2026.andgein.ru/api/tasks/eval"
HEADERS = {"Key": os.environ["ANDGEIN_API_KEY"]}


def check_expression(expr: str, n: int, prohibited_characters: list[str], max_answer_length: int) -> bool:
    if len(expr) > max_answer_length:
        return False
    if any(c in expr for c in prohibited_characters):
        return False
    try:
        if eval(expr) == n:
            return True
    except:
        return False
    return False


def solve(n: int, prohibited_characters: list[str], max_answer_length: int) -> str | None:
    """Find an expression that evaluates to n without using prohibited characters."""

    # Try simple additions: i + (n - i)
    for i in range(1000):
        expression = f"{i}+{n - i}"
        if check_expression(expression, n, prohibited_characters, max_answer_length):
            return expression

    # Try binary operations
    for i in range(1, 100):
        for j in range(1, 100):
            for op in ["<<", ">>", "&", "|", "^", "+", "-", "*", "//", "%", "**"]:
                expression = f"{i}{op}{j}"
                if check_expression(expression, n, prohibited_characters, max_answer_length):
                    return expression

    # Try triple operations
    for i in range(1, 10):
        for j in range(1, 10):
            for k in range(1, 10):
                for op1 in ["<<", ">>", "&", "|", "^", "+", "-", "*", "//", "%", "**"]:
                    for op2 in ["<<", ">>", "&", "|", "^", "+", "-", "*", "//", "%", "**"]:
                        expression = f"{i}{op1}{j}{op2}{k}"
                        if check_expression(expression, n, prohibited_characters, max_answer_length):
                            return expression

    return None


if __name__ == "__main__":
    # Solve as many levels as we can!
    while True:
        # Get current level
        # {"number": "2026", "prohibited_characters": ["2", "0", "6"], "max_answer_length": 7}
        response = requests.get(URL, headers=HEADERS)
        if response.status_code != 200:
            print("Task completed or error:", response.text)
            break

        task = response.json()
        n = int(task["parameters"]["number"])
        prohibited_characters = task["parameters"]["prohibited_characters"]
        max_answer_length = task["parameters"]["max_answer_length"]
        print(f"Level {task['current_level']}: n={n}, prohibited={prohibited_characters}, max_len={max_answer_length}")

        # Find answer
        answer = solve(n, prohibited_characters, max_answer_length)

        if answer is None:
            print("Could not find answer!")
            break

        print(f"Answer: {answer}")

        if not check_expression(answer, n, prohibited_characters, max_answer_length):
            print("Answer failed local check!")
            break

        r = requests.post(URL, headers=HEADERS, json={"level": task["current_level"], "answer": str(answer)}).json()
        if not r["is_correct"]:
            print(r["checker_output"])
            break

        print(f"Level {task['current_level']} complete!")
        time.sleep(0.1)
