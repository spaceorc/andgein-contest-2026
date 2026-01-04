import os
import time
import requests
from dotenv import load_dotenv
from z3 import *

load_dotenv()

URL = "https://2026.andgein.ru/api/tasks/bulls-and-cows"
HEADERS = {"Key": os.environ["ANDGEIN_API_KEY"]}


def count_bulls_cows(guess: list[int], secret: list[int]) -> tuple[int, int]:
    """Count bulls (exact matches) and cows (present but wrong position)."""
    bulls = sum(1 for g, s in zip(guess, secret) if g == s)

    secret_counts = {}
    guess_counts = {}
    for i, (g, s) in enumerate(zip(guess, secret)):
        if g != s:
            secret_counts[s] = secret_counts.get(s, 0) + 1
            guess_counts[g] = guess_counts.get(g, 0) + 1

    cows = sum(min(guess_counts.get(n, 0), secret_counts.get(n, 0)) for n in guess_counts)
    return bulls, cows


def is_valid(candidate: list[int], guesses: list[dict]) -> bool:
    """Check if candidate satisfies all constraints."""
    for g in guesses:
        bulls, cows = count_bulls_cows(g["numbers"], candidate)
        if bulls != g["bulls"] or cows != g["cows"]:
            return False
    return True


def solve_z3(guesses: list[dict], length: int) -> list[int] | None:
    """Solve using Z3 SMT solver."""

    # Collect all numbers from guesses to determine range
    all_nums = set()
    for g in guesses:
        all_nums.update(g["numbers"])
    max_num = max(all_nums) + 50

    print(f"Using Z3 solver, max_num={max_num}")

    # Create integer variables for each position
    X = [Int(f"x_{i}") for i in range(length)]

    solver = Optimize()  # Use Optimize for lexicographic minimization

    # Constraint: all values are positive integers
    for i in range(length):
        solver.add(X[i] >= 1)
        solver.add(X[i] <= max_num)

    # Constraint: all values are distinct
    solver.add(Distinct(X))

    # Add constraints for each guess
    for g_idx, g in enumerate(guesses):
        guess_nums = g["numbers"]
        target_bulls = g["bulls"]
        target_cows = g["cows"]

        # Bulls: X[i] == guess[i]
        bulls_indicators = [If(X[i] == guess_nums[i], 1, 0) for i in range(length)]
        bulls_count = Sum(bulls_indicators)
        solver.add(bulls_count == target_bulls)

        # Cows: number appears in both but not as bull
        # For each number in guess, check if it appears in X at different position
        cows_indicators = []
        for i in range(length):
            guess_val = guess_nums[i]
            # This guess value is a cow if:
            # 1. X[i] != guess_val (not a bull at this position)
            # 2. guess_val appears somewhere in X at position j where X[j] != guess_nums[j]

            # Check if guess_val appears in X at any position j where j != i
            appears_elsewhere = Or([And(X[j] == guess_val, X[j] != guess_nums[j]) for j in range(length) if j != i])

            # It's a cow if not a bull at position i AND appears elsewhere
            is_cow = And(X[i] != guess_val, appears_elsewhere)
            cows_indicators.append(If(is_cow, 1, 0))

        cows_count = Sum(cows_indicators)
        solver.add(cows_count == target_cows)

    # Lexicographic minimization: minimize x_0, then x_1, etc.
    for i in range(length):
        solver.minimize(X[i])

    print("Solving...")
    start = time.time()

    if solver.check() == sat:
        model = solver.model()
        solution = [model[X[i]].as_long() for i in range(length)]
        print(f"Found solution in {time.time() - start:.1f}s")
        return solution
    else:
        print(f"No solution found in {time.time() - start:.1f}s")
        return None


def solve_z3_iterative(guesses: list[dict], length: int) -> list[int] | None:
    """Solve using Z3 with iterative lexicographic minimization."""

    all_nums = set()
    for g in guesses:
        all_nums.update(g["numbers"])
    max_num = max(all_nums) + 50

    print(f"Using Z3 iterative solver, max_num={max_num}")

    # Fixed values found so far
    fixed = []

    for pos in range(length):
        print(f"Finding minimum for position {pos}...")

        X = [Int(f"x_{i}") for i in range(length)]
        solver = Solver()

        # Basic constraints
        for i in range(length):
            solver.add(X[i] >= 1)
            solver.add(X[i] <= max_num)

        solver.add(Distinct(X))

        # Fix already determined positions
        for i, val in enumerate(fixed):
            solver.add(X[i] == val)

        # Add guess constraints
        for g in guesses:
            guess_nums = g["numbers"]
            target_bulls = g["bulls"]
            target_cows = g["cows"]

            # Bulls count
            bulls_indicators = [If(X[i] == guess_nums[i], 1, 0) for i in range(length)]
            solver.add(Sum(bulls_indicators) == target_bulls)

            # Cows count
            cows_indicators = []
            for i in range(length):
                guess_val = guess_nums[i]
                appears_elsewhere = Or([And(X[j] == guess_val, X[j] != guess_nums[j])
                                        for j in range(length) if j != i] + [False])
                is_cow = And(X[i] != guess_val, appears_elsewhere)
                cows_indicators.append(If(is_cow, 1, 0))

            solver.add(Sum(cows_indicators) == target_cows)

        # Binary search for minimum at this position
        lo, hi = 1, max_num
        best = None

        while lo <= hi:
            mid = (lo + hi) // 2
            solver.push()
            solver.add(X[pos] <= mid)

            if solver.check() == sat:
                model = solver.model()
                best = model[X[pos]].as_long()
                hi = mid - 1
            else:
                lo = mid + 1

            solver.pop()

        if best is None:
            print(f"No valid value for position {pos}")
            return None

        fixed.append(best)
        print(f"  Position {pos}: {best}, partial: {fixed}")

    return fixed


def main():
    response = requests.get(URL, headers=HEADERS)
    if response.status_code != 200:
        print("Error:", response.text)
        return

    task = response.json()
    level = task["current_level"]
    guesses = task["parameters"]["guesses"]

    print(f"Level {level}/{task['levels_count']}")
    print(f"Guesses: {len(guesses)}")
    print(f"Sequence length: {len(guesses[0]['numbers'])}")

    length = len(guesses[0]["numbers"])

    print("\n" + "=" * 50)
    print("Solving with Z3 (iterative lex-min)...")
    print("=" * 50)

    start = time.time()
    solution = solve_z3_iterative(guesses, length)
    elapsed = time.time() - start

    if solution:
        print(f"\nFound solution in {elapsed:.1f}s:")
        answer = " ".join(map(str, solution))
        print(answer)

        # Verify
        print("\nVerifying...")
        if is_valid(solution, guesses):
            print("Solution is VALID!")
            print(f"\nAnswer to submit: {answer}")
        else:
            print("ERROR: Solution failed verification!")
            # Debug: show which constraint failed
            for i, g in enumerate(guesses):
                bulls, cows = count_bulls_cows(g["numbers"], solution)
                if bulls != g["bulls"] or cows != g["cows"]:
                    print(f"  Guess {i}: expected ({g['bulls']}, {g['cows']}), got ({bulls}, {cows})")
    else:
        print(f"No solution found in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
