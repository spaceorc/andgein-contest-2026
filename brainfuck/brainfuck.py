import math
import os
import time
import requests
from dotenv import load_dotenv

load_dotenv()

URL = "https://2026.andgein.ru/api/tasks/brainfuck"
HEADERS = {"Key": os.environ["ANDGEIN_API_KEY"]}

def build(s: str) -> str:
    # Uses 2 cells: cell 0 = multiplier, cell 1 = accumulator
    # Pointer starts and ends at cell 0
    return build_from(s, 0)


def build_from(s: str, start_value: int = 0) -> str:
    """Build string output starting from a given accumulator value."""
    result = []
    current = start_value
    at_cell1 = False
    for ch in s:
        target = ord(ch)
        diff = target - current

        # Calculate cost of adjusting vs rebuilding
        adjust_cost = abs(diff) + (1 if not at_cell1 else 0)  # +1 for '>' if needed

        # Find best multiplication for rebuild
        best = (target + 10, target, 1)  # (cost, a, b)
        for a in range(1, 25):
            for b in range(1, 25):
                rem = target - a * b
                cost = a + b + 6 + abs(rem)  # [>...<-]> = 6 overhead + a + b + rem
                if cost < best[0]:
                    best = (cost, a, b)
        rebuild_cost = best[0] + (4 if at_cell1 else 0)  # +4 for [-]< if needed

        if adjust_cost <= rebuild_cost:
            # Adjust cell 1
            if not at_cell1:
                result.append('>')
                at_cell1 = True
            result.append('+' * diff if diff > 0 else '-' * -diff)
        else:
            # Rebuild using multiplication
            _, a, b = best
            rem = target - a * b
            if at_cell1:
                result.append('[-]<')  # clear cell 1, go to cell 0
            result.append('+' * a)
            result.append('[>' + '+' * b + '<-]>')  # multiply, end at cell 1
            at_cell1 = True
            result.append('+' * rem if rem > 0 else '-' * -rem)
        current = target
        result.append('.')
    # Return pointer to cell 0
    if at_cell1:
        result.append('<')
    return ''.join(result)


def copy(offset: int) -> str:
    """Copy current cell to cell at offset. Uses cell at offset+1 as temp (or +1 if offset=-1).
    Pointer returns to original position."""
    if offset == 0:
        return ""

    # Temp cell: offset+1, unless that's 0 (when offset=-1), then use 1
    temp = 1 if offset == -1 else offset + 1

    def go(from_pos: int, to_pos: int) -> str:
        diff = to_pos - from_pos
        return '>' * diff if diff > 0 else '<' * (-diff)

    # Phase 1: move cell 0 to both target and temp
    phase1 = '[-' + go(0, offset) + '+' + go(offset, temp) + '+' + go(temp, 0) + ']'

    # Phase 2: move temp back to cell 0
    phase2 = go(0, temp) + '[-' + go(temp, 0) + '+' + go(0, temp) + ']' + go(temp, 0)

    return phase1 + phase2


def clear() -> str:
    """Clear current cell to 0. Pointer stays in place."""
    return '[-]'


def copy_from(offset: int) -> str:
    """Copy cell at offset to current cell. Uses temp cell. Pointer returns to original position.
    Current cell must be 0 before copy (or use clear() first)."""
    if offset == 0:
        return ""

    # Temp: offset+1 if offset > 0, else 1 (to avoid temp=0 when offset=-1)
    temp = offset + 1 if offset > 0 else 1

    def go(from_pos: int, to_pos: int) -> str:
        diff = to_pos - from_pos
        return '>' * diff if diff > 0 else '<' * (-diff)

    # Phase 1: go to source, move source to both target (0) and temp
    phase1 = go(0, offset) + '[-' + go(offset, 0) + '+' + go(0, temp) + '+' + go(temp, offset) + ']'

    # Phase 2: move temp back to source
    phase2 = go(offset, temp) + '[-' + go(temp, offset) + '+' + go(offset, temp) + ']'

    # Return to cell 0
    phase3 = go(temp, 0)

    return phase1 + phase2 + phase3


def if_(body: str) -> str:
    """If current cell != 0, execute body once. Clears the cell."""
    return '[' + body + '[-]]'


def subtract(n: int) -> str:
    """Subtract n from current cell using multiplication for efficiency.
    Uses cell+1 as temp. Pointer returns to original position."""
    if n <= 0:
        return ''
    if n <= 15:
        return '-' * n

    # Find best a*b close to n
    best = (n, n, 1)  # (cost, a, b)
    for a in range(1, 25):
        for b in range(1, 25):
            rem = n - a * b
            cost = a + b + 4 + abs(rem)  # loop overhead + remainder adjustment
            if cost < best[0]:
                best = (cost, a, b)

    _, a, b = best
    rem = n - a * b  # positive means we need to subtract more, negative means we subtracted too much

    # Pattern: >+++a[<---b>-]< then adjust remainder
    code = '>' + '+' * a + '[<' + '-' * b + '>-]<'
    if rem > 0:
        code += '-' * rem
    elif rem < 0:
        code += '+' * (-rem)

    return code


def if_else(if_body: str, else_body: str) -> str:
    """If current cell != 0, execute if_body, else execute else_body.
    Uses cell+1 as flag. Body runs at cell+2. Clears cells 0,1. Pointer returns to original."""
    return (
        '>+<'                              # set flag=1 at cell+1
        '[>>' + if_body + '<<>-<[-]]'      # if: move to cell+2, run body, back, clear flag, clear cond
        '>[>' + else_body + '<-]<'         # else: move to cell+2, run body, back, clear flag, back to cell 0
    )


def switch(cases: dict[int, str], else_body: str = "") -> str:
    """Switch on current cell value. cases = {value: body}.
    Cell 0 = value, Cell 1 = matched flag. Bodies run at cell 2+.
    Clears cells 0,1. Pointer returns to cell 0."""
    if not cases:
        return else_body

    sorted_cases = sorted(cases.items())  # [(v1, body1), (v2, body2), ...]

    # Pattern: -(v1)>+<[>-<  -(v2-v1)>+<[>-<  ... [>-< else_body ]>[bodyN-]<  ...]>[body2-]<  ]>[body1-]<
    # - Subtract case value
    # - Set flag=1 (assume match)
    # - If non-zero: clear flag, continue to next case
    # - After all checks, if flag set: run corresponding body

    def build_chain(idx: int) -> str:
        if idx >= len(sorted_cases):
            # Innermost: default case - clear cell, run body
            return '[-]>>' + else_body + '<<'

        val, body = sorted_cases[idx]
        diff = val - (sorted_cases[idx-1][0] if idx > 0 else 0)
        inner = build_chain(idx + 1)

        return (
            subtract(diff) +      # subtract to check this case (uses multiplication)
            '>+<' +               # set flag=1 (assume match)
            '[>-<' +              # if non-zero: clear flag
            inner +               # check remaining cases
            ']' +                 # end non-zero block
            '>[>>' + body + '<<-]<'  # if flag: run body at cell+2, clear flag
        )

    return build_chain(0)


def build_matcher(mapping: dict[str, str]) -> str:
    """Build a Brainfuck program that reads input and outputs based on mapping.

    Automatically finds the minimal distinguishing prefix for each input.
    Groups inputs by characters, skips common prefixes, switches where needed.

    Example:
        mapping = {
            "United States": "Santa Claus",
            "United Kingdom": "Father Christmas",
            "Russia": "Ded Moroz",
            "Netherlands": "Sinterklaas",
            "Norway": "Julenissen",
            "Finland": "Joulupukki",
            "Germany": "Weihnachtsmann",
        }
        prog = build_matcher(mapping)
        run(prog, "Russia")  # -> "Ded Moroz"
    """
    from collections import defaultdict

    items = list(mapping.items())

    def generate(items: list, pos: int) -> str:
        """Generate code to distinguish items starting at character position pos."""
        if len(items) == 1:
            # Only one possibility left - output it
            return build(items[0][1])

        # Group by character at position pos
        groups = defaultdict(list)
        for key, value in items:
            char = key[pos] if pos < len(key) else '\0'
            groups[char].append((key, value))

        if len(groups) == 1:
            # All items share the same char at this position - read and skip
            return ',' + generate(items, pos + 1)

        # Multiple groups - need to read and switch
        cases = {}
        for char, group in groups.items():
            cases[ord(char)] = generate(group, pos + 1)

        return ',' + switch(cases, '')

    return optimize(generate(items, 0))


def build_matcher_v2(mapping: dict[str, str]) -> str:
    """Smarter matcher that pre-reads chars and picks optimal switch position.

    Memory layout: cells 0,1,2,... hold chars read, switch at best position.
    For collisions, uses secondary checks on other cell values.
    """
    from collections import defaultdict

    keys = list(mapping.keys())
    max_len = max(len(k) for k in keys)

    # Find position with fewest collisions
    def count_collisions(pos):
        groups = defaultdict(list)
        for k in keys:
            ch = k[pos] if pos < len(k) else '\0'
            groups[ch].append(k)
        return sum(1 for g in groups.values() if len(g) > 1)

    best_pos = min(range(min(max_len, 5)), key=count_collisions)
    num_collisions = count_collisions(best_pos)

    # If best position has many collisions, try positions 0,1,2... sequentially
    if num_collisions > 2 or best_pos == 0:
        return build_matcher(mapping)

    # Read chars into cells 0..best_pos, switch on cell[best_pos]
    read_chars = ',>' * best_pos + ','  # read best_pos+1 chars, end at cell[best_pos]

    # Group by char at best_pos
    groups = defaultdict(list)
    for k, v in mapping.items():
        ch = k[best_pos] if best_pos < len(k) else '\0'
        groups[ch].append((k, v))

    def my_if_else(if_body, else_body, flag_offset=5, body_offset=6):
        go_flag = '>' * flag_offset
        back_flag = '<' * flag_offset
        go_body = '>' * body_offset
        back_body = '<' * body_offset
        return (
            go_flag + '+' + back_flag
            + '[' + go_flag + '-' + back_flag + '[-]'
            + go_body + if_body + back_body + ']'
            + go_flag + '[>' + else_body + '<-]' + back_flag
        )

    def handle_collision(items, check_pos):
        """Handle collision by checking character at check_pos."""
        if len(items) != 2:
            # For now, just take first item for >2 collisions
            return build(items[0][1])

        k0, v0 = items[0]
        k1, v1 = items[1]
        ch0 = ord(k0[check_pos]) if check_pos < len(k0) else 0
        ch1 = ord(k1[check_pos]) if check_pos < len(k1) else 0

        # We're at body position (cell[best_pos]+3)
        # Go to cell[check_pos], subtract one value, if zero -> one output, else -> other
        nav_to = '<' * (best_pos + 3 - check_pos)  # from body to check_pos
        nav_back = '>' * (best_pos + 3 - check_pos)

        # Clear adjacent cell for subtract temp
        clear_temp = '>[-]<' if check_pos + 1 <= best_pos else ''

        return (
            nav_to + clear_temp
            + subtract(ch0)  # if items[0], cell=0; if items[1], cell=ch1-ch0
            + my_if_else(build(v1), build(v0))
            + nav_back
        )

    cases = {}
    for ch, items in groups.items():
        if len(items) == 1:
            cases[ord(ch)] = build(items[0][1])
        else:
            # Collision - find a position that distinguishes them
            for check_pos in range(best_pos):
                chars_at_pos = set(k[check_pos] if check_pos < len(k) else '\0' for k, v in items)
                if len(chars_at_pos) == len(items):
                    cases[ord(ch)] = handle_collision(items, check_pos)
                    break
            else:
                # No good check position found, use first item
                cases[ord(ch)] = build(items[0][1])

    return optimize(read_chars + switch(cases, ''))


def shrink(code: str) -> str:
    return ''.join(c for c in code if c in "><+-.,[]")


def optimize(code: str) -> str:
    """Remove redundant sequences like ><, <>, +-, -+, and trailing pointer moves."""
    code = shrink(code)
    while True:
        prev = code
        # Cancel out adjacent opposite operations
        code = code.replace('><', '')
        code = code.replace('<>', '')
        code = code.replace('+-', '')
        code = code.replace('-+', '')
        if code == prev:
            break
    # Strip trailing pointer moves (program is done, position doesn't matter)
    code = code.rstrip('<>')
    return code


def pretty(code: str, indent: str = "  ") -> str:
    code = [c for c in code if c in "><+-.,[]"]
    lines = []
    level = 0
    line = ""
    for c in code:
        if c == ']':
            if line:
                lines.append(indent * level + line)
                line = ""
            level -= 1
            lines.append(indent * level + c)
        elif c == '[':
            line += c
            lines.append(indent * level + line)
            line = ""
            level += 1
        else:
            line += c
    if line:
        lines.append(indent * level + line)
    return '\n'.join(lines)


def run(code: str, input_data: str = "") -> str:
    code = [c for c in code if c in "><+-.,[]"]
    tape = [0] * 30000
    ptr = 0
    ip = 0
    input_idx = 0
    output = []

    # Precompute bracket matches
    brackets = {}
    stack = []
    for i, c in enumerate(code):
        if c == '[':
            stack.append(i)
        elif c == ']':
            j = stack.pop()
            brackets[j] = i
            brackets[i] = j

    while ip < len(code):
        cmd = code[ip]
        if cmd == '>':
            ptr = (ptr + 1) % len(tape)
        elif cmd == '<':
            ptr = (ptr - 1) % len(tape)
        elif cmd == '+':
            tape[ptr] = (tape[ptr] + 1) % 256
        elif cmd == '-':
            tape[ptr] = (tape[ptr] - 1) % 256
        elif cmd == '.':
            output.append(chr(tape[ptr]))
        elif cmd == ',':
            if input_idx < len(input_data):
                tape[ptr] = ord(input_data[input_idx])
                input_idx += 1
            else:
                tape[ptr] = 0
        elif cmd == '[':
            if tape[ptr] == 0:
                ip = brackets[ip]
        elif cmd == ']':
            if tape[ptr] != 0:
                ip = brackets[ip]
        ip += 1

    return ''.join(output)



def symbols_level25():
    """Level 25: Element symbols -> names - char0 switch with char1 sub-check.

    1. Read char0, copy to cell1 and cell2
    2. Switch on char0 (cell1)
    3. For collision cases (H, C, A), read char1 and sub-branch
    4. Reuse stored chars where possible
    """
    # Read char0 into cell0, move to cell1 (for switch) and cell2 (preserved copy)
    # ,[->>+>+<<<]>>> puts char0 at cell2 and cell3, ptr at cell3
    # Actually let's use: ,[-<+>>+<] which reads into cell1, moves to cell0 and cell2
    # Then we're at cell1 (empty), go to cell0 for switch

    # Setup: read char0, store at cell0 and cell2, ptr at cell0
    code = '>,[-<+>>+<]<'  # read at cell1, move to cell0 and cell2, go to cell0

    # char0 values:
    # A(65): Ag, Au - collision
    # C(67): C, Cu - collision
    # F(70): Fe only
    # H(72): H, He - collision
    # O(79): O only

    # From body at cell3:
    # - char0 copy at cell2: << to reach from cell3

    def body_simple(output, char0_val, adj=0):
        """Just reuse char0 (optionally adjusted) and build the rest."""
        # Body at cell3, char0 copy at cell2 = 1 cell left
        nav_to = '<'   # cell3 to cell2
        nav_back = '>'
        if adj == 0:
            # Reuse char0 directly
            return nav_to + '.' + nav_back + build(output[1:])
        else:
            # Adjust then print
            delta = '+' * adj if adj > 0 else '-' * (-adj)
            return nav_to + delta + '.' + nav_back + build(output[1:])

    def body_with_char1_check(char1_cases, char0_copy_at, body_offset):
        """Read char1 and switch on it. char0 preserved at char0_copy_at."""
        # At body position, read char1 then switch
        inner_code = ','  # read char1 at current position
        inner_code += switch(char1_cases)
        return inner_code

    # Inner switch: outer body at cell3 reads char1, switches on it
    # Inner body at cell3+3 = cell6
    # char0 at cell2, from cell6 that's 4 left
    # char1 was at cell3 but consumed by switch - can't directly reuse

    def inner_reuse_char0(rest):
        """From inner body (cell6), navigate to char0 at cell2, print, build rest."""
        return '<<<<.>>>>' + build(rest)

    def inner_reuse_char0_adjusted(adj, rest):
        """Adjust char0 and print, build rest."""
        delta = '+' * adj if adj > 0 else '-' * (-adj)
        return '<<<<' + delta + '.>>>>' + build(rest)

    def inner_body_helium():
        """Special case for Helium - char0=H at cell2, char1=e was consumed.
        Just build full output starting from char0."""
        # Navigate to cell2 (char0=H), print, build rest
        return '<<<<.>>>>' + build('elium')

    cases = {
        65: body_with_char1_check({  # A -> Ag or Au
            103: inner_reuse_char0_adjusted(18, 'ilver'),  # g -> Silver (A+18=S)
            117: inner_reuse_char0_adjusted(6, 'old'),     # u -> Gold (A+6=G)
        }, 2, 3),
        67: body_with_char1_check({  # C -> C or Cu
            0: inner_reuse_char0('arbon'),    # end -> Carbon
            117: inner_reuse_char0('opper'),  # u -> Copper
        }, 2, 3),
        70: body_simple('Iron', 70, 3),       # F+3=I -> Iron
        72: body_with_char1_check({  # H -> H or He
            0: inner_reuse_char0('ydrogen'),   # end -> Hydrogen
            101: inner_body_helium(),          # e -> Helium
        }, 2, 3),
        79: body_simple('Oxygen', 79),        # O -> Oxygen
    }

    code += switch(cases)
    return optimize(code)


def symbols_level25_nested():
    """Level 25: Nested switch - switch on char0, then char1 for collisions.

    char0 groups:
    - 65 (A): Ag(g=103), Au(u=117)
    - 67 (C): C(0), Cu(u=117)
    - 70 (F): Fe only
    - 72 (H): H(0), He(e=101)
    - 79 (O): O only
    """
    # Read char0, store copy, switch on it
    code = ',[->>+>+<<<]>>>'  # char0 at cell3, copy at cell2, ptr at cell3

    # For inner check, read char1 and switch on it
    # After outer switch body at cell6, read char1 at cell6

    def body_simple(output):
        """Just output the full string"""
        return build(output)

    def body_with_char1_check(char1_cases):
        """Read char1 and switch on it. Body at outer_cell+3 = cell6"""
        # Read char1 at current position (cell6), switch on it
        inner_code = ','  # read char1
        inner_code += switch(char1_cases)
        return inner_code

    # Build cases for char0
    cases = {
        65: body_with_char1_check({  # A -> Ag or Au
            103: build('Silver'),    # g -> Silver
            117: build('Gold'),      # u -> Gold
        }),
        67: body_with_char1_check({  # C -> C or Cu
            0: build('Carbon'),      # no second char -> Carbon
            117: build('Copper'),    # u -> Copper
        }),
        70: body_simple('Iron'),     # F -> Iron (Fe only)
        72: body_with_char1_check({  # H -> H or He
            0: build('Hydrogen'),    # no second char -> Hydrogen
            101: build('Helium'),    # e -> Helium
        }),
        79: body_simple('Oxygen'),   # O -> Oxygen only
    }

    code += switch(cases)
    return optimize(code)


def test_symbols_level25():
    """Test the level 25 solution"""
    mapping = {
        'H': 'Hydrogen',
        'He': 'Helium',
        'C': 'Carbon',
        'Cu': 'Copper',
        'O': 'Oxygen',
        'Fe': 'Iron',
        'Ag': 'Silver',
        'Au': 'Gold',
    }

    # Test sum approach
    code1 = symbols_level25()
    print(f"Sum approach: {len(code1)} chars (limit: 729)")

    # Test nested approach
    code2 = symbols_level25_nested()
    print(f"Nested approach: {len(code2)} chars")

    # Test standard matchers
    code3 = build_matcher(mapping)
    print(f"Standard matcher: {len(code3)} chars")

    code4 = build_matcher_v2(mapping)
    print(f"Matcher v2: {len(code4)} chars")

    # Pick best
    best_code = min([code1, code2, code3, code4], key=len)
    print(f"\nBest: {len(best_code)} chars")

    # Test best
    all_pass = True
    for inp, expected in mapping.items():
        output = run(best_code, inp)
        if output != expected:
            print(f"  {inp}: expected {expected!r}, got {output!r}")
            all_pass = False
    print(f"All tests pass: {all_pass}")

    return best_code


if __name__ == "__main__":
    # Quick debug - just generate, don't run
    code = symbols_level25()
    print(f"Generated: {len(code)} chars")
    print(code[:200] + "...")

    # Quick test with just one input
    out = run(code, 'O')
    print(f"O -> {out!r}")
    exit(0)

    # Решаем столько уровней, сколько можем!
    while True:
        # Получаем текущий уровень...
        # {"number": "2026", "prohibited_characters": ["2", "0", "6"], "max_answer_length": 7}
        task = requests.get(URL, headers=HEADERS).json()
        mapping = task["parameters"]["mapping"]
        max_answer_length = task["parameters"]["max_program_length"]
        print(mapping, max_answer_length)

        # Ищем ответ - пробуем оба матчера, берём короче
        answer1 = build_matcher(mapping)
        answer2 = build_matcher_v2(mapping)
        answer = answer1 if len(answer1) <= len(answer2) else answer2
        print(f"v1: {len(answer1)}, v2: {len(answer2)}, using: {len(answer)}")
        print(answer)

        if len(answer) > max_answer_length:
            print("Ответ слишком длинный:", len(answer), "символов, максимум", max_answer_length)
            exit(0)

        for k, v in mapping.items():
            output = run(answer, k)
            if output != v:
                print(f"Ответ не прошёл проверку локально для входа {k!r}: ожидалось {v!r}, получили {output!r}")
                exit(0)

        r = requests.post(URL, headers=HEADERS, json={"level": task["current_level"], "answer": str(answer)}).json()
        if not r["is_correct"]:
            print(r["checker_output"])
            exit(0)

        print("Отлично, ещё один уровень позади:", task["current_level"])

        # Отдохнём чуть-чуть...
        time.sleep(0.1)
