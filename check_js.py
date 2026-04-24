import sys, io, re
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

with open('c:/D in C/project/test_v5/phone_v2.html', encoding='utf-8') as f:
    content = f.read()

m = re.search(r'<script>(.*?)</script>', content, re.DOTALL)
js = m.group(1)
lines = js.split('\n')

# String-aware brace balance
depth = 0
in_single = False
in_double = False
in_template = False
escaped = False

for i, line in enumerate(lines, 1):
    j = 0
    while j < len(line):
        c = line[j]
        if escaped:
            escaped = False
            j += 1
            continue
        if c == chr(92):  # backslash
            escaped = True
            j += 1
            continue
        if in_single:
            if c == chr(39): in_single = False
        elif in_double:
            if c == chr(34): in_double = False
        elif in_template:
            if c == chr(96): in_template = False
        else:
            if c == chr(39): in_single = True
            elif c == chr(34): in_double = True
            elif c == chr(96): in_template = True
            elif c == '{': depth += 1
            elif c == '}': depth -= 1
        j += 1

    if depth < 0 or (i >= len(lines) - 3):
        print(f'L{i} depth={depth}: {line.rstrip()}')

print('Final depth:', depth)
