"""Terminal display helpers."""
import textwrap, shutil

WIDTH = min(shutil.get_terminal_size().columns, 88)

def print_answer(text: str):
    print()
    for para in text.split("\n"):
        if para.strip():
            print(textwrap.fill(para, WIDTH))
        else:
            print()

def print_header(filepath: str, n_chunks: int):
    print(f"\n{'─' * WIDTH}")
    print(f"  File   : {filepath}")
    print(f"  Chunks : {n_chunks}")
    print(f"  Type 'exit' to quit")
    print(f"{'─' * WIDTH}\n")
