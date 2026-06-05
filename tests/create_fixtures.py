"""Generate a synthetic JSON fixture that mirrors the real DevGPT schema.

Run this script once to create ``tests/fixtures/sample_sharing.json``.
"""

import json
from pathlib import Path

FIXTURE_DIR = Path(__file__).parent / "fixtures"
FIXTURE_DIR.mkdir(parents=True, exist_ok=True)

data = [
    # Item 0: empty ChatgptSharing array
    {
        "Type": "PullRequest",
        "ChatgptSharing": []
    },
    # Item 1: single valid English prompt
    {
        "Type": "Issue",
        "ChatgptSharing": [
            {
                "URL": "https://chatgpt.com/share/1",
                "Conversations": [
                    {"Prompt": "Write a Python function that reverses a linked list in-place."}
                ]
            }
        ]
    },
    # Item 2: multiple prompts, mix of valid, short, and non-English
    {
        "Type": "Commit",
        "ChatgptSharing": [
            {
                "URL": "https://chatgpt.com/share/2",
                "Conversations": [
                    {"Prompt": "Write a Python function that sorts a list using merge sort."},
                    {"Prompt": "ctx"},
                    {"Prompt": "用Python写一个排序算法"},
                    {"Prompt": "Напишите функцию сортировки на Python"},
                    {"Prompt": "Write a function to compute the Fibonacci sequence recursively."}
                ]
            }
        ]
    },
    # Item 3: duplicate prompt (same as item 1)
    {
        "Type": "CodeFile",
        "ChatgptSharing": [
            {
                "URL": "https://chatgpt.com/share/3",
                "Conversations": [
                    {"Prompt": "Write a Python function that reverses a linked list in-place."}
                ]
            }
        ]
    },
    # Item 4: whitespace-only and empty-string prompts
    {
        "Type": "Discussion",
        "ChatgptSharing": [
            {
                "URL": "https://chatgpt.com/share/4",
                "Conversations": [
                    {"Prompt": "   "},
                    {"Prompt": ""},
                    {"Prompt": "\n\t"}
                ]
            }
        ]
    },
    # Item 5: additional valid English prompts for sampling tests
    {
        "Type": "HackerNews",
        "ChatgptSharing": [
            {
                "URL": "https://chatgpt.com/share/5",
                "Conversations": [
                    {"Prompt": "Implement a binary search tree with insert and search methods."},
                    {"Prompt": "Create a REST API endpoint that handles user authentication."},
                    {"Prompt": "Write a Python class that implements a stack using two queues."},
                    {"Prompt": "Design a function that validates email addresses using regex."}
                ]
            }
        ]
    }
]

output_path = FIXTURE_DIR / "sample_sharing.json"
with open(output_path, "w", encoding="utf-8") as fh:
    json.dump(data, fh, indent=2, ensure_ascii=False)

print(f"Created fixture: {output_path}  (contains {len(data)} items)")
