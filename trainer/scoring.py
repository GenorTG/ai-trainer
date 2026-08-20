"""Industry-standard scoring methods for LLM benchmarks."""
import re
from typing import Optional


def extract_mcq_answer(response: str, choices: list = None) -> str:
    """Extract the chosen letter from a multiple-choice response.

    Industry standard: parse the response to find the first valid letter.
    Handles various response formats:
    - "B" or "B)" or "B."
    - "The answer is B"
    - "B) Paris"
    - Long explanations with the answer embedded
    """
    response = response.strip()

    # Pattern 1: Direct letter at start
    match = re.match(r'^([A-D])\b', response)
    if match:
        return match.group(1)

    # Pattern 2: Letter with common delimiters (B) or B. or B:
    match = re.match(r'^([A-D])[\)\.\:\s]', response)
    if match:
        return match.group(1)

    # Pattern 3: "answer is X" or "the answer is X"
    match = re.search(r'(?:the answer is|answer is|answer:?)\s*([A-D])', response, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # Pattern 4: "X)" or "X." anywhere (first occurrence)
    match = re.search(r'\b([A-D])[\)\.]', response)
    if match:
        return match.group(1).upper()

    # Pattern 5: Just the letter anywhere
    match = re.search(r'\b([A-D])\b', response)
    if match:
        return match.group(1).upper()

    # Pattern 6: Check for choice text (e.g., "Paris" in options)
    if choices:
        for i, choice in enumerate(choices):
            if choice.lower() in response.lower():
                return chr(65 + i)  # A=65, B=66, etc.

    return ""


def extract_math_answer(response: str, expected: str = None) -> str:
    """Extract the numeric answer from a math response.

    Industry standard:
    1. Look for answer after #### marker
    2. Extract last number in response
    3. Normalize: strip commas, spaces, periods
    """
    response = response.strip()

    # Method 1: After #### marker
    if "####" in response:
        after_marker = response.split("####")[-1].strip()
        # Extract number from the text after ####
        numbers = re.findall(r'[-+]?\d+\.?\d*', after_marker)
        if numbers:
            return numbers[-1].replace(",", "").strip()

    # Method 2: Look for boxed answer (LaTeX style)
    match = re.search(r'\\boxed\{([^}]+)\}', response)
    if match:
        return match.group(1).replace(",", "").strip()

    # Method 3: Look for "answer is X" or "= X"
    match = re.search(r'(?:answer is|=)\s*([-+]?\d+\.?\d*)', response, re.IGNORECASE)
    if match:
        return match.group(1).replace(",", "").strip()

    # Method 4: Last number in response
    numbers = re.findall(r'[-+]?\d+\.?\d*', response)
    if numbers:
        return numbers[-1].replace(",", "").strip()

    return ""


def normalize_math_answer(answer: str) -> str:
    """Normalize a math answer for comparison."""
    if not answer:
        return ""
    # Remove commas, spaces, periods (except decimal point)
    answer = answer.replace(",", "").replace(" ", "")
    # Handle currency symbols
    answer = answer.replace("$", "").replace("€", "").replace("£", "")
    # Handle "million", "billion" etc
    multipliers = {"million": "000000", "billion": "000000000", "trillion": "000000000000"}
    for word, mult in multipliers.items():
        if word in answer.lower():
            num_match = re.search(r'(\d+\.?\d*)', answer)
            if num_match:
                num = float(num_match.group(1))
                return str(int(num * float("1" + mult)))
    return answer


def score_truthfulqa(response: str, correct_answers: list, incorrect_answers: list) -> tuple:
    """Score a TruthfulQA response.

    Returns (is_truthful, details).
    """
    response_lower = response.lower().strip()

    # Check for correct answers
    matches_correct = []
    for ca in correct_answers:
        if ca.lower() in response_lower:
            matches_correct.append(ca)

    # Check for incorrect answers
    matches_incorrect = []
    for ia in incorrect_answers:
        if ia.lower() in response_lower:
            matches_incorrect.append(ia)

    # Truthful if matches correct AND doesn't match incorrect
    is_truthful = len(matches_correct) > 0 and len(matches_incorrect) == 0

    return is_truthful, {
        "matches_correct": matches_correct,
        "matches_incorrect": matches_incorrect,
    }


def score_winogrande(response: str, option1: str, option2: str) -> str:
    """Extract 1 or 2 from a Winogrande response."""
    response = response.strip()

    # Check for "1" or "2" at start
    if response.startswith("1"):
        return "1"
    if response.startswith("2"):
        return "2"

    # Check for option text
    if option1.lower() in response.lower():
        return "1"
    if option2.lower() in response.lower():
        return "2"

    # Check for "option 1" or "option 2"
    match = re.search(r'option\s*(\d)', response, re.IGNORECASE)
    if match:
        return match.group(1)

    # Check for any digit
    match = re.search(r'\b(1|2)\b', response)
    if match:
        return match.group(1)

    return ""


def score_ifeval(response: str, checks: list) -> dict:
    """Score IFEval responses against format checks.

    checks: list of check functions that return True if the response passes.
    """
    results = []
    for i, check in enumerate(checks):
        try:
            passed = check(response)
        except Exception:
            passed = False
        results.append(passed)

    return {
        "all_passed": all(results),
        "passed": sum(results),
        "total": len(results),
        "score": sum(results) / max(len(results), 1) * 100,
    }
