
SYSTEM_PROMPT = """
            ---Role---
            You are an expert tasked with evaluating two answers to the same question based on three criteria: **Comprehensiveness**, **Diversity**, and **Empowerment**. One of them is a reference answer and another is a candidate answer.
            """

def get_prompt(query, reference, candidate):

    USER_PROMPT = f"""
    You will evaluate two answers to the same question based on three criteria: **Comprehensiveness**, **Diversity**, and **Empowerment**.

    - **Comprehensiveness**: How much detail does the answer provide to cover all aspects and details of the question?
    - **Diversity**: How varied and rich is the answer in providing different perspectives and insights on the question?
    - **Empowerment**: How well does the answer help the reader understand and make informed judgments about the topic?

    For each criterion, compare the candidate answer against the reference answer and rate it on a scale of 1-100 for each of the three criteria.

    Here is the question:
    {query}

    Here are the two answers:

    **Reference Answer:**
    {reference}

    **Candidate Answer:**
    {candidate}

    Evaluate both answers using the three criteria listed above
    and provide a score for each criterion in the following JSON format:
    {{
        "comprehensiveness": int (1-100),
        "diversity": int (1-100),
        "empowerment": int (1-100)
    }}
    """
    
    return USER_PROMPT