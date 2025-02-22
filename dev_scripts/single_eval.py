from openai import OpenAI

def single_eval(query: str, answer1: str, answer2: str):
    client = OpenAI()

    sys_prompt = """
    ---Role---
    You are an expert tasked with evaluating two answers to the same question based on three criteria: **Comprehensiveness**, **Diversity**, and **Empowerment**.
    """

    prompt = f"""
    You will evaluate two answers to the same question based on three criteria: **Comprehensiveness**, **Diversity**, and **Empowerment**.

    - **Comprehensiveness**: How much detail does the answer provide to cover all aspects and details of the question?
    - **Diversity**: How varied and rich is the answer in providing different perspectives and insights on the question?
    - **Empowerment**: How well does the answer help the reader understand and make informed judgments about the topic?

    For each criterion, choose the better answer (either Answer 1 or Answer 2) and explain why. Then, select an overall winner based on these three categories.

    Here is the question:
    {query}

    Here are the two answers:

    **Answer 1:**
    {answer1}

    **Answer 2:**
    {answer2}

    Evaluate both answers using the three criteria listed above and provide detailed explanations for each criterion.

    Output your evaluation in the following JSON format:

    {{
        "Comprehensiveness": {{
            "Winner": "[Answer 1 or Answer 2]",
            "Explanation": "[Provide explanation here]"
        }},
        "Diversity": {{
            "Winner": "[Answer 1 or Answer 2]",
            "Explanation": "[Provide explanation here]"
        }},
        "Empowerment": {{
            "Winner": "[Answer 1 or Answer 2]",
            "Explanation": "[Provide explanation here]"
        }},
        "Overall Winner": {{
            "Winner": "[Answer 1 or Answer 2]",
            "Explanation": "[Summarize why this answer is the overall winner based on the three criteria]"
        }}
    }}
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": prompt},
        ],
    )

    print(response.choices[0].message.content)


if __name__ == "__main__":
    # Example usage
    query = "What are the top themes in this story?"
    
    # hybrid, llama
    answer1 = """
    **Themes in A Christmas Carol**

    The story of A Christmas Carol, written by Charles Dickens, revolves around several interconnected themes that contribute to its enduring appeal. While the tale is often attributed to promoting Christmas values, its scope encompasses a broad spectrum of themes that are still relevant today. Here are some of the top themes in this story:

    ### 1. **Redemption and Personal Growth**

    The novella's central theme revolves around the transformation of Ebenezer Scrooge, a miserly and isolated character. Through his encounters with the three spirits (Past, Present, and Yet to Come), Scrooge undergoes a profound transformation, shedding his bitter, miserly personality and developing a more compassionate and generous outlook. This theme highlights the potential for individuals to change and grow.

    ### 2. **Social Justice and Economic Inequality**

    Dickens uses A Christmas Carol to critique the social and economic conditions of the Victorian era, particularly the plight of the working class. Through Scrooge's mistreatment of his clerk, Bob Cratchit, and his disdain for the poor, the novel exposes the harsh realities of poverty, forced labor, and exploitation. Dickens argues for social reform and the need for a more equitable distribution of wealth.

    ### 3. **Family, Love, and Human Connection**

    The story emphasizes the importance of family, love, and human connection in overcoming adversity. Scrooge's relationship with his nephew, Fred, serves as a contrast to his own cold demeanor. The Cratchit family's resourcefulness and love for each other illustrate the resilience of human relationships in the face of hardship.

    ### 4. **Christmas and the Human Spirit**

    The novella ultimately celebrates the spirit of Christmas, which transcends the festive rituals and material appendages to convey a deeper sense of compassion, kindness, and generosity. Christmas becomes a symbol of the human capacity for redemption, renewal, and transformation.

    ### 5. **Mortality, Fear, and the Uncertainty of Life**

    A Christmas Carol explores the themes of mortality, fear, and the unknown. The visitations from the three spirits force Scrooge to confront the darkness of death and the uncertainty of his own mortality. This serves as a commentary on the human experience, highlighting our fundamental fears and the importance of re-evaluating our priorities.

    These themes, among others, contribute to the richness and complexity of A Christmas Carol, making it a timeless classic that continues to captivate readers and audiences alike.
    """
    
    # hybrid, gpt-4o-mini
    answer2 = """
    ### Key Themes in "A Christmas Carol"

    **1. Redemption and Transformation**  
    The narrative primarily revolves around Scrooge's transformation from a miserly, solitary figure to a generous and warm individual. His journey is shaped by encounters with supernatural spirits that compel him to confront his past, present, and potential future. This theme illustrates the idea that change is possible regardless of one's past actions, embodying the spirit of holiday redemption.

    **2. The Spirit of Christmas**  
    Christmas serves as a critical backdrop for the story, representing joy, goodwill, and communal spirit. Scrooge's initial disdain for Christmas contrasts sharply with the warmth and celebration it brings to others, notably the Cratchit family. Over the course of the story, Scrooge's attitude shifts, highlighting the significance of embracing the festive spirit.

    **3. Compassion and Social Responsibility**  
    Throughout the narrative, characters such as Bob Cratchit and Tiny Tim embody themes of compassion and familial love, accentuating the importance of caring for others, particularly those in need. Scrooge's interactions with the Cratchit family push him to reflect on his societal responsibilities and the impact of his wealth and actions on the less fortunate.

    **4. Isolation vs. Connection**  
    Scrooge's initial isolation is depicted through his lonely existence, contrasting starkly with the warmth of family gatherings he witnesses. His transformation signifies the importance of human connections and the joy that comes from being part of a community, as illustrated in his eventual reconnection with his family, particularly his nephew Fred.

    **5. The Consequences of Greed**  
    The story critiques greed and selfishness, as embodied by Scrooge's character at the beginning of the narrative. His fixation on wealth leads to profound loneliness and regret. Characters like Jacob Marley, who bears the chains of his own greed, serve as cautionary figures emphasizing the moral consequences of living solely for oneself.

    These themes intertwine throughout the story, providing rich commentary on human behavior, societal values, and the enduring spirit of charity and compassion during the holiday season.
    """

    single_eval(query, answer1, answer2) 