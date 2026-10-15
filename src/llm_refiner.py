import google.generativeai as genai

class SentenceBuilder:
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')

    def construct_sentence(self, keyword_sequence: list, target_language: str = "English") -> str:
        """Translates a list of predicted signs into a continuous grammatical sentence."""
        raw_text = " -> ".join(keyword_sequence)
        prompt = (
            f"You are SignBridge AI, a sign language translation assistant. "
            f"Convert this temporal sequence of extracted sign keywords: '{raw_text}' "
            f"into a single, fluent, natural spoken sentence in {target_language}. "
            f"Return ONLY the final translated sentence, without any explanations."
        )
        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            return " ".join(keyword_sequence).capitalize() + "."