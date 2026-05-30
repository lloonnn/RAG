'''Token Budgeting'''

from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

class TokenBudget:
    def __init__(self, max_tokens_per_request: int = 4000):
        self.max_per_request = max_tokens_per_request
        self.usage = {
            "total_input" : 0, # total input tokens
            "total_output" : 0, # total output tokens
            "requests": 0 # request count
        }

    def estimate_tokens(self, text: str) -> int:
        '''Rough token estimation (actual would use tiktoken)'''
        return int(len(text.split()) * 1.3)
    
    def check_budget(self, text: str) -> tuple[bool, int]:
        '''Check if request is within budget, if not reject the api call.'''
        tokens = self.estimate_tokens(text)
        return (tokens <= self.max_per_request, tokens)
    
    def record_usage(self, input_tokens: int, output_tokens: int):
        '''Record token usage'''
        self.usage["total_input"] += input_tokens
        self.usage["total_output"] += output_tokens
        self.usage["requests"] += 1

    def get_stats(self) -> dict:
        return {
            **self.usage,
            "total_tokens": self.usage["total_input"] + self.usage["total_output"],
            "avg_per_request": (
                (self.usage["total_input"] + self.usage["total_output"])
                / max(self.usage["requests"], 1)
            ),
        }
    

class BudgetedLLM:
    """LLM with token budgeting."""

    def __init__(self, max_tokens: int = 4000):
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash",temperature=0)
        self.budget = TokenBudget(max_tokens_per_request=max_tokens)

    # @traceable(name="budgeted_invoke") from langsmith to keep track on the activity
    def invoke(self, query: str) -> str:
        # Check budget
        within_budget, tokens = self.budget.check_budget(query)

        if not within_budget:
            raise ValueError(
                f"Query exceeds token budget: {tokens} > {self.budget.max_per_request}"
            )

        # Execute
        response = self.llm.invoke(query)
        result = response.content

        # Record usage
        output_tokens = self.budget.estimate_tokens(result)
        self.budget.record_usage(tokens, output_tokens)

        return result

    def get_stats(self) -> dict:
        return self.budget.get_stats()


def demo_token_budgeting():
    """Demonstrate token budgeting."""

    llm = BudgetedLLM(max_tokens=100)

    queries = [
        "What is AI?",  # Within budget
        "Explain " + "very " * 100 + "complex topic",  # Over budget
    ]

    print("\nToken Budgeting Demo:\n")

    for query in queries:
        try:
            result = llm.invoke(query)
            print(f"✅ {query[:40]}... -> {result[:30]}...")
        except ValueError as e:
            print(f"❌ {query[:40]}... -> {e}")

    print(f"\nUsage: {llm.get_stats()}")


if __name__ == "__main__":
    demo_token_budgeting()


'''
SAMPLE OUTPUT:

Token Budgeting Demo:

✅ What is AI?... -> **Artificial Intelligence (AI)...
❌ Explain very very very very very very ve... -> Query exceeds token budget: 133 > 100

Usage: {'total_input': 3, 'total_output': 767, 'requests': 1, 'total_tokens': 770, 'avg_per_request': 770.0}

'''


# In production log the get_stats for every users and keep track which user is draining tokens
# if the user is costing more tokens, maybe can charge higher price to that user, to save stakeholder's
# cost 