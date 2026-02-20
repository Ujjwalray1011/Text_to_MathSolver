import streamlit as st
import numexpr as ne
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler

## ─── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Text To Math Problem Solver",
    page_icon="🧮"
)
st.title("🧮 Text To Math Problem Solver Using LLaMA 3.1")

groq_api_key = st.sidebar.text_input(label="Groq API Key", type="password")

if not groq_api_key:
    st.info("Please add your Groq API key in the sidebar to continue.")
    st.stop()

llm = ChatGroq(model="llama-3.1-8b-instant", groq_api_key=groq_api_key)

## ─── Calculator (safe numexpr evaluator) ────────────────────────────────────
def calculate(expression: str) -> str:
    """Safely evaluates a numeric math expression using numexpr."""
    try:
        expression = expression.strip().strip("'\"")
        result = ne.evaluate(expression)
        return str(result)
    except Exception as e:
        return f"Could not evaluate '{expression}': {e}"

## ─── Wikipedia search ───────────────────────────────────────────────────────
wikipedia = WikipediaAPIWrapper()

def search_wikipedia(query: str) -> str:
    try:
        return wikipedia.run(query)
    except Exception as e:
        return f"Wikipedia search failed: {e}"

## ─── Main LLM chain (LCEL) ──────────────────────────────────────────────────
prompt = PromptTemplate(
    input_variables=["question", "wiki_context"],
    template="""You are an expert math tutor and problem solver.

Use the following Wikipedia context if relevant (it may be empty):
{wiki_context}

Solve the question below step by step, showing all working clearly.
Convert all word quantities to numbers (e.g. "a dozen" = 12).
At the end, clearly state: "The final answer is X."

Question: {question}
Answer:"""
)

chain = prompt | llm | StrOutputParser()

## ─── Session state ──────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi! I'm a Math chatbot. Ask me any math question!"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

## ─── Input ──────────────────────────────────────────────────────────────────
question = st.text_area(
    "Enter your question:",
    "I have 5 bananas and 7 grapes. I eat 2 bananas and give away 3 grapes. "
    "Then I buy a dozen apples and 2 packs of blueberries. "
    "Each pack of blueberries contains 25 berries. "
    "How many total pieces of fruit do I have at the end?"
)

if st.button("Find My Answer"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Solving your problem..."):

            st.session_state.messages.append({"role": "user", "content": question})
            st.chat_message("user").write(question)

            # Step 1 — optional Wikipedia lookup for context
            with st.expander("🔍 Step 1: Searching Wikipedia for context..."):
                wiki_context = search_wikipedia(question[:200])
                st.write(wiki_context[:500] if wiki_context else "No relevant Wikipedia context found.")

            # Step 2 — LLM solves the problem
            with st.expander("🧠 Step 2: LLM reasoning..."):
                response = chain.invoke({
                    "question": question,
                    "wiki_context": wiki_context or "None"
                })
                st.write(response)

            # Step 3 — extract and verify any math expression if present
            with st.expander("🔢 Step 3: Verifying calculation..."):
                # Ask LLM to extract just the final numeric expression
                extract_prompt = PromptTemplate(
                    input_variables=["solution"],
                    template="""From the solution below, extract ONLY the final arithmetic expression 
(e.g. '3 + 4 + 12 + 50'). Return just the expression, nothing else. 
If there is no single expression, return 'N/A'.

Solution: {solution}
Expression:"""
                )
                extract_chain = extract_prompt | llm | StrOutputParser()
                expression = extract_chain.invoke({"solution": response}).strip()
                st.write(f"Extracted expression: `{expression}`")

                if expression and expression != "N/A":
                    calc_result = calculate(expression)
                    st.write(f"Calculator verification: `{expression}` = **{calc_result}**")

            st.session_state.messages.append({"role": "assistant", "content": response})
            st.write("### ✅ Response:")
            st.success(response)