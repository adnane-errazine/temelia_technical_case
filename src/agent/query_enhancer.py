from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
# from src.config import OPENAI_API_KEY

# Prompt template for reformulating queries
system_template = """
Tu es **TemelIA**, l'assistante experte développée par **Temelion**, une entreprise basée à Paris qui révolutionne la conception des bâtiments grâce à l'intelligence artificielle.
Tu es spécialisée dans les domaines de l'ingénierie du bâtiment, notamment en :
- électricité  
- plomberie  
- chauffage/climatisation  
- gestion des spécifications  
- conception fonctionnelle  
- vérification des conceptions  
- génération de modèles 3D  
Tu vas recevoir une **question d'un utilisateur** relative à ces domaines.
## Objectif
Reformule la question en **une seule question claire et précise** en français, destinée à alimenter **un moteur de recherche documentaire**.
---
## Règles de reformulation
- La reformulation **doit être claire et précise**.
- Elle **ne doit pas dénaturer** la question initiale.
- Elle doit **englober tous les sujets abordés** dans la question pour maximiser la qualité de la recherche.
- Si la question est déjà claire, **tu peux la renvoyer telle quelle**.
"""

human_template = (
    "## Question de l'utilisateur\nQuestion: {question}\n## Reformulation\nTa réponse:"
)


chat_prompt = ChatPromptTemplate.from_messages(
    [("system", system_template), ("human", human_template)]
)


enhancer_llm = ChatOpenAI(model_name="gpt-4.1-nano", temperature=0)


def enhance_query(question_input: str) -> str:
    messages = chat_prompt.format_messages(question=question_input)

    response = enhancer_llm.invoke(messages)

    return response.content.strip()


if __name__ == "__main__":
    # Example question
    question = (
        "Quelles sont les exigences minimum requises pour les dispositifs d'isolation?"
    )
    enhanced_question = enhance_query(question)
    print(f"Original Question: {question}")
    print(f"Enhanced Question: {enhanced_question}")
