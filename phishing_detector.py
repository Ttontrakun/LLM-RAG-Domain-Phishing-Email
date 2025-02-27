# 1. โหลดเครื่องมือที่จำเป็น
import pandas as pd
from langchain.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEndpoint
import os
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# 2. ตั้งค่า API Key
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "hf_"

# 3. อ่านข้อมูลจากไฟล์ CSV และทำความสะอาด
data = pd.read_csv("phishing_data.csv")
data["Email Text"] = data["Email Text"].fillna("").astype(str)
data["Email Type"] = data["Email Type"].fillna("").astype(str)
emails = data["Email Text"].tolist()

# 4. สร้าง Embedding Model และ Vector DB
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_db = Chroma.from_texts(emails, embedding_model)

# 5. ตั้งค่าโมเดลภาษา (LLM) - ใช้ Mixtral
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
    temperature=0.5,
    max_new_tokens=50,  # ลดเพื่อให้สั้นลง
    top_p=0.9,
    return_full_text=False,
    huggingfacehub_api_token=os.environ["HUGGINGFACEHUB_API_TOKEN"]
)

# 6. ทดสอบด้วยคำถาม
query = "Is this email phishing: 'Click here to win $1000: bit.ly/abc'?"

# ---- LLM + RAG ----
docs = vector_db.similarity_search(query)
closest_email = docs[0].page_content
closest_index = emails.index(closest_email)
closest_type = data["Email Type"].iloc[closest_index]
prompt_rag = (
    f"Based on this example email: '{closest_email}' (Type: {closest_type}), "
    f"answer this question: Is '{query}' a phishing email? "
    "Respond with: 'Yes, this is a phishing email because it contains a suspicious link and an unrealistic reward.' "
    "If not phishing, say 'No' and explain briefly in one sentence."
)
answer_rag = llm.invoke(prompt_rag)
print("LLM + RAG Answer:", answer_rag)

# ---- LLM อย่างเดียว ----
prompt_llm = (
    f"Example: 'Click here to win a prize: bit.ly/xyz' -> 'Yes, this is a phishing email because it contains a suspicious link and an unrealistic reward.'\n"
    f"Question: Is '{query}' a phishing email? "
    "Respond with: 'Yes, this is a phishing email because it contains a suspicious link and an unrealistic reward.' "
    "If not phishing, say 'No' and explain briefly in one sentence."
)
answer_llm = llm.invoke(prompt_llm)
print("LLM Only Answer:", answer_llm)

# 7. คำตอบอ้างอิงและคำนวณ BLEU Score
reference = "Yes, this is a phishing email because it contains a suspicious link and an unrealistic reward."
ref_tokens = reference.split()
rag_tokens = answer_rag.split()
llm_tokens = answer_llm.split()

smoothie = SmoothingFunction().method1
for n, weights in [(1, (1.0, 0, 0, 0)), (2, (0.5, 0.5, 0, 0)), (3, (0.33, 0.33, 0.33, 0)), (4, (0.25, 0.25, 0.25, 0.25))]:
    bleu_rag = sentence_bleu([ref_tokens], rag_tokens, weights=weights, smoothing_function=smoothie)
    bleu_llm = sentence_bleu([ref_tokens], llm_tokens, weights=weights, smoothing_function=smoothie)
    print(f"BLEU-{n} (LLM + RAG): {bleu_rag:.4f}")
    print(f"BLEU-{n} (LLM Only): {bleu_llm:.4f}")
    if bleu_rag > bleu_llm:
        print(f"--> BLEU-{n}: LLM + RAG performs better")
    elif bleu_llm > bleu_rag:
        print(f"--> BLEU-{n}: LLM Only performs better")
    else:
        print(f"--> BLEU-{n}: Both perform equally")
    print()