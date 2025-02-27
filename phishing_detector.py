# 1. โหลดเครื่องมือที่จำเป็น
import pandas as pd
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceHub
import os
from nltk.translate.bleu_score import sentence_bleu

# 2. ตั้งค่า API Key
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

# 3. อ่านข้อมูลจากไฟล์ CSV และทำความสะอาด
data = pd.read_csv("phishing_data.csv")
data["Email Text"] = data["Email Text"].fillna("").astype(str)
data["Email Type"] = data["Email Type"].fillna("").astype(str)
emails = data["Email Text"].tolist()

# 4. สร้าง Embedding Model และ Vector DB (สำหรับ RAG)
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_db = Chroma.from_texts(emails, embedding_model)

# 5. ตั้งค่าโมเดลภาษา (LLM) - ใช้ Falcon-7B-Instruct
llm = HuggingFaceHub(repo_id="tiiuae/falcon-7b-instruct", model_kwargs={"temperature": 0.1, "max_length": 512})

# 6. ทดสอบด้วยคำถาม
query = "Is this email phishing: 'Click here to win $1000: bit.ly/abc'?"

# ---- LLM + RAG ----
docs = vector_db.similarity_search(query)
closest_email = docs[0].page_content
closest_index = emails.index(closest_email)
closest_type = data["Email Type"].iloc[closest_index]
prompt_rag = f"Based on this email: '{closest_email}' (Type: {closest_type})\nQuestion: Is '{query}' a phishing email? Answer 'Yes' or 'No', then explain briefly in 1-2 sentences."
answer_rag = llm(prompt_rag)
print("LLM + RAG Answer:", answer_rag)

# ---- LLM อย่างเดียว ----
prompt_llm = f"Question: Is '{query}' a phishing email? Answer 'Yes' or 'No', then explain briefly in 1-2 sentences."
answer_llm = llm(prompt_llm)
print("LLM Only Answer:", answer_llm)

# 7. คำตอบอ้างอิง (Reference) และคำนวณ BLEU Score
reference = "Yes, this is a phishing email because it contains a suspicious link and an unrealistic reward."
ref_tokens = reference.split()
rag_tokens = answer_rag.split()
llm_tokens = answer_llm.split()

# คำนวณ BLEU Score หลาย n-gram
for n, weights in [(1, (1.0, 0, 0, 0)), (2, (0.5, 0.5, 0, 0)), (3, (0.33, 0.33, 0.33, 0)), (4, (0.25, 0.25, 0.25, 0.25))]:
    bleu_rag = sentence_bleu([ref_tokens], rag_tokens, weights=weights)
    bleu_llm = sentence_bleu([ref_tokens], llm_tokens, weights=weights)
    print(f"BLEU-{n} (LLM + RAG): {bleu_rag:.4f}")
    print(f"BLEU-{n} (LLM Only): {bleu_llm:.4f}")
    if bleu_rag > bleu_llm:
        print(f"--> BLEU-{n}: LLM + RAG performs better")
    elif bleu_llm > bleu_rag:
        print(f"--> BLEU-{n}: LLM Only performs better")
    else:
        print(f"--> BLEU-{n}: Both perform equally")
    print()