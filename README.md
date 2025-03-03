# Phishing Email Detection: LLM + RAG vs LLM Only

![Phishing Detection](https://img.shields.io/badge/Project-Phishing_Detection-blueviolet) ![Python](https://img.shields.io/badge/Python-3.9+-blue) ![License](https://img.shields.io/badge/License-MIT-green)

---

## 📝 Overview

โปรเจกต์นี้พัฒนาระบบตรวจจับ **phishing email** โดยเปรียบเทียบ 2 วิธีการใช้ **Large Language Model (LLM)**:
1. **LLM + RAG**: ใช้โมเดลภาษาคู่กับ **Retrieval-Augmented Generation (RAG)** เพื่อดึงข้อมูลจากฐานข้อมูลอีเมล แล้วตอบคำถามว่า "อีเมลนี้เป็น phishing หรือไม่" พร้อมเหตุผล
2. **LLM Only**: ใช้ LLM เพียงอย่างเดียว โดยไม่ดึงข้อมูลเพิ่มเติม ตอบจากความรู้ที่มีในโมเดล

เราวัดผลด้วย **BLEU Score** (1-gram ถึง 4-gram) เพื่อดูว่าวิธีไหนให้คำตอบใกล้เคียงกับคำตอบที่ถูกต้องมากกว่า

---

## 🎯 Goals

- ตรวจจับอีเมล phishing ด้วยเทคโนโลยี AI
- เปรียบเทียบประสิทธิภาพระหว่าง LLM + RAG และ LLM Only
- วัดผลอย่างเป็นวิทยาศาสตร์ด้วย BLEU Score

---

## 🛠️ Tools & Technologies

| **ส่วนประกอบ**          | **เครื่องมือ**                  | **หน้าที่**                              |
|--------------------------|---------------------------------|------------------------------------------|
| **ภาษาโปรแกรม**         | Python 3.9+                   | รันโค้ดทั้งหมด                          |
| **จัดการข้อมูล**         | Pandas                        | อ่านและทำความสะอาดข้อมูลจาก CSV         |
| **Embedding**            | HuggingFaceEmbeddings         | แปลงข้อความเป็นเวกเตอร์                 |
| **Vector Database**      | ChromaDB                      | เก็บและค้นหาข้อมูลอีเมลในรูปแบบเวกเตอร์ |
| **RAG Framework**        | LangChain                     | จัดการ RAG และเชื่อมต่อกับ LLM           |
| **LLM**                  | Falcon-7B-Instruct (Hugging Face) | สร้างคำตอบจากคำถาม                   |
| **วัดผล**                | NLTK (sentence_bleu)          | คำนวณ BLEU Score                       |

---

## 🚀 How It Works

1. **เตรียมข้อมูล**
   - อ่านไฟล์ `phishing_data.csv` ที่มีคอลัมน์ `"Email Text"` (ข้อความอีเมล) และ `"Email Type"` (Phishing Email หรือ Safe Email)
   - ทำความสะอาดข้อมูลด้วย Pandas

2. **สร้าง Vector Database (สำหรับ RAG)**
   - แปลงอีเมลเป็นเวกเตอร์ด้วย `HuggingFaceEmbeddings`
   - เก็บใน **ChromaDB** เพื่อให้ค้นหาข้อมูลที่ใกล้เคียงกับคำถามได้

3. **LLM + RAG**
   - ดึงอีเมลที่ใกล้เคียงที่สุดจาก ChromaDB
   - ส่ง prompt พร้อมบริบทไปให้ **Falcon-7B-Instruct** ตอบ

4. **LLM Only**
   - ส่งคำถามไปให้ Falcon-7B โดยไม่มีบริบทเพิ่มเติม

5. **วัดผล**
   - เปรียบเทียบคำตอบกับ **reference** (คำตอบที่ถูกต้อง)
   - คำนวณ BLEU Score (1-gram ถึง 4-gram)

---

## 📊 Example Output

**คำถาม:** `Is this email phishing: 'Click here to win $1000: bit.ly/abc'?`

- **LLM + RAG Answer:**  
  `Yes, it is a phishing email. It contains a suspicious link (bit.ly/abc) and offers an unrealistic reward ($1000), similar to the example.`

- **LLM Only Answer:**  
  `Yes, it might be a phishing email. The link looks suspicious.`

- **BLEU Scores:**

---

## 🚀 How It Works

1. **เตรียมข้อมูล**
   - อ่านไฟล์ `phishing_data.csv` ที่มีคอลัมน์ `"Email Text"` (ข้อความอีเมล) และ `"Email Type"` (Phishing Email หรือ Safe Email)
   - ทำความสะอาดข้อมูลด้วย Pandas

2. **สร้าง Vector Database (สำหรับ RAG)**
   - แปลงอีเมลเป็นเวกเตอร์ด้วย `HuggingFaceEmbeddings`
   - เก็บใน **ChromaDB** เพื่อให้ค้นหาข้อมูลที่ใกล้เคียงกับคำถามได้

3. **LLM + RAG**
   - ดึงอีเมลที่ใกล้เคียงที่สุดจาก ChromaDB
   - ส่ง prompt พร้อมบริบทไปให้ **Falcon-7B-Instruct** ตอบ

4. **LLM Only**
   - ส่งคำถามไปให้ Falcon-7B โดยไม่มีบริบทเพิ่มเติม

5. **วัดผล**
   - เปรียบเทียบคำตอบกับ **reference** (คำตอบที่ถูกต้อง)
   - คำนวณ BLEU Score (1-gram ถึง 4-gram)

---

## 📊 Example Output

**คำถาม:** `Is this email phishing: 'Click here to win $1000: bit.ly/abc'?`

- **LLM + RAG Answer:**  
  `Yes, it is a phishing email. It contains a suspicious link (bit.ly/abc) and offers an unrealistic reward ($1000), similar to the example.`

- **LLM Only Answer:**  
  `Yes, it might be a phishing email. The link looks suspicious.`

- **BLEU Scores:**

---

## 📋 Prerequisites

ก่อนรันโปรเจกต์ ต้องเตรียม:

1. **ติดตั้ง Python 3.9+**
 - ดาวน์โหลดจาก [python.org](https://www.python.org/)

2. **ติดตั้ง Library**
 - รันคำสั่งใน Terminal:
   ```bash
   pip install pandas langchain huggingface_hub sentence-transformers chromadb nltk