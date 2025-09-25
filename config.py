LANGUAGE_SET = ["Chinese", "English", "French", "Spanish", "Portuguese", "German", "Italian", "Russian", "Japanese", "Korean", "Vietnamese", "Thai", "Arabic", "Hindi", "Turkish", "Bengali"]


num_to_let = {
    "1": "A",
    "2": "B",
    "3": "C",
    "4": "D" 
}

let_to_num = {
    "A": 0,
    "B": 1,
    "C": 2,
    "D": 3
}

def pick_out_lang(data_file):
    if "_NR" in data_file: 
        df_lang = data_file.split('_')[-4]
    else:
        df_lang = data_file.split('_')[-3]
    return df_lang

def set_split(temp_dataset):
    global NUM_TEST, NUM_TRAIN
    if "culture" not in temp_dataset:
        NUM_TRAIN = 20000
        NUM_TEST = 10000
    else:
        NUM_TRAIN = 5000
        NUM_TEST = 1533

EVALUATION_LLMS = [
    "Qwen/Qwen3-0.6B",
    "google/gemma-3-1b-it",
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
    "google/gemma-3-12b-it",
    "CohereLabs/aya-23-8B",
    "Qwen/Qwen3-8B",
    "Qwen/Qwen3-14B",
]

EVALUATION_DATASETS = [
    # "social_iqa",
    # "culture_atlas",
    "blend",
]

NUM_TRAIN = 20000
NUM_TEST = 10000



TRANSLATED_INSTRUCTIONS = {
    "English": lambda input: f"""Question: {input['input']}
Answer choices: {input['choices']}.
Think about it in English, and then select one of the answer choices. Fill out the following JSON:
{{
    "reasoning_in_English": "<your reasoning steps in English>",
    "final_answer": "<output answer here>"
}}
""",

    "Hindi": lambda input: f"""प्रश्न: {input['input']}
उत्तर विकल्प: {input['choices']}.
इसके बारे में हिंदी में सोचें, और फिर एक उत्तर विकल्प चुनें। निम्नलिखित JSON भरें:
{{
    "reasoning_in_Hindi": "<अपने तर्क कदम हिंदी में लिखें>",
    "final_answer": "<यहाँ अंतिम उत्तर लिखें>"
}}
""",

    "Turkish": lambda input: f"""Soru: {input['input']}
Cevap seçenekleri: {input['choices']}.
Türkçe olarak düşünün ve ardından cevap seçeneklerinden birini seçin. Aşağıdaki JSON'u doldurun:
{{
    "reasoning_in_Turkish": "<Türkçe akıl yürütme adımlarınız>",
    "final_answer": "<çıktı cevabı buraya>"
}}
""",

    "Bengali": lambda input: f"""প্রশ্ন: {input['input']}
উত্তরের বিকল্পসমূহ: {input['choices']}।
বাংলায় ভেবে দেখুন, তারপর একটি উত্তর বিকল্প নির্বাচন করুন। নিচের JSON পূরণ করুন:
{{
    "reasoning_in_Bengali": "<আপনার বিশ্লেষণমূলক ধাপগুলি বাংলায় লিখুন>",
    "final_answer": "<এখানে চূড়ান্ত উত্তর লিখুন>"
}}
""",

    "Farsi": lambda input: f"""سؤال: {input['input']}
گزینه‌های پاسخ: {input['choices']}.
به فارسی فکر کنید و سپس یکی از گزینه‌های پاسخ را انتخاب کنید. بخش JSON زیر را پر کنید:
{{
    "reasoning_in_Farsi": "<مراحل استدلال خود را به فارسی بنویسید>",
    "final_answer": "<پاسخ نهایی را اینجا وارد کنید>"
}}
""",


    "Chinese": lambda input: f"""问题：{input['input']}
答案选项：{input['choices']}。
请用中文思考，然后从答案选项中选择一个。请按以下 JSON 格式填写：
{{
    "reasoning_in_Chinese": "<你的中文推理步骤>",
    "final_answer": "<在此处填写你的答案>"
}}
""",

    "French": lambda input: f"""Question : {input['input']}
Options de réponse : {input['choices']}.
Réfléchissez en français, puis sélectionnez l'une des options de réponse. Veuillez remplir le JSON suivant :
{{
    "reasoning_in_French": "<vos étapes de raisonnement en français>",
    "final_answer": "<votre réponse finale ici>"
}}
""",

    "Spanish": lambda input: f"""Pregunta: {input['input']}
Opciones de respuesta: {input['choices']}.
Piensa en español y luego selecciona una de las opciones de respuesta. Completa el siguiente JSON:
{{
    "reasoning_in_Spanish": "<tus pasos de razonamiento en español>",
    "final_answer": "<tu respuesta final aquí>"
}}
""",

    "Portuguese": lambda input: f"""Pergunta: {input['input']}
Opções de resposta: {input['choices']}.
Pense em português e, em seguida, selecione uma das opções de resposta. Preencha o seguinte JSON:
{{
    "reasoning_in_Portuguese": "<seus passos de raciocínio em português>",
    "final_answer": "<sua resposta final aqui>"
}}
""",

    "German": lambda input: f"""Frage: {input['input']}
Antwortmöglichkeiten: {input['choices']}.
Denken Sie auf Deutsch nach und wählen Sie dann eine der Antwortmöglichkeiten aus. Füllen Sie das folgende JSON aus:
{{
    "reasoning_in_German": "<Ihre Begründungsschritte auf Deutsch>",
    "final_answer": "<Ihre endgültige Antwort hier>"
}}
""",

    "Italian": lambda input: f"""Domanda: {input['input']}
Scelte di risposta: {input['choices']}.
Rifletti in italiano, quindi seleziona una delle opzioni di risposta. Compila il seguente JSON:
{{
    "reasoning_in_Italian": "<i tuoi passaggi di ragionamento in italiano>",
    "final_answer": "<la tua risposta finale qui>"
}}
""",

    "Russian": lambda input: f"""Вопрос: {input['input']}
Варианты ответа: {input['choices']}.
Обдумайте на русском, а затем выберите один из вариантов ответа. Заполните следующий JSON:
{{
    "reasoning_in_Russian": "<ваши шаги рассуждения на русском>",
    "final_answer": "<ваш окончательный ответ здесь>"
}}
""",

    "Japanese": lambda input: f"""質問: {input['input']}
回答選択肢: {input['choices']}。
日本語で考え、その後回答の選択肢の中からひとつを選んでください。次のJSONを埋めてください：
{{
    "reasoning_in_Japanese": "<あなたの日本語での思考ステップ>",
    "final_answer": "<最終的な回答をここに記入>"
}}
""",

    "Korean": lambda input: f"""질문: {input['input']}
답변 선택지: {input['choices']}.
한국어로 생각한 후 답변 선택지 중 하나를 선택하세요. 다음 JSON을 작성하세요:
{{
    "reasoning_in_Korean": "<한국어로 된 추론 단계>",
    "final_answer": "<최종 답변을 여기에 입력>"
}}
""",

    "Vietnamese": lambda input: f"""Câu hỏi: {input['input']}
Các lựa chọn trả lời: {input['choices']}.
Hãy suy nghĩ bằng tiếng Việt, sau đó chọn một trong các lựa chọn trả lời. Điền vào JSON sau:
{{
    "reasoning_in_Vietnamese": "<các bước lập luận bằng tiếng Việt của bạn>",
    "final_answer": "<câu trả lời cuối cùng của bạn tại đây>"
}}
""",

    "Thai": lambda input: f"""คำถาม: {input['input']}
ตัวเลือกคำตอบ: {input['choices']}.
โปรดคิดเป็นภาษาไทย แล้วเลือกคำตอบหนึ่งในตัวเลือก จากนั้นกรอก JSON ดังนี้:
{{
    "reasoning_in_Thai": "<ขั้นตอนการวิเคราะห์ของคุณเป็นภาษาไทย>",
    "final_answer": "<คำตอบสุดท้ายของคุณที่นี่>"
}}
""",

    "Arabic": lambda input: f"""سؤال: {input['input']}
خيارات الإجابة: {input['choices']}.
فكر باللغة العربية ثم اختر أحد خيارات الإجابة. املأ JSON التالي:
{{
    "reasoning_in_Arabic": "<خطواتك في التفكير باللغة العربية>",
    "final_answer": "<إجابتك النهائية هنا>"
}}
""",
}

NR_TRANSLATED_INSTRUCTIONS = {
    "English": lambda input: f"""Question: {input['input']}
Answer choices: {input['choices']}. 

Select one of the answer choices. Fill out the following JSON:
{{
    "final_answer": "<output answer here {input['choices_possibilities']}>"
}}
""",

    "Hindi": lambda input: f"""प्रश्न: {input['input']}
उत्तर विकल्प: {input['choices']}।

एक उत्तर विकल्प चुनें। निम्नलिखित JSON भरें:
{{
    "final_answer": "<यहाँ अंतिम उत्तर लिखें {input['choices_possibilities']}>"
}}
""",

    "Turkish": lambda input: f"""Soru: {input['input']}
Cevap seçenekleri: {input['choices']}.

Bir cevap seçeneği seçin. Aşağıdaki JSON'u doldurun:
{{
    "final_answer": "<çıktı cevabı buraya {input['choices_possibilities']}>"
}}
""",

    "Bengali": lambda input: f"""প্রশ্ন: {input['input']}
উত্তরের বিকল্পসমূহ: {input['choices']}।

একটি উত্তর বিকল্প নির্বাচন করুন। নিচের JSON পূরণ করুন:
{{
    "final_answer": "<এখানে চূড়ান্ত উত্তর লিখুন {input['choices_possibilities']}>"
}}
""",

    "Farsi": lambda input: f"""سؤال: {input['input']}
گزینه‌های پاسخ: {input['choices']}.

یکی از گزینه‌های پاسخ را انتخاب کنید. بخش JSON زیر را پر کنید:
{{
    "final_answer": "<پاسخ نهایی را اینجا وارد کنید {input['choices_possibilities']}>"
}}
""",

    "Chinese": lambda input: f"""问题：{input['input']}
答案选项：{input['choices']}。

选择一个答案选项。请按以下 JSON 格式填写：
{{
    "final_answer": "<在此处填写你的答案 {input['choices_possibilities']}>"
}}
""",

    "French": lambda input: f"""Question : {input['input']}
Options de réponse : {input['choices']}.

Sélectionnez l'une des options de réponse. Veuillez remplir le JSON suivant :
{{
    "final_answer": "<votre réponse finale ici {input['choices_possibilities']}>"
}}
""",

    "Spanish": lambda input: f"""Pregunta: {input['input']}
Opciones de respuesta: {input['choices']}.

Selecciona una de las opciones de respuesta. Completa el siguiente JSON:
{{
    "final_answer": "<tu respuesta final aquí {input['choices_possibilities']}>"
}}
""",

    "Portuguese": lambda input: f"""Pergunta: {input['input']}
Opções de resposta: {input['choices']}.

Selecione uma das opções de resposta. Preencha o seguinte JSON:
{{
    "final_answer": "<sua resposta final aqui {input['choices_possibilities']}>"
}}
""",

    "German": lambda input: f"""Frage: {input['input']}
Antwortmöglichkeiten: {input['choices']}.

Wählen Sie eine der Antwortmöglichkeiten aus. Füllen Sie das folgende JSON aus:
{{
    "final_answer": "<Ihre endgültige Antwort hier {input['choices_possibilities']}>"
}}
""",

    "Italian": lambda input: f"""Domanda: {input['input']}
Scelte di risposta: {input['choices']}.

Seleziona una delle opzioni di risposta. Compila il seguente JSON:
{{
    "final_answer": "<la tua risposta finale qui {input['choices_possibilities']}>"
}}
""",

    "Russian": lambda input: f"""Вопрос: {input['input']}
Варианты ответа: {input['choices']}.

Выберите один из вариантов ответа. Заполните следующий JSON:
{{
    "final_answer": "<ваш окончательный ответ здесь {input['choices_possibilities']}>"
}}
""",

    "Japanese": lambda input: f"""質問: {input['input']}
回答選択肢: {input['choices']}。

回答選択肢の中から1つを選んでください。次のJSONを埋めてください：
{{
    "final_answer": "<最終的な回答をここに記入 {input['choices_possibilities']}>"
}}
""",

    "Korean": lambda input: f"""질문: {input['input']}
답변 선택지: {input['choices']}.

답변 선택지 중 하나를 선택하세요. 다음 JSON을 작성하세요:
{{
    "final_answer": "<최종 답변을 여기에 입력 {input['choices_possibilities']}>"
}}
""",

    "Vietnamese": lambda input: f"""Câu hỏi: {input['input']}
Các lựa chọn trả lời: {input['choices']}.

Chọn một trong các lựa chọn trả lời. Điền vào JSON sau:
{{
    "final_answer": "<câu trả lời cuối cùng của bạn tại đây {input['choices_possibilities']}>"
}}
""",

    "Thai": lambda input: f"""คำถาม: {input['input']}
ตัวเลือกคำตอบ: {input['choices']}.

เลือกหนึ่งคำตอบจากตัวเลือก จากนั้นกรอก JSON ดังนี้:
{{
    "final_answer": "<คำตอบสุดท้ายของคุณที่นี่ {input['choices_possibilities']}>"
}}
""",

    "Arabic": lambda input: f"""سؤال: {input['input']}
خيارات الإجابة: {input['choices']}.

اختر أحد خيارات الإجابة. املأ JSON التالي:
{{
    "final_answer": "<إجابتك النهائية هنا {input['choices_possibilities']}>"
}}
""",
}

